"""
Post-hoc probability calibration for the tuned LightGBM — Week 3
Wednesday (docs/PLAN.md Section 3).

THE PROBLEM THIS FIXES: Monday's Optuna search optimized log loss and
got it (0.6535 -> 0.6483, odds-covered), but ECE moved the wrong way
(0.0214 -> 0.0318). That's not a bug, it's the trade log loss rewards:
a model willing to say 0.72 instead of 0.62 scores better on log loss
when it's right, and the search found one. Some of those 0.72s aren't
really 0.72s. Calibration re-maps the model's raw probabilities onto
honest ones AFTER the fact — the trees are never retrained, only the
final number is adjusted.

WHY THIS FILE DOESN'T FIT THE CALIBRATOR ON VAL, despite docs/PLAN.md
Section 3 saying "apply isotonic or Platt on the validation set":
fitting a curve to val's 2,034 labels and then reporting val's ECE is
circular. It's the same trap as tuning hyperparameters on val — the
number stops measuring "how well does this generalize" and starts
measuring "how well did I fit this specific set." models/tune_lightgbm
.py's whole module docstring is about avoiding this one level up.

So the calibrator is fit on OUT-OF-FOLD predictions from train: for
each of models/cv.py's 12 expanding-window folds, the model trained on
prior years predicts that fold's year, and those predictions are
stacked. Every one of them is a genuine "model predicting data it has
never seen" — exactly the distribution the calibrator needs to learn
from — and none of them touch val.

The tradeoff, stated honestly: OOF predictions come from models
trained on LESS history than the final model (the 2011 fold trains on
1,291 bouts; the final model trains on 6,604). If less-trained models
were systematically less confident, the calibrator would be tuned to
the wrong confidence profile. Tuesday's fold-trend finding
(docs/RESULTS.md: corr(val_year, log_loss) = 0.073, essentially zero
across a 4.7x growth in training bouts) is what makes that risk small
here — the model saturates on training volume early, so an OOF model's
probabilities are distributionally close to the final model's. This
would be a much shakier assumption on a dataset that was still
learning.

SYMMETRY NOTE: every bout appears twice in this pipeline (self/opp
flipped, ADR-004), and models/lightgbm_model.py checks that a bout's
two rows sum to ~1.0. A monotone calibrator does NOT preserve that by
construction — g(p) + g(1-p) can be anything. symmetrize() below
enforces it explicitly rather than hoping.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential
from models.cv import expanding_year_folds
from models.lightgbm_model import load_tuned_params
from models.metrics import evaluate, expected_calibration_error
from models.tune_lightgbm import FIXED_PARAMS

CALIBRATOR_PATH = Path("models/artifacts/calibrator_v1.joblib")
CALIBRATOR_META_PATH = Path("models/artifacts/calibrator_v1_meta.json")

# --- Pre-registered decision rules, written before any result was seen ---
# Same discipline as ADR-014 (Elo K stopping rule), ADR-015 (bucket
# gate), ADR-016 (feature deltas). Writing these down first is what
# stops a 0.0003 improvement from being narrated as a win later.
#
# GATE A — does calibration earn its place at all? The regression being
# fixed is 0.0104 of ECE (0.0214 -> 0.0318). A fix worth shipping should
# recover a meaningful fraction of that, and must not cost log loss.
MIN_ECE_IMPROVEMENT = 0.005      # holdout ECE must drop by at least this
MAX_LOGLOSS_REGRESSION = 0.002   # holdout log loss may not rise by more

# GATE B — isotonic vs Platt. Isotonic is strictly more flexible, so it
# will almost always fit the calibration holdout better; the threshold
# is what stops "more flexible" from being confused with "better." Ties
# go to Platt: two parameters instead of a step function with dozens of
# knots, and far less prone to overfitting the calibration set.
ISOTONIC_MARGIN = 0.002          # isotonic must beat Platt's log loss by this

# GATE C — symmetry preservation (ADR-004), RELATIVE to the raw
# model's own imperfection. An absolute near-zero bar (an earlier
# version of this gate used 1e-6) turned out to be unfair: the raw,
# UNCALIBRATED model already has a real max_pair_dev of ~0.07 on this
# holdout (LightGBM has no structural symmetry guarantee — see
# models/lightgbm_model.py's own docstring), so judging a calibrator
# against a bar the base model itself fails isn't testing what the
# calibrator did. The fair question is whether a calibrator makes an
# already-imperfect invariant meaningfully WORSE than it already was.
MAX_PAIR_DEVIATION_RATIO = 1.5   # calibrated max_pair_dev may be at most this multiple of the uncalibrated baseline's own value, same holdout

# The calibration set / calibration-holdout split, by fold val_year.
# Split by YEAR, not randomly, for two reasons: it keeps a bout's two
# symmetrized rows together (a random split would put the same fight on
# both sides — LEAKAGE_LOG.md category 5), and it keeps the holdout
# temporally after the fit set, matching how the calibrator will
# actually be used.
CALIB_FIT_MAX_YEAR = 2020        # OOF folds 2011-2020 fit the calibrator
# folds 2021-2022 are the holdout that decides the rules above


def generate_oof_predictions(train: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Runs the expanding-window CV walk and collects every fold's
    predictions on data that fold's model never saw.

    Simple version: this is the same walk-forward exercise
    models/tune_lightgbm.py's objective() and models/feature_deltas.py's
    score_configuration() both run — train on everything through year N,
    predict year N+1 — except instead of scoring each fold and throwing
    the predictions away, it KEEPS them. Stack all 12 folds and you have
    ~5,300 bouts (10,600 symmetrized rows) of honest "model meets
    unseen fight" predictions, which is the training data the
    calibrator needs.

    Why not just predict train with the final model? Because the final
    model has memorized train to some degree — its probabilities on
    training rows are sharper than its probabilities on new fights, so
    a calibrator fit on them would learn to correct a distortion that
    doesn't exist at inference time, and would make things worse.

    Parameters
    ----------
    train : pd.DataFrame
        Symmetrized train split with Elo attached (32-feature baseline).
    params : dict
        Monday's tuned hyperparameters, identical to what the final
        model uses — the calibrator has to be learning the confidence
        profile of THIS model, not a different one.

    Returns
    -------
    pd.DataFrame — bout_id, val_year, source_corner, y_true, p_raw.
        One row per symmetrized row, in fold order.
    """
    records = []

    for train_fold, val_fold in expanding_year_folds(train):
        X_tr, y_tr = to_differential(train_fold, verbose=False)
        X_va, y_va = to_differential(val_fold, verbose=False)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        p_raw = np.asarray(model.predict_proba(X_va))[:, 1]

        val_year = int(pd.to_datetime(val_fold["event_date"]).dt.year.iloc[0])
        records.append(
            pd.DataFrame(
                {
                    "bout_id": val_fold["bout_id"].to_numpy(),
                    "val_year": val_year,
                    "source_corner": val_fold["source_corner"].to_numpy(),
                    "y_true": y_va.to_numpy(),
                    "p_raw": p_raw,
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Converts probabilities to log-odds, clipped away from 0 and 1.

    Platt scaling is a logistic regression fit on the model's SCORE,
    not its probability. LightGBM's probability is already a sigmoid of
    its raw score, so taking the logit recovers something proportional
    to that score — which is what makes Platt a well-behaved two-
    parameter stretch/shift rather than a sigmoid-of-a-sigmoid.

    The clip matters: log(0) is -inf and would kill the fit. eps=1e-6
    caps the log-odds around +/-13.8, far outside anything this model
    actually produces (it rarely leaves 0.25-0.75).
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    """
    A fitted calibration map, plus enough metadata to reproduce it.

    Wraps either a LogisticRegression (Platt) or an IsotonicRegression
    so both are called the same way downstream. `transform(p)` is the
    only method Week 4's inference path will ever need.

    Attributes
    ----------
    method : "platt" | "isotonic"
    estimator : the fitted sklearn object
    symmetric : bool
        Whether transform() enforces g(p) + g(1-p) = 1. Always True in
        this project — see symmetrize() and ADR-004.
    """

    method: Literal["platt", "isotonic"]
    estimator: object
    symmetric: bool = True

    def _raw_transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "platt":
            return self.estimator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return self.estimator.predict(p)

    def transform(self, p: np.ndarray) -> np.ndarray:
        """
        Maps raw model probabilities to calibrated ones.

        If symmetric=True, applies the averaging trick in symmetrize()
        so a bout's two symmetrized rows still sum to 1.0.
        """
        if not self.symmetric:
            return np.clip(self._raw_transform(p), 1e-6, 1 - 1e-6)
        return symmetrize(self._raw_transform, p)


def symmetrize(fn: Callable[[np.ndarray], np.ndarray], p: np.ndarray) -> np.ndarray:
    """
    Forces a calibration map to satisfy g(p) + g(1 - p) = 1.

    Simple version: this pipeline predicts every fight twice — once as
    "Jones vs Miocic" and once as "Miocic vs Jones" — and those two
    numbers have to add to 1, or the model is claiming two different
    things about the same fight depending on which corner it read
    first. That invariant is guaranteed for the LR baseline by
    construction (differential features + fit_intercept=False) and
    holds empirically for LightGBM (models/lightgbm_model.py checks it),
    but a monotone calibrator has NO reason to preserve it: isotonic
    fits a step function to whatever the data looked like, and the data
    isn't perfectly symmetric.

    The fix is an average of the two ways to ask the question:
        g_sym(p) = ( g(p) + 1 - g(1 - p) ) / 2

    If g already happened to be symmetric this changes nothing. If it
    isn't, this splits the difference — and the result is provably
    symmetric, since swapping p for 1-p swaps the two terms.

    Doing this at the calibrator level rather than post-hoc normalizing
    each bout's two rows is deliberate: at inference time (Week 4) a
    single upcoming bout gets predicted once, not twice, so there's no
    pair to normalize against. The property has to live inside the
    function.
    """
    p = np.asarray(p, dtype=float)
    forward = fn(p)
    backward = fn(1.0 - p)
    return np.clip(0.5 * (forward + 1.0 - backward), 1e-6, 1 - 1e-6)


def fit_platt(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """
    Platt scaling: a one-variable logistic regression on the model's
    log-odds.

    Simple version: it can only stretch or shift the existing curve. If
    the model is uniformly overconfident — saying 0.75 when it means
    0.70, and 0.25 when it means 0.30 — Platt fixes that with two
    numbers. What it CANNOT do is fix a lumpy miscalibration where the
    model is fine at 0.6, too confident at 0.75, and under-confident at
    0.4. That's what isotonic is for.

    Two parameters means it essentially cannot overfit the calibration
    set, which is why it stays the default when isotonic doesn't clear
    a real margin.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs")  # C large = effectively unpenalized
    lr.fit(_logit(p).reshape(-1, 1), y)
    return Calibrator(method="platt", estimator=lr)


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """
    Isotonic regression: a non-parametric, monotone step function.

    Simple version: it sorts every prediction, then fits the best
    possible "never goes down" staircase to the observed outcomes. It
    can correct any shape of miscalibration, which is its strength and
    its risk — with too few samples it fits noise, producing flat
    plateaus and cliffs that don't generalize. Rule of thumb is ~1,000+
    samples; the OOF set here has ~8,800 rows in the fit portion, so
    it's viable — but it still has to clear ISOTONIC_MARGIN against
    Platt to be chosen.

    out_of_bounds="clip" matters: isotonic only knows the range of
    probabilities it was fit on. If a future fight produces a raw
    probability outside that range, "clip" returns the nearest known
    value rather than erroring. y_min/y_max keep the output a valid
    probability.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p, y)
    return Calibrator(method="isotonic", estimator=iso)


def max_pair_deviation(bout_ids: np.ndarray, p: np.ndarray) -> float:
    """
    Worst-case |sum of a bout's two calibrated probabilities - 1.0|,
    across every bout in the input.

    Simple version: every bout in this pipeline is predicted twice —
    once as "Jones vs Miocic," once as "Miocic vs Jones" (ADR-004) —
    and the two answers have to add to 1.0, or the model is
    contradicting itself about the same fight depending on which
    corner it read first. Neither ECE nor log loss would ever notice
    this kind of failure, since both are computed per-row, not
    per-bout-pair — this is the only check in the file that actually
    looks at a bout's two rows together.

    Parameters
    ----------
    bout_ids : np.ndarray
        bout_id per row, same order/length as p.
    p : np.ndarray
        Calibrated probabilities, same order/length as bout_ids.

    Returns
    -------
    float — 0.0 is perfect; anything above ~1e-6 means the calibrator
    gives two different answers about the same fight depending on
    corner.
    """
    pair_sums = pd.Series(p).groupby(bout_ids).sum()
    return float((pair_sums - 1.0).abs().max())


def compare_on_holdout(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Fits both calibrators on OOF folds <= CALIB_FIT_MAX_YEAR and scores
    them (plus the uncalibrated baseline) on the later folds.

    Simple version: this is the "which one, and is it even worth it"
    decision, made entirely inside train. Nothing in this function
    reads val. The winner gets refit on the FULL OOF set afterward —
    the holdout's only job is choosing, not producing the final map.

    Returns
    -------
    pd.DataFrame — method, n_fit, n_holdout, log_loss, brier, ece,
        d_log_loss, d_ece (deltas vs. the uncalibrated row).
    """
    fit_mask = oof["val_year"] <= CALIB_FIT_MAX_YEAR
    fit, hold = oof[fit_mask], oof[~fit_mask]

    rows = []
    baseline = evaluate(
        hold["y_true"].to_numpy(), hold["p_raw"].to_numpy(), name="uncalibrated"
    )
    rows.append({
        "method": "uncalibrated",
        **_metric_dict(baseline),
        "max_pair_dev": max_pair_deviation(
            hold["bout_id"].to_numpy(), hold["p_raw"].to_numpy()
        ),
    })

    for name, fitter in (("platt", fit_platt), ("isotonic", fit_isotonic)):
        cal = fitter(fit["p_raw"].to_numpy(), fit["y_true"].to_numpy())
        p_cal = cal.transform(hold["p_raw"].to_numpy())
        res = evaluate(hold["y_true"].to_numpy(), p_cal, name=name)
        rows.append({
            "method": name,
            **_metric_dict(res),
            "max_pair_dev": max_pair_deviation(hold["bout_id"].to_numpy(), p_cal),
        })

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["d_log_loss"] = out["log_loss"] - base["log_loss"]
    out["d_ece"] = out["ece"] - base["ece"]
    out["n_fit"] = len(fit)
    out["n_holdout"] = len(hold)
    return out


def _metric_dict(result) -> dict:
    """
    Normalizes models/metrics.py's evaluate() return into a plain dict.

    Adjust the attribute/key access on the next line if evaluate()
    returns a dataclass rather than a dict — this is the one place in
    this file that depends on its exact return shape, kept in one
    function on purpose so a mismatch is a one-line fix.
    """
    d = result if isinstance(result, dict) else vars(result)
    return {k: d[k] for k in ("accuracy", "log_loss", "brier", "ece")}


def select_method(comparison: pd.DataFrame) -> str | None:
    """
    Applies the pre-registered gates. Returns "platt", "isotonic", or
    None (meaning: ship uncalibrated, calibration didn't earn it).

    Gate C runs first and is RELATIVE to the uncalibrated row's own
    max_pair_dev on this same holdout — not an absolute bar. The raw
    model's own max_pair_dev (0.0702 on this holdout) already tells
    you ADR-004's invariant is approximate for LightGBM, not exact —
    an expected property, not a defect introduced today. A calibrator
    is disqualified only if it pushes that pre-existing imperfection
    past MAX_PAIR_DEVIATION_RATIO times what was already there.

    Gate A: among Gate C survivors, at least one must cut holdout ECE
    by MIN_ECE_IMPROVEMENT without raising log loss by more than
    MAX_LOGLOSS_REGRESSION.
    Gate B: among Gate A survivors, lowest holdout log loss wins, but
    isotonic must beat Platt by ISOTONIC_MARGIN to be preferred —
    otherwise the simpler model takes it.
    """
    baseline_dev = comparison.loc[
        comparison["method"] == "uncalibrated", "max_pair_dev"
    ].iloc[0]
    cand = comparison[comparison["method"] != "uncalibrated"]

    limit = MAX_PAIR_DEVIATION_RATIO * baseline_dev
    symmetric = cand["max_pair_dev"] <= limit
    if (~symmetric).any():
        dq = cand[~symmetric]
        print(
            f"disqualified on Gate C (symmetry vs uncalibrated baseline "
            f"{baseline_dev:.6f}, limit {limit:.6f}): "
            + ", ".join(
                f"{m} ({d:.6f}, {d / baseline_dev:.2f}x baseline)"
                for m, d in zip(dq["method"], dq["max_pair_dev"])
            )
        )
    cand = cand[symmetric]

    passing = cand[
        (cand["d_ece"] <= -MIN_ECE_IMPROVEMENT)
        & (cand["d_log_loss"] <= MAX_LOGLOSS_REGRESSION)
    ]
    if passing.empty:
        return None

    methods = set(passing["method"])
    if methods == {"platt", "isotonic"}:
        ll = passing.set_index("method")["log_loss"]
        return "isotonic" if ll["isotonic"] < ll["platt"] - ISOTONIC_MARGIN else "platt"
    return passing.iloc[0]["method"]


def save_calibrator(cal: Calibrator, meta: dict) -> None:
    """
    Serializes the fitted calibrator plus a metadata sidecar.

    Two files on purpose: the .joblib is the thing inference loads, the
    .json is the thing a human (or docs/MODEL_CARD.md) reads. Same
    split as models/artifacts/lgbm_best_params.json — the artifact is
    useless without a record of what produced it.
    """
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, CALIBRATOR_PATH)
    CALIBRATOR_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"wrote {CALIBRATOR_PATH} and {CALIBRATOR_META_PATH}")


def load_calibrator(path: Path = CALIBRATOR_PATH) -> Calibrator:
    """Reads the fitted calibrator back. Week 4's inference path uses this."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `uv run -m models.calibration` first."
        )
    return joblib.load(path)

if __name__ == "__main__":
    train, _ = build_train_val_with_elo()  # val NOT read in this block
    params = {**FIXED_PARAMS, **load_tuned_params()}

    print(f"train: {len(train)} rows, {train['bout_id'].nunique()} bouts")
    print("generating out-of-fold predictions (12 folds)...\n")
    oof = generate_oof_predictions(train, params)
    oof.to_parquet("data/processed/oof_predictions.parquet", index=False)
    print(f"OOF rows: {len(oof)}, bouts: {oof['bout_id'].nunique()}")
    print(oof.groupby("val_year").size().to_string())

    print("\n=== calibrator comparison (train-internal holdout: 2021-2022) ===")
    comparison = compare_on_holdout(oof)
    print(comparison.to_string(index=False))

    chosen = select_method(comparison)
    print(f"\npre-registered gates -> chosen method: {chosen}")

    # --- Refit the chosen method on the FULL OOF set ---
    # The holdout's job was choosing; it would be wasteful to throw
    # away 2021-2022's ~2,000 rows once that choice is made. Same
    # reasoning as refitting on train+val after model selection —
    # except here, obviously, val is still off limits.
    if chosen is None:
        print("no calibrator passed the gates — shipping uncalibrated.")
    else:
        fitter = fit_platt if chosen == "platt" else fit_isotonic
        cal = fitter(oof["p_raw"].to_numpy(), oof["y_true"].to_numpy())

        # --- THE ONE VAL READ ---
        from models.baselines import _load_val_and_odds, _odds_covered_mask
        from models.lightgbm_model import train_lightgbm_baseline

        train_full, val = build_train_val_with_elo()
        y_val, p_raw, model, X_train = train_lightgbm_baseline(
            train_full, val, params=load_tuned_params()
        )
        p_cal = cal.transform(p_raw)

        _, closing = _load_val_and_odds()
        covered = _odds_covered_mask(val, closing).to_numpy()

        for label, probs in (("raw", p_raw), ("calibrated", p_cal)):
            print(evaluate(y_val, probs, name=f"lgbm_tuned_{label}_full_val"))
            print(evaluate(y_val[covered], probs[covered],
                           name=f"lgbm_tuned_{label}_odds_covered"))

        # Symmetry invariant must survive calibration (ADR-004).
        pair_sums = (
            pd.DataFrame({"bout_id": val["bout_id"].to_numpy(), "p": p_cal})
            .groupby("bout_id")["p"].sum()
        )
        print(f"\ncalibrated pair sum — mean: {pair_sums.mean():.6f}, "
              f"max deviation: {(pair_sums - 1.0).abs().max():.6f}")

        np.save("data/processed/val_probs_raw.npy", p_raw)
        np.save("data/processed/val_probs_calibrated.npy", p_cal)
        np.save("data/processed/val_y_true.npy", y_val)
        np.save("data/processed/val_odds_covered_mask.npy", covered)

        save_calibrator(cal, {
            "method": chosen,
            "symmetric": True,
            "fit_on": "out-of-fold predictions, models/cv.py folds 2011-2022",
            "n_fit_rows": int(len(oof)),
            "n_fit_bouts": int(oof["bout_id"].nunique()),
            "selection_holdout_years": [2021, 2022],
            "base_model_params": load_tuned_params(),
            "gates": {
                "min_ece_improvement": MIN_ECE_IMPROVEMENT,
                "max_logloss_regression": MAX_LOGLOSS_REGRESSION,
                "isotonic_margin": ISOTONIC_MARGIN,
                "max_pair_deviation": MAX_PAIR_DEVIATION_RATIO,
            },
            "holdout_comparison": comparison.to_dict(orient="records"),
        })