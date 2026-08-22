"""
Week 3 Thursday, Step 2 (docs/PLAN.md Section 3 / ADR-018): does an
ensemble of LR + LightGBM + Elo have anything real to work with?

WHY THIS RUNS BEFORE ANY BLENDING CODE: an ensemble only helps if its
components make DIFFERENT mistakes. Two models that are both wrong on
the same fights, in the same direction, cancel nothing when averaged
— you just get the same wrong answer with extra steps. This module
measures that directly (residual correlation, disagreement rate, an
oracle ceiling) BEFORE Step 3 spends an hour building blend variants
that this file could rule out in twenty minutes.

ALL THREE MODELS SCORED ON THE SAME OUT-OF-FOLD (OOF) ROWS, using
models/cv.py's expanding-window folds — the same discipline
models/calibration.py used for Wednesday's calibrator and
models/feature_deltas.py used for Tuesday's Tier 3 groups. A model's
prediction on a bout it trained on tells you nothing about how it'll
behave on a fight it's never seen; comparing THAT would make every
number in this file meaningless.

THREE DIFFERENT "REFIT PER FOLD" ANSWERS, and that's deliberate, not
inconsistent:
  - LightGBM: yes, refit — reuses models/calibration.py's
    generate_oof_predictions() verbatim (or its cached output).
  - Logistic Regression: yes, refit — including the imputer and
    scaler inside build_logreg_pipeline(). Those are fitted
    transforms too; fitting them on the full train set and predicting
    a fold they were fit on would be a real (if mild) leak.
  - Elo: NO refit needed. Elo isn't a fitted model in the sklearn
    sense — features/elo.py's compute_elo_ratings() is a strictly
    sequential walk where a bout's pre-fight rating is built ONLY
    from that fighter's prior bouts, already point-in-time-safe by
    construction (models/cv.py's own docstring makes this same
    argument for why filtering by date, with no recomputation, is
    sufficient for every column already in `train`). The Elo ratings
    already attached to `train` (self_elo_pre/opp_elo_pre) are just
    read directly and converted to a probability — there is no
    "training set" for Elo to have peeked at.
"""

from pathlib import Path
from collections.abc import Callable, Iterator

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential
from features.elo import expected_score
from models.baselines import build_logreg_pipeline
from models.calibration import (
    CALIB_FIT_MAX_YEAR,
    _logit,
    generate_oof_predictions,
    max_pair_deviation,
)
from models.cv import expanding_year_folds
from models.lightgbm_model import load_tuned_params
from models.metrics import evaluate
from models.tune_lightgbm import FIXED_PARAMS

LGBM_OOF_CACHE_PATH = Path("data/processed/oof_predictions.parquet")
COMPONENT_OOF_PATH = Path("data/processed/ensemble_component_oof.parquet")

KEY_COLS = ["bout_id", "source_corner"]
MODEL_COLS = ["p_lgbm", "p_lr", "p_elo"]


def generate_lr_oof(train: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the same walk-forward exercise as models/calibration.py's
    generate_oof_predictions(), swapping LightGBM for the LR pipeline
    (models/baselines.py's build_logreg_pipeline()).

    Simple version: for every fold, fit a fresh LR pipeline on
    everything through year N, predict year N+1, keep the
    predictions, move forward one year. Twelve folds later you have
    LR's honest opinion on every fight from 2011-2022, none of it
    from a model that had already seen the answer.

    Parameters
    ----------
    train : pd.DataFrame
        Symmetrized train split with Elo attached — same object
        generate_oof_predictions() takes.

    Returns
    -------
    pd.DataFrame — bout_id, val_year, source_corner, y_true, p_lr.
        Same shape/column convention as generate_oof_predictions(),
        with p_raw renamed to p_lr up front so it merges cleanly.
    """
    records = []

    for train_fold, val_fold in expanding_year_folds(train):
        X_tr, y_tr = to_differential(train_fold, verbose=False)
        X_va, y_va = to_differential(val_fold, verbose=False)

        pipeline = build_logreg_pipeline()
        pipeline.fit(X_tr, y_tr)
        p_lr = pipeline.predict_proba(X_va)[:, 1]

        val_year = int(pd.to_datetime(val_fold["event_date"]).dt.year.iloc[0])
        records.append(
            pd.DataFrame(
                {
                    "bout_id": val_fold["bout_id"].to_numpy(),
                    "val_year": val_year,
                    "source_corner": val_fold["source_corner"].to_numpy(),
                    "y_true": y_va.to_numpy(),
                    "p_lr": p_lr,
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def elo_oof_from_train(train: pd.DataFrame) -> pd.DataFrame:
    """
    Reads Elo's pre-fight win probability straight off `train` — no
    fold loop, no refitting. See the module docstring for why that's
    correct here rather than an inconsistency with the other two
    components.

    Parameters
    ----------
    train : pd.DataFrame
        Needs self_elo_pre, opp_elo_pre (attached by
        features.build_lgbm_matrix.build_train_val_with_elo(), per
        models/cv.py's docstring), plus bout_id, source_corner,
        event_date, self_won.

    Returns
    -------
    pd.DataFrame — bout_id, val_year, source_corner, y_true, p_elo.
        Covers every year in `train` (1999-2022); the merge in
        generate_component_oof() naturally restricts this down to
        the 2011-2022 population the fold-based models can cover.
    """
    p_elo = expected_score(
        train["self_elo_pre"].to_numpy(), train["opp_elo_pre"].to_numpy()
    )
    val_year = pd.to_datetime(train["event_date"]).dt.year

    return pd.DataFrame(
        {
            "bout_id": train["bout_id"].to_numpy(),
            "val_year": val_year.to_numpy(),
            "source_corner": train["source_corner"].to_numpy(),
            "y_true": train["self_won"].astype(int).to_numpy(),
            "p_elo": p_elo,
        }
    )


def generate_component_oof(
    train: pd.DataFrame, lgbm_params: dict, use_cached_lgbm_oof: bool = True
) -> pd.DataFrame:
    """
    Aligns all three models' OOF predictions on the same rows.

    Parameters
    ----------
    train : pd.DataFrame
        Symmetrized train split with Elo attached.
    lgbm_params : dict
        Monday's tuned hyperparameters — only used if the cached
        LightGBM OOF file isn't found or use_cached_lgbm_oof=False.
    use_cached_lgbm_oof : bool
        If True (default) and data/processed/oof_predictions.parquet
        exists (written by Wednesday's models/calibration.py run),
        reuse it instead of retraining 12 LightGBM models again.
        Wednesday's Step 5b confirmed this file reproduces exactly on
        rerun, so reuse is safe as long as nothing upstream of it
        (features, tuned params) has changed since. Set False to
        force a fresh regeneration if you're not sure.

    Returns
    -------
    pd.DataFrame — bout_id, val_year, source_corner, y_true, p_lgbm,
        p_lr, p_elo. One row per symmetrized bout row, restricted to
        the population all three sources cover (2011-2022).

    Raises
    ------
    AssertionError if the three independently-derived y_true columns
    disagree after the join — that would mean the join key isn't
    actually unique/aligned the way it's assumed to be, and every
    downstream number in this file would be silently wrong. Better to
    fail loudly here than average over a bug.
    """
    if use_cached_lgbm_oof and LGBM_OOF_CACHE_PATH.exists():
        print(f"loading cached LightGBM OOF from {LGBM_OOF_CACHE_PATH}")
        lgbm = pd.read_parquet(LGBM_OOF_CACHE_PATH)
    else:
        print("generating LightGBM OOF (12 folds, no cache found)...")
        lgbm = generate_oof_predictions(train, lgbm_params)
    lgbm = lgbm.rename(columns={"p_raw": "p_lgbm", "y_true": "y_true_lgbm"})

    print("generating LR OOF (12 folds)...")
    lr = generate_lr_oof(train).rename(columns={"y_true": "y_true_lr"})

    print("computing Elo OOF (no fold loop needed)...")
    elo = elo_oof_from_train(train).rename(columns={"y_true": "y_true_elo"})

    merged = lgbm.merge(
        lr[KEY_COLS + ["p_lr", "y_true_lr"]], on=KEY_COLS, how="inner", validate="one_to_one"
    )
    merged = merged.merge(
        elo[KEY_COLS + ["p_elo", "y_true_elo"]], on=KEY_COLS, how="inner", validate="one_to_one"
    )

    assert (merged["y_true_lgbm"] == merged["y_true_lr"]).all(), (
        "LR y_true disagrees with LightGBM's after the join -- alignment bug."
    )
    assert (merged["y_true_lgbm"] == merged["y_true_elo"]).all(), (
        "Elo y_true disagrees with LightGBM's after the join -- alignment bug."
    )

    merged = merged.rename(columns={"y_true_lgbm": "y_true"}).drop(
        columns=["y_true_lr", "y_true_elo"]
    )

    expected_n = lgbm[KEY_COLS].drop_duplicates().shape[0]
    if len(merged) != expected_n:
        print(
            f"WARNING: merged to {len(merged)} rows, expected {expected_n} "
            f"from LightGBM's OOF alone -- some rows were dropped by the "
            f"join. Check LR/Elo cover the same fold years."
        )

    return merged


def pairwise_correlations(oof: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Pearson correlation of the three models' raw predicted
    probabilities. High correlation here (LR vs. LightGBM especially)
    is expected — both read the same ~32 differential features, just
    through a different function shape. This number alone doesn't
    settle whether blending helps; residual_correlations() below is
    the one that actually does.
    """
    return oof[cols].corr()


def residual_correlations(oof: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Pearson correlation of each model's signed residual
    (y_true - p_pred) against the others'.

    This is the number that actually determines whether an ensemble
    can help. Two models can produce nearly identical PROBABILITIES
    (high pairwise_correlations) and still be worth blending, if their
    ERRORS point in different directions on different fights. The
    reverse is also possible: different-looking probabilities that
    are simply wrong about the same fights in the same way, which
    blending cannot fix. Low or negative correlation here is the
    green light Step 3-6 are looking for; correlation near +1 across
    the board means all three models miss the same fights, and no
    weighting scheme will conjure signal that isn't there.
    """
    residuals = pd.DataFrame({c: oof["y_true"] - oof[c] for c in cols})
    return residuals.corr()


def disagreement_rate(oof: pd.DataFrame, cols: list) -> float:
    """
    Fraction of rows where the three models don't all call the same
    side of the fight (probability > 0.5).

    0.0 means the three models are unanimous on every single fight —
    an ensemble would then just BE whichever model, since there's
    never a disagreement for a blend to resolve in a smarter
    direction. Anything meaningfully above 0 means there's real room
    for a blend to move the needle, one way or the other.
    """
    sides = oof[cols] > 0.5
    unanimous = sides.nunique(axis=1) == 1
    return float((~unanimous).mean())


def per_row_log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Row-level log loss — models/metrics.py's _log_loss averages this
    across all rows; oracle_ceiling() below needs the per-row values
    themselves, to pick a per-row minimum rather than an overall mean.
    """
    p = np.clip(p, eps, 1 - eps)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def oracle_ceiling(oof: pd.DataFrame, cols: list) -> dict:
    """
    The best log loss achievable if, on every single fight, you could
    magically pick whichever of the three models happened to be
    closest to right on THAT fight.

    Not a real, buildable predictor — no ensemble can know in advance
    which model is right on a fight it hasn't seen the outcome of —
    but it's the honest upper bound on what ANY combination of these
    three inputs could ever reach. If this ceiling is barely better
    than the best single model already achieves alone, that's the
    twenty-minute signal that no amount of clever weighting in Step
    3-6 is going to find much daylight, and today becomes a
    documented negative result rather than four hours spent finding
    that out the slow way.

    Returns
    -------
    dict: oracle_log_loss, best_single_model, best_single_log_loss,
        single_log_losses (dict per column), max_possible_gain
        (best_single_log_loss - oracle_log_loss — the absolute
        ceiling on what blending could ever recover).
    """
    y = oof["y_true"].to_numpy()
    losses = np.column_stack(
        [per_row_log_loss(y, oof[c].to_numpy()) for c in cols]
    )
    oracle_loss = float(losses.min(axis=1).mean())

    single_losses = {c: float(losses[:, i].mean()) for i, c in enumerate(cols)}
    best_single = min(single_losses, key=single_losses.get)

    return {
        "oracle_log_loss": oracle_loss,
        "best_single_model": best_single,
        "best_single_log_loss": single_losses[best_single],
        "single_log_losses": single_losses,
        "max_possible_gain": single_losses[best_single] - oracle_loss,
    }

# --- Pre-registered gates (ADR-018), written before any blend was scored ---
# Same discipline as ADR-014/015/016/017. These numbers exist so a
# 0.0004 improvement can't be narrated into a win after the fact.
MIN_LOGLOSS_IMPROVEMENT = 0.002   # Gate A: must beat best single model by this
STACKER_MARGIN = 0.002            # Gate B: stacker must beat best fixed blend by this
MAX_PAIR_DEVIATION_RATIO = 1.5    # Gate C: vs. best single model's own deviation
MAX_ECE_REGRESSION = 0.005        # Gate D: holdout ECE may not worsen by more

WEIGHT_GRID_STEP = 0.05           # resolution of the weight simplex search


def blend_prob_mean(probs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Weighted average in PROBABILITY space.

    Simple version: if LightGBM says 0.70 and Elo says 0.55, this
    says 0.625. Straightforward, but it has a known downside — it
    pulls everything toward 0.5. When two models are confident and
    both right, averaging their probabilities throws away some of
    that confidence, and log loss specifically punishes you for
    hedging on calls you got right. That's why blend_logit_mean()
    below is the more principled default.

    Included anyway as the naive baseline: if the fancier version
    can't beat plain averaging, that's worth knowing.
    """
    probs = np.asarray(probs, dtype=float)
    if weights is None:
        weights = np.ones(probs.shape[1]) / probs.shape[1]
    return np.clip(probs @ weights, 1e-6, 1 - 1e-6)


def blend_logit_mean(probs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Weighted average in LOG-ODDS space, mapped back through a sigmoid.

    Simple version: instead of averaging the three answers, this
    averages the three models' *evidence* and then converts back to a
    probability. Two models each independently leaning "70%" combine
    into something more confident than 70%, rather than being averaged
    back down to exactly 70% — which matches how independent evidence
    actually stacks, and is what log loss rewards.

    Also preserves ADR-004's symmetry property cleanly: the logit of
    (1-p) is exactly the negative logit of p, so flipping every input's
    corner flips the blended logit's sign and the output correctly
    becomes 1 - p. Probability-space averaging happens to preserve this
    too; a fitted stacker with an intercept does NOT (see fit_stacker).
    """
    probs = np.asarray(probs, dtype=float)
    if weights is None:
        weights = np.ones(probs.shape[1]) / probs.shape[1]
    z = _logit(probs) @ weights
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def _simplex_grid(n: int, step: float = WEIGHT_GRID_STEP) -> Iterator[np.ndarray]:
    """
    Every non-negative weight vector of length n summing to 1.0, on a
    grid of the given step size.

    Weights are constrained non-negative on purpose. An unconstrained
    fit can hand a model a NEGATIVE weight — mathematically it might
    lower the fit-set loss, but "subtract Elo's opinion" is almost
    always the optimizer exploiting noise rather than finding a real
    inverse signal, and it won't survive to the holdout. Constraining
    to the simplex keeps the blend interpretable as "how much do I
    trust each model."
    """
    ticks = int(round(1.0 / step))
    if n == 1:
        yield np.array([1.0])
        return
    for i in range(ticks + 1):
        head = i * step
        for rest in _simplex_grid(n - 1, step):
            yield np.concatenate([[head], rest * (1.0 - head)])


def fit_blend_weights(
    fit: pd.DataFrame, cols: list, blend_fn: Callable = blend_logit_mean
) -> np.ndarray:
    """
    Grid-searches the weight simplex for the combination with the
    lowest log loss ON THE FIT FOLDS ONLY (2011-2020).

    The holdout (2021-2022) is never touched here. Its only job is
    grading the finished candidates in evaluate_blends() — exactly the
    same fit/holdout separation models/calibration.py used for the
    calibrator, one level up from models/tune_lightgbm.py's val
    protection.

    Grid search rather than scipy.optimize: with three weights and a
    0.05 step this is a few hundred evaluations of a vectorized
    function -- fast, deterministic, and there's no optimizer
    configuration to defend in an interview.

    Returns
    -------
    np.ndarray — weights in the same order as `cols`.
    """
    y = fit["y_true"].to_numpy()
    P = fit[cols].to_numpy()

    best_w, best_loss = None, np.inf
    for w in _simplex_grid(len(cols)):
        loss = float(per_row_log_loss(y, blend_fn(P, w)).mean())
        if loss < best_loss:
            best_w, best_loss = w, loss

    return best_w


def fit_stacker(fit: pd.DataFrame, cols: list, fit_intercept: bool = False):
    """
    Level-2 logistic regression on the three component LOG-ODDS.

    Simple version: instead of you choosing how much to trust each
    model, a small model learns it from the data — and unlike the
    weight grid, it isn't constrained to non-negative weights summing
    to 1, so it has genuinely more freedom. That freedom is also the
    only real overfit risk in today's candidate set, which is exactly
    what Gate B is guarding: it has to beat the simpler fixed-weight
    blend by a real margin, not just fit the fit-folds better.

    fit_intercept=False by default, and that is a SYMMETRY decision,
    not a modeling preference. An intercept adds a constant shift to
    the blended log-odds. Flip a bout's corners and every input logit
    negates, so the fitted terms negate too -- but the intercept
    doesn't, and P(self) + P(opp) stops equalling 1.0. That's the same
    ADR-004 invariant isotonic calibration failed in ADR-017's Gate C.
    models/baselines.py's build_logreg_pipeline() sets
    fit_intercept=False for precisely this reason one level down.

    The intercept version is still scored as a separate candidate so
    Gate C has something real to catch -- a demonstrated failure is
    better evidence than an asserted one.
    """
    lr = LogisticRegression(fit_intercept=fit_intercept, max_iter=1000, C=1e10)
    lr.fit(_logit(fit[cols].to_numpy()), fit["y_true"].to_numpy())
    return lr


def _stacker_predict(model, probs: np.ndarray) -> np.ndarray:
    """Applies a fitted stacker to component probabilities."""
    return np.clip(model.predict_proba(_logit(np.asarray(probs)))[:, 1], 1e-6, 1 - 1e-6)


def evaluate_blends(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Builds every candidate blend on the fit folds, scores all of them
    (plus the three single models) on the holdout.

    Candidate set, in increasing order of complexity:
      1. each single model alone — the thing a blend has to beat
      2. equal-weight probability average (naive baseline)
      3. equal-weight logit average
      4. fitted-weight logit blend, all three models
      5. fitted-weight logit blend, LightGBM + Elo ONLY
      6. stacker, no intercept
      7. stacker, with intercept (expected to fail Gate C)

    Candidate 5 exists because of Step 2's diagnostic: p_lgbm and p_lr
    correlate at 0.845 (same 32 features, different function shape)
    while Elo correlates at only 0.49 with either. If the gain is
    coming from Elo's independence rather than from LR, the two-model
    blend should match or beat the three-model one -- and it would be
    a cheaper artifact for Week 4's inference path (two models loaded
    per upcoming bout instead of three). "LR added nothing once
    LightGBM was present" is a real finding worth testing for
    directly, not inferring.

    Returns
    -------
    pd.DataFrame — method, n_components, accuracy, log_loss, brier,
        ece, max_pair_dev, weights (string, for the ADR table).
    """
    fit = oof[oof["val_year"] <= CALIB_FIT_MAX_YEAR]
    hold = oof[oof["val_year"] > CALIB_FIT_MAX_YEAR]

    y_hold = hold["y_true"].to_numpy()
    bout_hold = hold["bout_id"].to_numpy()
    rows = []

    def _record(method: str, p: np.ndarray, n_comp: int, weights: str = ""):
        res = evaluate(y_hold, p, name=method)
        rows.append({
            "method": method,
            "n_components": n_comp,
            "accuracy": res["accuracy"],
            "log_loss": res["log_loss"],
            "brier": res["brier"],
            "ece": res["ece"],
            "max_pair_dev": max_pair_deviation(bout_hold, p),
            "weights": weights,
        })

    # 1. single models
    for c in MODEL_COLS:
        _record(f"single_{c}", hold[c].to_numpy(), 1)

    P_hold_all = hold[MODEL_COLS].to_numpy()

    # 2-3. equal-weight blends
    _record("equal_prob_mean", blend_prob_mean(P_hold_all), 3, "equal")
    _record("equal_logit_mean", blend_logit_mean(P_hold_all), 3, "equal")

    # 4. fitted weights, all three
    w3 = fit_blend_weights(fit, MODEL_COLS)
    _record(
        "weighted_logit_3", blend_logit_mean(P_hold_all, w3), 3,
        ", ".join(f"{c}={v:.2f}" for c, v in zip(MODEL_COLS, w3)),
    )

    # 5. fitted weights, LightGBM + Elo only
    two = ["p_lgbm", "p_elo"]
    w2 = fit_blend_weights(fit, two)
    _record(
        "weighted_logit_lgbm_elo", blend_logit_mean(hold[two].to_numpy(), w2), 2,
        ", ".join(f"{c}={v:.2f}" for c, v in zip(two, w2)),
    )

    # 6-7. stackers
    for label, use_intercept in (("stacker_no_intercept", False), ("stacker_intercept", True)):
        model = fit_stacker(fit, MODEL_COLS, fit_intercept=use_intercept)
        _record(
            label, _stacker_predict(model, P_hold_all), 3,
            ", ".join(f"{c}={v:.2f}" for c, v in zip(MODEL_COLS, model.coef_[0])),
        )

    out = pd.DataFrame(rows)
    best_single = out[out["n_components"] == 1]["log_loss"].min()
    out["d_log_loss"] = out["log_loss"] - best_single
    return out


def select_blend(comparison: pd.DataFrame) -> str | None:
    """
    Applies ADR-018's four pre-registered gates. Returns the chosen
    method name, or None (meaning: ship the tuned LightGBM alone,
    unchanged, and today is a documented negative result).

    Gate order matters. Gate C (symmetry) runs FIRST and is
    structural — a blend that contradicts itself about the same fight
    depending on which corner it read is disqualified regardless of
    how good its log loss looks, same as isotonic in ADR-017. Gates A
    and D then handle magnitude, and Gate B breaks the tie in favor of
    the simpler candidate.

    Note the comparison target throughout is the BEST SINGLE MODEL'S
    OWN HOLDOUT NUMBERS, not any figure from val. Sizing a gate to a
    different population than the one it's applied to was ADR-017's
    one process mistake; not repeating it.
    """
    singles = comparison[comparison["n_components"] == 1]
    best_single = singles.loc[singles["log_loss"].idxmin()]
    cand = comparison[comparison["n_components"] > 1].copy()

    # --- Gate C: symmetry (ADR-004) ---
    limit = MAX_PAIR_DEVIATION_RATIO * best_single["max_pair_dev"]
    failed = cand[cand["max_pair_dev"] > limit]
    if not failed.empty:
        print(
            f"Gate C failures (limit {limit:.6f} = {MAX_PAIR_DEVIATION_RATIO}x "
            f"{best_single['method']}'s {best_single['max_pair_dev']:.6f}): "
            + ", ".join(
                f"{m} ({d:.6f}, {d / best_single['max_pair_dev']:.2f}x)"
                for m, d in zip(failed["method"], failed["max_pair_dev"])
            )
        )
    cand = cand[cand["max_pair_dev"] <= limit]

    # --- Gate D: calibration guard ---
    ece_limit = best_single["ece"] + MAX_ECE_REGRESSION
    failed_d = cand[cand["ece"] > ece_limit]
    if not failed_d.empty:
        print(
            f"Gate D failures (holdout ECE > {ece_limit:.4f}): "
            + ", ".join(f"{m} ({e:.4f})" for m, e in zip(failed_d["method"], failed_d["ece"]))
        )
    cand = cand[cand["ece"] <= ece_limit]

    # --- Gate A: must actually beat the best single model ---
    cand = cand[cand["log_loss"] <= best_single["log_loss"] - MIN_LOGLOSS_IMPROVEMENT]
    if cand.empty:
        print(
            f"Gate A: nothing beat {best_single['method']} "
            f"({best_single['log_loss']:.6f}) by {MIN_LOGLOSS_IMPROVEMENT}."
        )
        return None

    # --- Gate B: stacker must earn its extra complexity ---
    fixed = cand[~cand["method"].str.startswith("stacker")]
    stackers = cand[cand["method"].str.startswith("stacker")]

    if fixed.empty:
        return stackers.loc[stackers["log_loss"].idxmin(), "method"]

    best_fixed = fixed.loc[fixed["log_loss"].idxmin()]
    if not stackers.empty:
        best_stacker = stackers.loc[stackers["log_loss"].idxmin()]
        if best_stacker["log_loss"] < best_fixed["log_loss"] - STACKER_MARGIN:
            return best_stacker["method"]
        print(
            f"Gate B: stacker ({best_stacker['log_loss']:.6f}) did not beat "
            f"{best_fixed['method']} ({best_fixed['log_loss']:.6f}) by "
            f"{STACKER_MARGIN} -- simpler blend wins."
        )
    return best_fixed["method"]


if __name__ == "__main__":
    if COMPONENT_OOF_PATH.exists():
        print(f"loading component OOF from {COMPONENT_OOF_PATH}")
        oof = pd.read_parquet(COMPONENT_OOF_PATH)
    else:
        train, _ = build_train_val_with_elo()  # val NOT read in this block
        oof = generate_component_oof(train, {**FIXED_PARAMS, **load_tuned_params()})
        COMPONENT_OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(COMPONENT_OOF_PATH, index=False)

    n_fit = (oof["val_year"] <= CALIB_FIT_MAX_YEAR).sum()
    print(
        f"fit folds (2011-{CALIB_FIT_MAX_YEAR}): {n_fit} rows | "
        f"holdout ({CALIB_FIT_MAX_YEAR + 1}-2022): {len(oof) - n_fit} rows"
    )

    print("\n=== blend candidates, scored on the train-internal holdout ===")
    comparison = evaluate_blends(oof)
    print(comparison.to_string(index=False))

    print("\n=== applying ADR-018 gates ===")
    chosen = select_blend(comparison)
    print(f"\nchosen: {chosen}")

    comparison.to_csv("data/processed/ensemble_holdout_comparison.csv", index=False)
    print("wrote data/processed/ensemble_holdout_comparison.csv")

