"""
Optuna hyperparameter search for LightGBM — Week 3 Monday
(docs/PLAN.md Section 3: "Sensible defaults first, then optuna with
time-series CV (expanding window, never random KFold)").

WHY OPTUNA IS ITSELF A LEAKAGE SURFACE, and how this file defends
against it: Optuna's whole job is to try hundreds of hyperparameter
combinations and keep whichever scored best. Point it at the real val
set and it WILL find a combination that scores well on those specific
1,017 bouts — including by luck, not skill. Val would quietly stop
being an honest check and become just another thing tuned against,
the same trap docs/PLAN.md Section 3 warns about for the test set
("if you tune against it, it becomes another validation set and
you've lost your only honest estimate"). So every score Optuna sees
in this file comes from expanding-window folds carved out of TRAIN
ONLY (models/cv.py). The real val set is not read anywhere in this
file — it's touched exactly once, in Step 8, after tuning is finished
and the search is closed.

Optimizes LOG LOSS, not accuracy. Accuracy only cares whether the
probability landed on the right side of 0.50; log loss cares how
confident it was and punishes confident mistakes hard. Since this
project's actual bar (Section 0.1) is calibration, not raw accuracy,
tuning against accuracy would be optimizing for the wrong thing.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential
from models.cv import expanding_year_folds

BEST_PARAMS_PATH = Path("models/artifacts/lgbm_best_params.json")

# Fixed across every trial — not part of the search space. objective/
# random_state/verbose aren't things to tune, they're things to hold
# constant so trials differ only by the hyperparameters actually
# under test.
FIXED_PARAMS = {
    "objective": "binary",
    "random_state": 42,
    "verbose": -1,
}


def objective(trial: optuna.Trial, train: pd.DataFrame) -> float:
    """
    Scores ONE hyperparameter combination: fits a model on every
    expanding-window fold, returns the average log loss across folds.

    Simple version: Optuna hands over one set of settings to try. This
    function takes those settings, runs the full "train on everything
    through year N, test on year N+1" walk-forward exercise (12 folds
    per models/cv.py), and hands back a single number — the average
    log loss. Lower is better. Optuna calls this over and over with
    different settings and keeps track of what worked.

    Averaging across folds rather than taking the best or last fold is
    deliberate: a combination that happens to nail 2017 but falls apart
    in 2020 isn't a good combination, it's a lucky one. The average is
    what makes this a test of "does this setting generalize across
    eras," which is the actual question.

    NOTE ON PREPROCESSING: to_differential() is called fresh inside
    each fold, but it's a pure row-by-row transformation (self_X minus
    opp_X on the same row) — nothing is FIT on the fold's data the way
    a SimpleImputer or StandardScaler would be. That's why it's safe
    to call here without worrying about cross-fold contamination, and
    why LightGBM needing neither of those preprocessing steps
    (see models/lightgbm_model.py's docstring) removes a leakage
    surface rather than just saving a few lines.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna's handle for this attempt — trial.suggest_* is how a
        search space gets declared. Optuna reads these calls to learn
        what it's allowed to vary.
    train : pd.DataFrame
        The symmetrized train split WITH Elo attached. Never val,
        never test.

    Returns
    -------
    float — mean log loss across all folds. Optuna minimizes this.
    """
    params = {
        **FIXED_PARAMS,
        # How big a step each tree takes. Lower = slower but steadier
        # learning, usually paired with more trees.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # How many trees. More isn't automatically better — past a
        # point they just memorize the training data.
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        # How complex ONE tree is allowed to get. The main overfitting
        # dial for LightGBM's leaf-wise growth, and the one most worth
        # keeping modest on a ~13K-row dataset.
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        # Minimum fights that must land in a leaf before it's allowed
        # to exist — stops the model from building a rule off 2 bouts.
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        # Randomly use only a fraction of features / rows per tree.
        # Both are anti-overfitting: they stop every tree from keying
        # off the same one or two dominant features (diff_age,
        # diff_reach_to_height_ratio in Step 4's importance table).
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        # Regularization — penalizes overly confident/complex fits.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    fold_losses = []
    for train_fold, val_fold in expanding_year_folds(train):
        X_tr, y_tr = to_differential(train_fold, verbose=False)
        X_va, y_va = to_differential(val_fold, verbose=False)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_va)[:, 1]
        fold_losses.append(log_loss(y_va, y_prob))

    return float(np.mean(fold_losses))


def run_study(
    train: pd.DataFrame, n_trials: int = 60, seed: int = 42
) -> optuna.Study:
    """
    Optuna scaffolding — creates the study, runs the search, returns
    it. Boilerplate around objective(), which is where the real logic
    lives.

    Simple version of what Optuna's doing: instead of trying every
    combination (grid search — far too many here) or guessing blindly
    (random search), it looks at which settings have scored well so
    far and concentrates its next guesses in those neighborhoods.
    Like adjusting your aim after each shot rather than firing
    randomly at the target.

    n_trials=60 is a deliberate stopping point, not a maximum. Each
    trial fits 12 models (one per fold), so 60 trials is ~720 fits.
    Past a certain point, the differences between top candidates
    shrink to noise and further searching risks tuning to this
    specific fold structure rather than finding real signal — the same
    reasoning that stopped ADR-014's Elo tuning after two grid rounds.

    Parameters
    ----------
    train : pd.DataFrame
        Symmetrized train split with Elo attached.
    n_trials : int
        How many hyperparameter combinations to try.
    seed : int
        Fixed so a rerun reproduces the same search path — the same
        reason the shuffle-label test used seed=42.

    Returns
    -------
    optuna.Study — .best_params and .best_value hold the winner.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",  # log loss: lower is better
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name="lgbm_expanding_window_cv",
    )
    study.optimize(lambda trial: objective(trial, train), n_trials=n_trials)
    return study


def save_best_params(study: optuna.Study, path: Path = BEST_PARAMS_PATH) -> None:
    """
    Writes the winning hyperparameters to JSON so Step 8 (and Week 3
    Saturday's model freeze) can load them without rerunning the
    search — a ~20 minute job that shouldn't have to happen twice.

    Saves the CV score alongside them deliberately: the number Optuna
    achieved in-fold is context for reading Step 8's val score. If
    val comes in dramatically better than the CV average, that's worth
    a second look, not a celebration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": study.best_params,
        "best_cv_log_loss": study.best_value,
        "n_trials": len(study.trials),
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    train, _ = build_train_val_with_elo()  # val deliberately discarded — not read here
    print(f"tuning on train only: {len(train)} rows, {train['bout_id'].nunique()} bouts")

    study = run_study(train, n_trials=60)

    print(f"\nbest CV log loss: {study.best_value:.6f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    save_best_params(study)