"""
Baseline LightGBM model — Week 3 Monday (docs/PLAN.md Section 3).
Default hyperparameters, no tuning yet — that's Step 7's job, later
today. This step exists purely to confirm the pipeline works
end-to-end and to get a first, honest "before tuning" number logged,
same discipline every other baseline in this project has followed
(market Thursday, LR/Elo Friday — always log the first honest attempt
before touching a knob).

Why no SimpleImputer here, unlike build_logreg_pipeline in
models/baselines.py: LightGBM natively handles missing values — it
learns, per feature, per split, which direction (left or right) a NaN
should default to. LogisticRegression has no such mechanism, which is
the entire reason that pipeline needed an imputer in the first place.
Passing real NaN straight through here isn't a shortcut — it's the
CORRECT way to hand LightGBM this data. Filling debutants' missing
career-rate stats with a median would be actively worse: it would
tell the model "this debutant has an average career," which is false
information, when "we don't know yet" (NaN) is the true fact.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential
from models.baselines import _load_val_and_odds, _odds_covered_mask

# Deliberately conservative defaults, not LightGBM's library defaults.
# num_leaves=31 / n_estimators=100 are lightgbm's own out-of-the-box
# values — kept as-is here on purpose. The point of THIS step is
# "what does an untuned model do," so tuning anything before Step 7
# would defeat the purpose of having a before/after comparison at all.
DEFAULT_PARAMS = {
    "objective": "binary",
    "random_state": 42,
    "verbose": -1,  # suppress LightGBM's own per-iteration logging
}


def train_lightgbm_baseline(
    train: pd.DataFrame, val: pd.DataFrame, params: dict | None = None
) -> tuple[np.ndarray, np.ndarray, lgb.LGBMClassifier, pd.DataFrame]:
    """
    Fits an LGBMClassifier on train's diff_ features with mostly-
    default hyperparameters, scores it on val.

    Simple version: same shape as logistic_regression_baseline in
    models/baselines.py — fit on train, predict on val, hand back
    (y_true, y_prob) so it plugs into models.metrics.evaluate exactly
    like every other baseline. The only real difference is what's
    INSIDE the box: no imputer/scaler pipeline, because LightGBM
    doesn't need either (see module docstring).

    Parameters
    ----------
    train, val : pd.DataFrame
        Symmetrized splits WITH Elo attached — i.e. the output of
        features.build_lgbm_matrix.build_train_val_with_elo, not the
        raw train.parquet/val.parquet (those are missing self_elo_pre/
        opp_elo_pre, and to_differential would build a 30-feature
        matrix instead of 31 without erroring — a silent, not loud,
        way to lose today's whole point).
    params : dict, optional
        Overrides DEFAULT_PARAMS. Not used today, but Step 7's tuning
        script will call this same function with Optuna-chosen params
        — kept as an argument now so that reuse doesn't require
        touching this function again later.

    Returns
    -------
    (y_true, y_prob, model, X_train)
        y_true, y_prob : same shape/meaning as every other baseline's
            return.
        model : the fitted LGBMClassifier — needed for Step 9's
            feature-importance check.
        X_train : kept around so Step 9 has the exact column names/
            order the model was actually fit on, same reason
            top_features() in baselines.py needs feature_names passed
            in explicitly rather than re-deriving them.
    """
    X_train, y_train = to_differential(train, verbose=False)
    X_val, y_val = to_differential(val, verbose=False)

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    return y_val.to_numpy(), y_prob, model, X_train


def top_lgbm_features(
    model: lgb.LGBMClassifier, feature_names: list, n: int = 10
) -> pd.DataFrame:
    """
    Ranks features by LightGBM's built-in gain importance — the
    tree-model equivalent of top_features() in models/baselines.py,
    same purpose: a quick interpretability check and a leakage
    smell-test in one.

    "Gain" here means: every time this feature was used to split a
    tree, how much did that split reduce prediction error, summed
    across every split in every tree. A feature with high gain is one
    the model leaned on heavily to separate winners from losers.

    Unlike LR's coefficients, there's no missingness-indicator
    columns to worry about here (LightGBM doesn't need
    SimpleImputer's add_indicator=True), so feature_names lines up
    with the model's importances directly — no expansion step needed,
    unlike top_features()'s get_feature_names_out call.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        A fitted model from train_lightgbm_baseline.
    feature_names : list
        Column names of X the model was fit on, in order — pass
        X_train.columns.tolist() (or the X_train returned by
        train_lightgbm_baseline directly).
    n : int
        How many top features to return.

    Returns
    -------
    pd.DataFrame: feature, gain — top n by gain, sorted descending.
    """
    result = pd.DataFrame(
        {"feature": feature_names, "gain": model.feature_importances_}
    )
    return result.sort_values("gain", ascending=False).head(n).reset_index(drop=True)


if __name__ == "__main__":
    from models.metrics import evaluate

    train, val = build_train_val_with_elo()
    print(f"train: {len(train)} rows, val: {len(val)} rows")

    y_val, y_prob, model, X_train = train_lightgbm_baseline(train, val)
    print(evaluate(y_val, y_prob, name="lightgbm_default_full_val"))

    # --- Odds-covered subset, apples-to-apples vs. every other baseline ---
    _, closing = _load_val_and_odds()
    covered_mask = _odds_covered_mask(val, closing).to_numpy()
    print(
        evaluate(
            y_val[covered_mask],
            y_prob[covered_mask],
            name="lightgbm_default_odds_covered_subset",
        )
    )

    # --- Feature importance, first look (Step 9 goes deeper) ---
    print(top_lgbm_features(model, X_train.columns.tolist()))