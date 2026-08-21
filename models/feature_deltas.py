"""
Measures what each Week 3 Tuesday feature is actually worth — the
"measure each addition's delta" half of docs/PLAN.md Section 3's
Tuesday entry.

WHY THIS MEASURES ON CV FOLDS AND NEVER TOUCHES VAL: the obvious way
to test a feature is to add it, score val, and keep it if val
improved. Do that six times and you've done feature SELECTION against
val — val stops being an honest estimate exactly as surely as if you'd
tuned hyperparameters on it. Same trap models/tune_lightgbm.py was
built to avoid, one level up: there it was hyperparameters, here it's
which columns exist. So every number in this file comes from
models/cv.py's expanding-window folds carved out of TRAIN ONLY. Val
is read once, in Step 7, after the feature set is already decided.

HYPERPARAMETERS ARE HELD FIXED at Monday's Optuna winner. Retuning
per configuration would confound two changes at once — you'd never
know whether a config won because the feature helped or because it
got a luckier hyperparameter draw. The tuned params were found on the
32-feature set, so they're mildly favorable to the baseline
configuration here; that's the conservative direction (it makes new
features work harder to prove themselves), which is the right way to
be biased.

INTERPRETING THE NUMBERS: Monday's tuned CV log loss was 0.660179 on
the 32-feature set. Per ADR-014's stopping rule and the same
reasoning in models/tune_lightgbm.py's run_study docstring,
differences of ~0.001-0.002 at this fold structure are noise, not
signal. A feature group that moves mean CV log loss by less than
~0.002 has not earned a place in the model.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score

from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential
from models.cv import expanding_year_folds
from models.lightgbm_model import load_tuned_params
from models.tune_lightgbm import FIXED_PARAMS

# Feature groups, named by what they ARE rather than which Step built
# them — a future reader shouldn't need the session history to know
# what "sos" means. Each maps to the diff_ columns that group
# contributes to the matrix.
FEATURE_GROUPS = {
    "sos": ["diff_sos_last_3", "diff_sos_last_5"],
    "damage": ["diff_recent_damage_24mo"],
    "interactions": ["diff_layoff_x_age", "diff_age_x_experience"],
    "weight_change": ["diff_weight_class_change"],
}

# Every Tier 3 column added today, as one list — the full set that
# gets stripped to produce the Monday-equivalent baseline.
ALL_TIER3_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]


def score_configuration(
    train: pd.DataFrame, drop_cols: list[str], params: dict
) -> tuple[float, list[float]]:
    """
    Runs the full expanding-window CV walk for ONE feature
    configuration, returning mean log loss across folds.

    Simple version: same walk-forward exercise models/tune_lightgbm
    .py's objective() does — train on everything through year N,
    score on year N+1, repeat for all 12 folds — except here the
    hyperparameters stay fixed and the FEATURE SET is what varies.

    drop_cols is applied AFTER to_differential inside each fold,
    deliberately: to_differential auto-discovers columns from the
    self_/opp_ pairs present, so dropping at the diff_ level is the
    only place a column set can be varied without rebuilding the
    whole matrix (which would mean re-running the per-bout damage
    query 5+ times for no benefit).

    Parameters
    ----------
    train : pd.DataFrame
        Symmetrized train split with ALL Tier 3 features attached.
        Never val.
    drop_cols : list[str]
        diff_ column names to exclude from this configuration.
        Missing names are ignored (errors="ignore") so a config can
        name a column that a prior drop already removed.
    params : dict
        Fixed hyperparameters, identical across every configuration.

    Returns
    -------
    (mean_log_loss, mean_accuracy, per_fold)
        per_fold : pd.DataFrame, one row per fold in chronological
        order — val_year, n_train_bouts, n_val_bouts, log_loss,
        accuracy. Kept separate from the mean so a caller can check
        WHETHER later folds (more training history) actually score
        better, rather than only ever seeing one averaged number that
        could hide a fold going badly wrong.
    """
    records = []

    for train_fold, val_fold in expanding_year_folds(train):
        X_tr, y_tr = to_differential(train_fold, verbose=False)
        X_va, y_va = to_differential(val_fold, verbose=False)

        X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
        X_va = X_va.drop(columns=drop_cols, errors="ignore")

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        y_prob = np.asarray(model.predict_proba(X_va))[:, 1]
        val_year = int(pd.to_datetime(val_fold["event_date"]).dt.year.iloc[0])

        records.append({
            "val_year": val_year,
            "n_train_bouts": train_fold["bout_id"].nunique(),
            "n_val_bouts": val_fold["bout_id"].nunique(),
            "log_loss": log_loss(y_va, y_prob),
            "accuracy": accuracy_score(y_va, y_prob > 0.5),
        })

    per_fold = pd.DataFrame(records)
    return float(per_fold["log_loss"].mean()), float(per_fold["accuracy"].mean()), per_fold


def run_cumulative_deltas(train: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds feature groups one at a time, in order, measuring the CV
    log loss after each addition.

    Simple version: start with Monday's 32-feature model, add SoS and
    remeasure, then add damage on top and remeasure, and so on. Each
    row's delta is "what did THIS group add, given everything before
    it."

    Order matters and is not neutral: a group added later has to
    prove itself against a stronger baseline. SoS goes first because
    it's the plan's headline Tier 3 item and the one with the
    clearest mechanism; weight_change goes last because its
    distribution (75% zeros, 7.4% non-zero signal) makes it the most
    likely to be marginal.

    Returns
    -------
    pd.DataFrame — config, n_features, mean_cv_log_loss,
    delta_vs_previous, delta_vs_baseline.
    """
    order = ["sos", "damage", "interactions", "weight_change"]

    rows = []
    active: list[str] = []

    baseline_loss, baseline_acc, baseline_fold = score_configuration(train, ALL_TIER3_COLS, params)
    rows.append({
        "config": "baseline (Monday, 32 feat)",
        "n_features": 32,
        "mean_cv_log_loss": baseline_loss,
        "mean_cv_accuracy": baseline_acc,
        "delta_vs_previous": np.nan,
        "delta_vs_baseline": 0.0,
        "delta_accuracy_vs_baseline": 0.0,
    })

    previous_loss = baseline_loss
    full_fold = baseline_fold  # overwritten below, kept for the last (full) config

    for group in order:
        active.extend(FEATURE_GROUPS[group])
        drop = [c for c in ALL_TIER3_COLS if c not in active]
        loss, acc, fold_detail = score_configuration(train, drop, params)

        rows.append({
            "config": f"+ {group}",
            "n_features": 32 + len(active),
            "mean_cv_log_loss": loss,
            "mean_cv_accuracy": acc,
            "delta_vs_previous": loss - previous_loss,
            "delta_vs_baseline": loss - baseline_loss,
            "delta_accuracy_vs_baseline": acc - baseline_acc,
        })
        previous_loss = loss
        full_fold = fold_detail  # last iteration = full 38-feature config

    return pd.DataFrame(rows), baseline_fold, full_fold


def run_leave_one_out(train: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Drops each group from the FULL 38-feature model, one at a time —
    the same shape as notebooks/leakage_audit.py's
    feature_ablation_test(), applied to today's additions.

    Simple version: cumulative deltas answer "was this worth adding
    when I added it." Leave-one-out answers "if I removed just this
    one now, would I miss it." Those can disagree — a group can look
    useless when added (because another group already covered the
    same ground) and still be the one carrying the signal when both
    are present. Running both is what stops a correlated pair from
    being wrongly cut.

    Relevant here specifically because diff_sos_last_3 and
    diff_sos_last_5 measure nearly the same thing, and diff_elo_pre
    correlates with both (beating strong opponents raises your own
    rating).

    Returns
    -------
    pd.DataFrame — dropped_group, mean_cv_log_loss, mean_cv_accuracy,
    delta_vs_full, delta_accuracy_vs_full. POSITIVE log-loss delta =
    removing it made things WORSE = the group was contributing.
    POSITIVE accuracy delta on removal = the group was actually
    HURTING accuracy, even if it helped log loss — the two can
    disagree, which is the whole reason accuracy got added today.
    """
    full_loss, full_acc, _ = score_configuration(train, [], params)

    rows = [{
        "dropped_group": "(none — full 38 feat)",
        "mean_cv_log_loss": full_loss,
        "mean_cv_accuracy": full_acc,
        "delta_vs_full": 0.0,
        "delta_accuracy_vs_full": 0.0,
    }]

    for group, cols in FEATURE_GROUPS.items():
        loss, acc, _ = score_configuration(train, cols, params)
        rows.append({
            "dropped_group": group,
            "mean_cv_log_loss": loss,
            "mean_cv_accuracy": acc,
            "delta_vs_full": loss - full_loss,
            "delta_accuracy_vs_full": acc - full_acc,
        })

    # Also test the two SoS windows individually — they're the one
    # group where an internal redundancy is likely, and cutting one
    # of two correlated columns is a real option the group-level
    # test can't see.
    for col in FEATURE_GROUPS["sos"]:
        loss, acc, _ = score_configuration(train, [col], params)
        rows.append({
            "dropped_group": f"  {col} only",
            "mean_cv_log_loss": loss,
            "mean_cv_accuracy": acc,
            "delta_vs_full": loss - full_loss,
            "delta_accuracy_vs_full": acc - full_acc,
        })

    return pd.DataFrame(rows)

def print_fold_trend(per_fold: pd.DataFrame, label: str) -> None:
    """
    Prints log loss and accuracy fold-by-fold, oldest to newest, so
    you can eyeball whether performance actually improves as training
    history grows — the intuitive expectation (2022's fold trains on
    23 years of history; 2011's trains on 12) but not a guarantee.

    CAVEAT WORTH READING BEFORE INTERPRETING THIS: a downward log-loss
    trend across years is consistent with "more history helps," but
    it's equally consistent with era drift — Saturday's KS-test
    findings (LEAKAGE_LOG.md) already showed several features have
    genuinely different distributions in different eras (the sport
    professionalizing, skill gaps narrowing). A clean trend here isn't
    proof of the "more data helps" story specifically; it's just
    correlated with it. Worth noting, not worth over-reading.

    Also prints the correlation between val_year and log_loss as a
    single number rather than making you eyeball 12 rows for a trend
    — negative correlation means log loss falls as years increase
    (improving), which is the expected direction.
    """
    print(f"\n--- fold-by-fold trend: {label} ---")
    print(per_fold.to_string(index=False))

    corr = per_fold["val_year"].corr(per_fold["log_loss"])
    print(f"corr(val_year, log_loss) = {corr:.3f}  "
          f"({'improving with more history' if corr < 0 else 'no clear improvement, or noisy'})")

if __name__ == "__main__":
    train, _ = build_train_val_with_elo()
    params = {**FIXED_PARAMS, **load_tuned_params()}

    print(f"train: {len(train)} rows, {train['bout_id'].nunique()} bouts")
    print(f"hyperparameters held fixed at Monday's Optuna winner\n")

    print("=== Cumulative additions ===")
    print("(negative delta = improvement; |delta| < 0.002 is noise)")
    cumulative, baseline_fold, full_fold = run_cumulative_deltas(train, params)
    print(cumulative.to_string(index=False))

    print_fold_trend(baseline_fold, "baseline (32 feat, Monday)")
    print_fold_trend(full_fold, "full (38 feat, today's additions)")

    print("\n=== Leave-one-out from full model ===")
    print("(positive delta = removing it HURT = group was contributing)")
    loo = run_leave_one_out(train, params)
    print(loo.to_string(index=False))