"""
Week 2 Saturday — Leakage audit script (docs/PLAN.md Section 3, Saturday).

Built incrementally over the audit session, one check per step. Today
holds Step 2: the shuffle-label test.

WHY THIS CHECK EXISTS
----------------------
Every hobby UFC predictor that reports suspiciously high accuracy has a
leak (docs/PLAN.md Section 0.2). The shuffle-label test is the most
direct way to catch one: randomly scramble which fighter actually won
each TRAINING fight, train the exact same pipeline on that scrambled
data, and score it against the real, untouched val labels. With no real
signal left to learn from, a clean pipeline should land at ~coin-flip:
accuracy ~0.50, log loss ~0.6931 (ln 2).

If it does NOT collapse to those numbers, the model found some pattern
besides "who actually won" to key on — the definition of a leak.

IMPORTANT: only y_train gets shuffled. y_val stays completely untouched
— we're testing whether a model trained on garbage labels can still
fool a REAL, honest scorecard.
"""

import numpy as np
import pandas as pd
from scipy import stats

from features.differential import to_differential
from models.baselines import build_logreg_pipeline
from models.metrics import evaluate

FEATURE_GROUPS = {
    "physical": [
        "diff_age", "diff_height_cm", "diff_reach_cm",
        "diff_reach_to_height_ratio",
    ],
    "striking": [
        "diff_slpm", "diff_sapm", "diff_significant_strike_rate",
        "diff_striking_accuracy", "diff_striking_defense",
        "diff_striking_output_decay",
    ],
    "grappling": [
        "diff_td_avg_per_15", "diff_takedown_accuracy", "diff_takedown_defense",
        "diff_sub_attempts_per_15", "diff_control_time_pct",
        "diff_time_controlled_pct", "diff_takedown_output_decay",
    ],
    "outcome_tendencies": [
        "diff_career_win_pct", "diff_decision_win_pct", "diff_decision_loss_pct",
        "diff_submission_win_count", "diff_submission_success_rate",
        "diff_finish_rate", "diff_ko_loss_rate", "diff_sub_loss_rate",
    ],
    "experience_recency": [
        "diff_total_ufc_fights", "diff_days_since_last_fight",
        "diff_avg_fight_time_seconds", "diff_title_fight_experience",
    ],
    "knockdowns": [
        "diff_times_knocked_down", "diff_knockdown_rate",
    ],
}

def shuffle_label_test(
    train: pd.DataFrame, val: pd.DataFrame, seed: int = 42
) -> dict:
    """
    Trains the LR baseline pipeline on RANDOMLY SHUFFLED training
    labels, then scores it against the real, unshuffled val labels.

    Parameters
    ----------
    train : pd.DataFrame
        The symmetrized train split (data/processed/train.parquet).
    val : pd.DataFrame
        The symmetrized val split (data/processed/val.parquet) — used
        UNCHANGED, as the real scorecard.
    seed : int
        Fixes the shuffle so this run is reproducible. Worth rerunning
        with 2-3 different seeds afterward — a single seed landing near
        50% is good, but if you want extra confidence that this wasn't
        a lucky shuffle, a couple more seeds all landing in the same
        neighborhood makes the result more trustworthy.

    Returns
    -------
    dict — same shape as models.metrics.evaluate()'s return, so this
    slots into docs/RESULTS.md the same way as every other baseline,
    name="shuffle_label_test".
    """
    X_train, y_train = to_differential(train, verbose=False)
    X_val, y_val = to_differential(val, verbose=False)

    rng = np.random.default_rng(seed)
    y_train_shuffled = pd.Series(
        rng.permutation(y_train.to_numpy()),
        index=y_train.index,
        name=y_train.name,
    )

    pipeline = build_logreg_pipeline()
    pipeline.fit(X_train, y_train_shuffled)

    y_prob = pipeline.predict_proba(X_val)[:, 1]
    return evaluate(y_val.to_numpy(), y_prob, name="shuffle_label_test")

def naive_corner_only_baseline(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """
    Predicts self_won using ONLY source_corner as information — no
    fighter stats at all. Trained by taking the group-average win rate
    per corner in train, applied to val.

    THIS IS EXPECTED TO SCORE ~0.63 ACCURACY, NOT ~0.50. That's not a
    leak — source_corner directly encodes the original red/blue label,
    and red wins ~63% of the time in the raw data (LEAKAGE_LOG.md).
    This function exists purely as a REFERENCE POINT for the real check
    below: "how well could you do by cheating with corner alone?" —
    so we have a number to compare the real model against.
    """
    train_rates = train.groupby("source_corner")["self_won"].mean()

    y_true = val["self_won"].astype(int).to_numpy()
    y_prob = val["source_corner"].map(train_rates).to_numpy()

    return evaluate(y_true, y_prob, name="naive_corner_only_baseline")


def corner_symmetry_check(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """
    THE REAL CHECK. Trains the actual LR pipeline (diff_ features only
    — source_corner is never an input), then scores it SEPARATELY on
    the red-sourced half of val and the blue-sourced half of val.

    Simple version: the model never gets told which corner a fighter
    came from. If it's genuinely learning from skill differences, it
    should be equally good (or equally bad) at predicting red-sourced
    rows and blue-sourced rows. If it's secretly riding along on corner
    position anyway (e.g. via some feature that correlates with it),
    one half will score noticeably better than the other — and if
    either half's accuracy drifts toward 0.63/0.37, that's the same
    signature naive_corner_only_baseline produces, which means the
    model rediscovered the corner bias through the back door.

    Also verifies source_corner is ~50/50 in both train and val —
    confirms symmetrization actually produced a balanced dataset,
    which is the mechanism this whole defense depends on.

    Returns
    -------
    pd.DataFrame: one row per (split, corner) combo — name, n,
    accuracy, log_loss, brier, ece — plus the balance check printed
    separately.
    """
    for split_name, split_df in [("train", train), ("val", val)]:
        balance = split_df["source_corner"].value_counts(normalize=True)
        print(f"{split_name} source_corner balance:\n{balance}\n")

    X_train, y_train = to_differential(train, verbose=False)
    X_val, y_val = to_differential(val, verbose=False)

    pipeline = build_logreg_pipeline()
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    scored = val[["source_corner"]].copy()
    scored["y_true"] = y_val.to_numpy()
    scored["y_prob"] = y_prob

    rows = []
    for corner in ["red", "blue"]:
        subset = scored[scored["source_corner"] == corner]
        rows.append(
            evaluate(
                subset["y_true"].to_numpy(),
                subset["y_prob"].to_numpy(),
                name=f"lr_corner_{corner}",
            )
        )
    return pd.DataFrame(rows)

def print_corner_win_rates_by_era(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """
    Directly computes the red-corner win rate within train and val
    SEPARATELY — no classifier, no threshold, just the raw rate — to
    confirm (or rule out) whether naive_corner_only_baseline's lower-
    than-0.6319 accuracy is genuine train/val era drift in the corner
    bias itself, rather than a bug somewhere upstream.
    """
    for name, df in [("train", train), ("val", val)]:
        red_rate = df.loc[df["source_corner"] == "red", "self_won"].mean()
        n_bouts = (df["source_corner"] == "red").sum()
        print(f"{name} red-corner win rate: {red_rate:.4f}  (n={n_bouts} bouts)")

def list_differential_features(train: pd.DataFrame) -> list[str]:
    """
    Prints the real diff_ column names from to_differential's output,
    so the FEATURE_GROUPS dict below can be built against what
    actually exists rather than guessed from docs/PLAN.md Section 2's
    feature list (which describes intent, not final column names).
    """
    X_train, _ = to_differential(train, verbose=False)
    cols = X_train.columns.tolist()
    print(f"{len(cols)} differential features:")
    for c in cols:
        print(f"  {c}")
    return cols

def feature_ablation_test(
    train: pd.DataFrame, val: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    """
    For each feature group, drops just that group's columns, refits
    the LR pipeline on what's left, and scores on val — repeated once
    per group, plus one "full model" row with nothing dropped as the
    reference point.

    Simple version: same model, same everything else, minus one
    category of stats at a time. The gap between the full-model row
    and a given group's row is roughly "how much that group was
    contributing." No single group should be propping up the whole
    model, and no group should be causing a wild, unexplainable swing.

    Parameters
    ----------
    train, val : pd.DataFrame
        The symmetrized splits.
    groups : dict[str, list[str]]
        FEATURE_GROUPS above — group name -> list of diff_ column
        names belonging to it. Every name must exist in
        to_differential's output or this will KeyError loudly (which
        is the point — a silent typo here would quietly ablate nothing).

    Returns
    -------
    pd.DataFrame, one row per group (plus "full_model"): name, n,
    accuracy, log_loss, brier, ece, delta_log_loss (vs. full model —
    positive means removing this group made log loss WORSE, i.e. this
    group was helping).
    """
    X_train, y_train = to_differential(train, verbose=False)
    X_val, y_val = to_differential(val, verbose=False)

    all_grouped = {c for cols in groups.values() for c in cols}
    missing = all_grouped - set(X_train.columns)
    if missing:
        raise KeyError(
            f"These group columns don't exist in to_differential's output: "
            f"{missing} — check list_differential_features's printout above."
        )

    def _fit_score(X_tr, X_v, name):
        pipeline = build_logreg_pipeline()
        pipeline.fit(X_tr, y_train)
        y_prob = pipeline.predict_proba(X_v)[:, 1]
        return evaluate(y_val.to_numpy(), y_prob, name=name)

    results = [_fit_score(X_train, X_val, "full_model")]
    full_log_loss = results[0]["log_loss"]

    for group_name, cols in groups.items():
        X_tr_dropped = X_train.drop(columns=cols)
        X_v_dropped = X_val.drop(columns=cols)
        row = _fit_score(X_tr_dropped, X_v_dropped, f"minus_{group_name}")
        row["delta_log_loss"] = row["log_loss"] - full_log_loss
        results.append(row)

    results[0]["delta_log_loss"] = 0.0
    return pd.DataFrame(results)

def train_val_distribution_check(
    train: pd.DataFrame, val: pd.DataFrame, alpha: float = 0.01
) -> pd.DataFrame:
    """
    Runs a two-sample Kolmogorov-Smirnov test on every diff_ feature,
    comparing its train-era distribution against its val-era
    distribution.

    Simple version: for each stat, are train and val drawn from
    roughly the same "shape" of distribution, or does one look
    shifted/stretched relative to the other in a way that's suspicious?
    A real difference isn't automatically a bug — the sport genuinely
    changes over a decade (e.g. this project's own PLAN_ADDENDUM.md
    notes a ~2.4% pre-Unified-Rules-era exclusion, and rule/format
    changes are real). The point of this check is to surface
    CANDIDATES for a manual look, not to auto-flag anything as broken.

    alpha=0.01 (not the usual 0.05) is deliberate: running 31
    independent tests at 0.05 would produce ~1-2 false alarms on
    pure chance alone (multiple-comparisons problem) even with zero
    real drift anywhere. A stricter threshold cuts down on chasing
    noise, at the cost of maybe missing a borderline-real shift — an
    acceptable tradeoff for an exploratory audit check, not a
    formal statistical claim.

    Parameters
    ----------
    train, val : pd.DataFrame
        The symmetrized splits.
    alpha : float
        p-value threshold below which a feature gets flagged.

    Returns
    -------
    pd.DataFrame, one row per diff_ feature, sorted by p-value
    ascending (most suspicious first): feature, train_mean, val_mean,
    train_std, val_std, ks_statistic, p_value, flagged.
    """
    X_train, _ = to_differential(train, verbose=False)
    X_val, _ = to_differential(val, verbose=False)

    rows = []
    for col in X_train.columns:
        train_vals = X_train[col].dropna().to_numpy()
        val_vals = X_val[col].dropna().to_numpy()

        ks_stat, p_value = stats.ks_2samp(train_vals, val_vals)

        rows.append(
            {
                "feature": col,
                "train_mean": train_vals.mean(),
                "val_mean": val_vals.mean(),
                "train_std": train_vals.std(),
                "val_std": val_vals.std(),
                "ks_statistic": ks_stat,
                "p_value": p_value,
                "flagged": p_value < alpha,
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    n_flagged = result["flagged"].sum()
    print(f"{n_flagged} / {len(result)} features flagged at alpha={alpha}")
    return result

def split_integrity_check(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame | None = None
) -> None:
    """
    Verifies the physical train/val/(test) split has no contamination:

      1. No bout_id appears in more than one split.
      2. Every bout_id's two symmetrized rows (source_corner='red' and
         source_corner='blue') land in the SAME split — never split
         across two different files.
      3. Each split's own row count is exactly 2x its distinct bout_id
         count (i.e. every bout really does have both symmetrized rows
         present, not a stray unpaired row from some upstream bug).

    Simple version: this is checking that the wall between train/val/
    test doesn't have any cracks in it. Everything else audited today
    assumed the split itself was solid ground — this is the check that
    actually verifies the ground, rather than assuming it.

    test is optional and left out by convention (docs/PLAN.md's "don't
    even look at the locked drawer" rule) — pass it explicitly only if
    you've deliberately decided to unlock it for this check. Checks 1
    and 2 still run correctly with just train/val if test is None;
    you're simply not verifying test's own boundary against the other
    two yet.

    Raises
    ------
    Nothing — prints findings and returns None. This is a diagnostic,
    not an assertion; you decide what counts as a failure once you see
    the output, same as every other check today.
    """
    splits = {"train": train, "val": val}
    if test is not None:
        splits["test"] = test

    # --- Check 1: no bout_id appears in more than one split ---
    bout_id_sets = {name: set(df["bout_id"]) for name, df in splits.items()}
    names = list(bout_id_sets.keys())
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = bout_id_sets[names[i]] & bout_id_sets[names[j]]
            if overlap:
                any_overlap = True
                print(
                    f"CONTAMINATION: {len(overlap)} bout_id(s) appear in "
                    f"BOTH {names[i]} and {names[j]}: {sorted(overlap)[:10]}"
                    f"{' ...' if len(overlap) > 10 else ''}"
                )
    if not any_overlap:
        print("Check 1 PASS: no bout_id appears in more than one split.")

    # --- Check 2: both symmetrized rows of a bout stay together ---
    any_split_pair = False
    for name, df in splits.items():
        counts = df.groupby("bout_id")["source_corner"].nunique()
        singleton_bouts = counts[counts != 2]
        if len(singleton_bouts) > 0:
            any_split_pair = True
            print(
                f"WITHIN-{name.upper()} ISSUE: {len(singleton_bouts)} bout_id(s) "
                f"don't have exactly 2 distinct source_corner values in {name} "
                f"alone (could mean a duplicate or a missing pair): "
                f"{singleton_bouts.index.tolist()[:10]}"
            )
    if not any_split_pair:
        print("Check 2 PASS: every bout_id has exactly 2 rows (both corners) within its own split.")

    # --- Check 3: row count == 2x distinct bout_id count, per split ---
    any_count_mismatch = False
    for name, df in splits.items():
        n_rows = len(df)
        n_bouts = df["bout_id"].nunique()
        if n_rows != 2 * n_bouts:
            any_count_mismatch = True
            print(
                f"COUNT MISMATCH in {name}: {n_rows} rows but {n_bouts} distinct "
                f"bout_ids (expected {2 * n_bouts} rows for full symmetrization)."
            )
        else:
            print(f"{name}: {n_bouts} bouts x 2 = {n_rows} rows. Matches.")
    if not any_count_mismatch:
        print("Check 3 PASS: every split's row count is exactly 2x its bout count.")

if __name__ == "__main__":
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")

    
    result = shuffle_label_test(train, val)
    print(result)

    # Quick pass/fail framing, not a hard assertion — eyeball it, same
    # judgment call as the rest of today's checks.
    acc = result["accuracy"]
    log_loss = result["log_loss"]
    if abs(acc - 0.5) < 0.03 and abs(log_loss - 0.6931) < 0.03:
        print("PASS: collapses to ~coin-flip, as expected.")
    else:
        print(
            "INVESTIGATE: shuffled-label model did better than chance — "
            "possible leak. Check corner-ordering leakage first "
            "(docs/PLAN.md Section 0.2, point 2)."
        )
    
    print("\n--- Naive corner-only baseline (reference point, ~0.63 EXPECTED) ---")
    print(naive_corner_only_baseline(train, val))

    print("\n--- Corner symmetry check on the REAL LR model ---")
    print(corner_symmetry_check(train, val))
    
    print("\n--- Red-corner win rate by era ---")
    print_corner_win_rates_by_era(train, val)
    

    print(list_differential_features(train))
    print("\n--- Feature group ablation ---")
    print(feature_ablation_test(train, val, FEATURE_GROUPS))
    
    print("\n--- Train/val distribution drift check ---")
    drift_result = train_val_distribution_check(train, val)
    print(drift_result.to_string())
    
    print("\n--- Split integrity check ---")
    split_integrity_check(train, val)  # test intentionally excluded — stays locked