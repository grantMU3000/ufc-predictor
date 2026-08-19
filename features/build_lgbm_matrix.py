"""
Assembles the LightGBM training matrix — Week 3 Monday (docs/PLAN.md
Section 3). Pure glue: it doesn't compute any feature values itself.
It merges Friday's Elo ratings (features/elo.py) onto the already-
symmetrized train/val splits, then leans on features/differential.py's
existing self_/opp_ pattern-matching to turn the result into a diff_
matrix — no changes needed to that file.

Why a new file instead of extending models/baselines.py: elo_baseline()
there only ever attaches Elo to val, because that's all a standalone
baseline needs to be graded. LightGBM TRAINS on Elo, so it needs it on
train too — different enough of a job to earn its own file rather than
overloading elo_baseline's merge logic to do double duty.

Nothing here decides HOW Elo is computed (features/elo.py owns that)
or how symmetrization works (features/symmetrize.py owns that) — this
file only bridges the two.
"""

import duckdb
import numpy as np
import pandas as pd

from features.elo import compute_elo_ratings, k_factor_by_experience
from features.labels import get_completed_decided_bouts
from features.split import TEST_START


def _load_elo_ratings(
    k_new: float = 80.0,
    k_veteran: float = 24.0,
    decay_scale: float = 3.0,
) -> pd.DataFrame:
    """
    Computes pre-fight Elo for every train+val-era bout, using the
    tuned experience-based K from ADR-014 (defaults match
    models/baselines.py's elo_baseline — kept in sync deliberately,
    not re-tuned here).

    Filtered to event_date < TEST_START before compute_elo_ratings
    ever sees it — same "caller decides what it's allowed to see"
    rule as features/elo.py's own module docstring. Friday is the
    test-set unlock, not today.

    Returns
    -------
    pd.DataFrame — bout_id, red_elo_pre, blue_elo_pre, covering every
    decided train+val bout in ONE call. compute_elo_ratings walks
    train+val history as one continuous timeline (a fighter's rating
    doesn't reset at the train/val boundary — only the model's
    TRAINING does), so this single pass already covers both splits.
    """
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    labels = get_completed_decided_bouts(con)
    labels = labels[
        pd.to_datetime(labels["event_date"]) < TEST_START
    ].reset_index(drop=True)
    con.close()

    def k_fn(fight_count: int) -> float:
        return k_factor_by_experience(
            fight_count, k_new=k_new, k_veteran=k_veteran, decay_scale=decay_scale
        )

    return compute_elo_ratings(labels, k_factor=k_fn)


def attach_elo(df: pd.DataFrame, elo_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Merges pre-fight Elo onto an already-symmetrized split (train or
    val), producing self_elo_pre / opp_elo_pre — same self_/opp_
    naming convention every other feature already uses. That's what
    lets to_differential() pick this up automatically as diff_elo_pre
    with zero changes to that file.

    Simple version: elo_ratings still speaks "red"/"blue," same as
    every raw feature before symmetrize.py ever touches it. This
    function does the same ME/THEM relabeling _symmetrize_row does
    elsewhere — just for one extra column, bolted on after the fact
    instead of inside store.py's original per-bout build.

    Parameters
    ----------
    df : pd.DataFrame
        A symmetrized split (train or val). Needs bout_id and
        source_corner (both already present per symmetrize.py).
    elo_ratings : pd.DataFrame
        Output of compute_elo_ratings — bout_id, red_elo_pre,
        blue_elo_pre.

    Returns
    -------
    pd.DataFrame — df plus self_elo_pre and opp_elo_pre. Row count
    and row order match df's original (left merge) — every row in df
    is expected to find a match, checked below rather than assumed.

    Raises
    ------
    ValueError if any row fails to match, or if the merge changes row
    count — either would mean train/val and the Elo history disagree
    about which bouts are in scope, which should fail loud, not
    silently hand LightGBM a NaN Elo for a real fight.
    """
    merged = df.merge(elo_ratings, on="bout_id", how="left")

    if len(merged) != len(df):
        raise ValueError(
            f"row count changed after merge ({len(df)} -> {len(merged)}) "
            f"— check for duplicate bout_id in elo_ratings."
        )

    missing = merged["red_elo_pre"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} row(s) had no matching bout_id in elo_ratings "
            f"— check both were built from the same decided-bout "
            f"population before TEST_START."
        )

    is_red = merged["source_corner"] == "red"
    merged["self_elo_pre"] = np.where(
        is_red, merged["red_elo_pre"], merged["blue_elo_pre"]
    )
    merged["opp_elo_pre"] = np.where(
        is_red, merged["blue_elo_pre"], merged["red_elo_pre"]
    )

    return merged.drop(columns=["red_elo_pre", "blue_elo_pre"])


def build_train_val_with_elo() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience wrapper: loads train.parquet/val.parquet,
    attaches Elo to both, hands back symmetrized dataframes ready for
    features.differential.to_differential (or straight to LightGBM,
    if a future step wants raw self_/opp_ columns too — not needed
    today, differential features stay the input, consistent with LR).

    Returns
    -------
    (train, val) — same shape as the raw parquet reads, plus
    self_elo_pre / opp_elo_pre on both.
    """
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")

    elo_ratings = _load_elo_ratings()

    train = attach_elo(train, elo_ratings)
    val = attach_elo(val, elo_ratings)

    return train, val


if __name__ == "__main__":
    from features.differential import to_differential

    train, val = build_train_val_with_elo()
    print(f"train: {len(train)} rows, val: {len(val)} rows")

    X_train, y_train = to_differential(train, verbose=True)
    assert "diff_elo_pre" in X_train.columns, (
        "diff_elo_pre didn't show up — check self_elo_pre/opp_elo_pre "
        "landed with those exact names."
    )
    print(f"diff_elo_pre present. X_train shape: {X_train.shape}")