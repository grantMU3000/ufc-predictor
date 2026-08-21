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
from features.tier3 import build_recent_damage_by_bout, build_weight_class_change_by_bout


def _load_labels_and_elo(
    k_new: float = 80.0,
    k_veteran: float = 24.0,
    decay_scale: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    (labels, elo_ratings)
        labels: the decided-bout frame Elo was computed from -
            bout_id, event_date, fighter ids, winner_id. Returned
            rather than discarded because features/tier3.py's SoS
            needs the same population and the same date filtering,
            and reloading it would mean a second DuckDB round-trip
            for data already in hand.
        elo_ratings: bout_id, red_elo_pre, blue_elo_pre.
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

    return labels, compute_elo_ratings(labels, k_factor=k_fn)


def attach_by_corner(
    df: pd.DataFrame, bout_level: pd.DataFrame, stems: list[str]
) -> pd.DataFrame:
    """
    Merges bout-level red_X/blue_X columns onto a symmetrized split
    and relabels them to self_X/opp_X based on source corner.

    Generalized from the original attach_elo - SoS needs the exact
    same ME/THEM flip that Elo did, and a second near-identical
    function would be two places to fix the same bug. 'stems' is the
    list of names WITHOUT the corner prefix: pass["elo_pre"] and it
    looks for red_elo_pre/blue_elo_pre and produces self_elo_pre/
    opp_elo_pre.

    That self_opp_ naming is the whole point - it's what lets 
    features/differential.py's to_differential() auto-discover these
    as diff_ columns with zero edits to that file.

    Parameters
    ----------
    df : pd.DataFrame
        A symmetrized split. Needs bout_id and source_corner.
    bout_level : pd.DataFrame
        One row per bout, with red_{stem}/blue_{stem} for every stem.
    stems : list[str]
        Feature names without corner prefix, e.g. ["elo_pre"] or
        ["sos_last_3", "sos_last_5"].

    Returns
    -------
    pd.DataFrame — df plus self_{stem}/opp_{stem} for each steam

    Raises
    ------
    ValueError on row-count change or unmatched rows — either means
    the split and the bout-level frame disagree about scope, which
    should fail loud rather than hand LightGBM a NaN for a real
    fight.

    NOTE on NaN: an unmatched BOUT raises. A matched bout whose value
    is legitimately NaN (a debutant has no strength of schedule) is
    fine and passes through — LightGBM handles it natively. The check
    below is deliberately on bout_id matching, not on value
    nullity, so those two cases stay distinguishable.
    """
    before = len(df)
    merged = df.merge(bout_level, on="bout_id", how="left", indicator=True)

    if len(merged) != before:
        raise ValueError(
            f"row count changed after merge ({before} -> {len(merged)}) "
            f"— check for duplicate bout_id in bout_level."
        )

    unmatched = (merged["_merge"] != "both").sum()
    if unmatched:
        raise ValueError(
            f"{unmatched} row(s) had no matching bout_id — check both "
            f"were built from the same decided-bout population before "
            f"TEST_START."
        )
    merged = merged.drop(columns=["_merge"])

    is_red = merged["source_corner"] == "red"
    for stem in stems:
        merged[f"self_{stem}"] = np.where(
            is_red, merged[f"red_{stem}"], merged[f"blue_{stem}"]
        )
        merged[f"opp_{stem}"] = np.where(
            is_red, merged[f"blue_{stem}"], merged[f"red_{stem}"]
        )

    drop_cols = [f"{c}_{stem}" for stem in stems for c in ("red", "blue")]
    return merged.drop(columns=drop_cols)

def _load_recent_damage(labels: pd.DataFrame) -> pd.DataFrame:
    """
    Opens its own connection (separate from _load_labels_and_elo's)
    since it needs bout_stats, which the Elo loader has no reason to
    touch. Kept separate rather than widening that function's scope —
    its job is "give me Elo," not "give me every table any future
    Tier 3 feature might eventually want."
    """
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts", "bout_stats"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )
    damage = build_recent_damage_by_bout(con, labels)
    con.close()
    return damage

def build_train_val_with_elo(
    include_sos: bool = False,
    include_damage: bool = False,
    include_weight: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience wrapper: loads train.parquet/val.parquet,
    attaches Elo to both, hands back symmetrized dataframes ready for
    features.differential.to_differential (or straight to LightGBM,
    if a future step wants raw self_/opp_ columns too — not needed
    today, differential features stay the input, consistent with LR).

    Include_sos toggles Tier 3 strength-of-schedule columns. Defaults
    True, but exposed as a flag specifically so models/feature_deltas
    .py can build the exact pre-SoS baseline for a clean before/after
    — measuring a feature's delta requires being able to turn it off.

    Returns
    -------
    (train, val) — same shape as the raw parquet reads, plus
    self_elo_pre / opp_elo_pre and (if enabled) sos_last_3/5,
    recent_damage_24mo, layoff_x_age, age_x_experience.
    """
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")

    labels, elo_ratings = _load_labels_and_elo()

    train = attach_by_corner(train, elo_ratings, stems=["elo_pre"])
    val = attach_by_corner(val, elo_ratings, stems=["elo_pre"])

    if include_sos:
        from features.tier3 import SOS_WINDOWS, build_sos_by_bout

        sos = build_sos_by_bout(labels, elo_ratings)
        stems = [f"sos_last_{n}" for n in SOS_WINDOWS]
        train = attach_by_corner(train, sos, stems=stems)
        val = attach_by_corner(val, sos, stems=stems)

    if include_damage:
        from features.tier3 import add_interaction_features

        damage = _load_recent_damage(labels)
        train = attach_by_corner(train, damage, stems=["recent_damage_24mo"])
        val = attach_by_corner(val, damage, stems=["recent_damage_24mo"])

        # Interactions need self_/opp_ age, days_since_last_fight,
        # total_ufc_fights already in place — true regardless of the
        # flags above, since those come from the original symmetrized
        # parquet, not from anything attached in this function.
        train = add_interaction_features(train)
        val = add_interaction_features(val)

    if include_weight:
        wc = _load_weight_class_change(labels)
        train = attach_by_corner(train, wc, stems=["weight_class_change"])
        val = attach_by_corner(val, wc, stems=["weight_class_change"])

    return train, val

def _load_weight_class_change(labels: pd.DataFrame) -> pd.DataFrame:
    """Same connection pattern as _load_recent_damage — only needs
    bouts/events, but reuses the same view set for consistency."""
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )
    result = build_weight_class_change_by_bout(con, labels)
    con.close()
    return result

if __name__ == "__main__":
    from features.differential import to_differential

    train, val = build_train_val_with_elo()
    print(f"train: {len(train)} rows, val: {len(val)} rows")

    X_train, y_train = to_differential(train, verbose=True)

    expected_new = [
        "diff_elo_pre",
        "diff_sos_last_3",
        "diff_sos_last_5",
        "diff_recent_damage_24mo",
        "diff_layoff_x_age",
        "diff_age_x_experience",
        "diff_weight_class_change",
    ]
    missing = [c for c in expected_new if c not in X_train.columns]
    assert not missing, f"missing expected diff_ columns: {missing}"

    print(f"all Tier 3 diff_ columns present. X_train shape: {X_train.shape}")
    print(train["self_weight_class_change"].value_counts(dropna=False))