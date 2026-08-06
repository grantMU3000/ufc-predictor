from pathlib import Path

import pandas as pd

from data.ingestion.parsers import (
    read_events,
    read_fight_results,
    read_fight_stats,
    read_fighters,
)

LOG_DIR = Path("data/ingestion/logs")


def _build_fighter_lookup(fighters_df: pd.DataFrame) -> dict[str, int | None]:
    """
    Map real_name -> fighter_id for names that appear exactly once. The 8
    known name collisions map to None — deliberately ambiguous, not
    guessed. Anything referencing one of these names gets logged and
    excluded downstream rather than resolved silently.
    """
    counts = fighters_df["real_name"].value_counts()
    unique_names = counts[counts == 1].index

    lookup = dict(zip(
        fighters_df.loc[fighters_df["real_name"].isin(unique_names), "real_name"],
        fighters_df.loc[fighters_df["real_name"].isin(unique_names), "fighter_id"],
    ))
    for name in counts[counts > 1].index:
        lookup[name] = None  # ambiguous, on purpose

    return lookup


def build_fighters_table() -> pd.DataFrame:
    """Explicit, deterministic fighter_id"""
    fighters = read_fighters().sort_values("source_url").reset_index(drop=True)
    fighters["fighter_id"] = fighters.index + 1
    return fighters


def build_events_table() -> pd.DataFrame:
    events = read_events().sort_values("source_url").reset_index(drop=True)
    events["event_id"] = events.index + 1
    return events


def build_bouts_table(events: pd.DataFrame, fighter_lookup) -> pd.DataFrame:
    """
    Resolve fight_results.csv's free-text names into real FKs. Rows with
    an ambiguous or unmatched fighter/event name are logged to
    data/ingestion/logs/unresolved_bout_fighters.csv and excluded.
    """
    bouts = read_fight_results()

    event_lookup = dict(zip(events["name"], events["event_id"]))

    bouts["fighter_red_id"] = bouts["fighter_red_name"].map(fighter_lookup)
    bouts["fighter_blue_id"] = bouts["fighter_blue_name"].map(fighter_lookup)
    bouts["event_id"] = bouts["event_name"].map(event_lookup)

    unresolved_mask = (
        bouts["fighter_red_id"].isna()
        | bouts["fighter_blue_id"].isna()
        | bouts["event_id"].isna()
    )

    if unresolved_mask.any():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        bouts[unresolved_mask].to_csv(
            LOG_DIR / "unresolved_bout_fighters.csv", index=False
        )
        print(f"build_bouts_table: {unresolved_mask.sum()} rows unresolved "
              f"(ambiguous/unmatched name) — logged, excluded from load.")

    bouts = bouts[~unresolved_mask].reset_index(drop=True)

    bouts["winner_id"] = bouts.apply(
        lambda row: row["fighter_red_id"] if row["winner_side"] == "red"
        else row["fighter_blue_id"] if row["winner_side"] == "blue"
        else None,
        axis=1,
    )

    bouts["bout_id"] = bouts.index + 1

    # event_name/bout_matchup kept for the bout_stats join below — not
    # real schema columns, loaders.py should drop them before inserting.
    keep_cols = [
        "bout_id", "event_id", "fighter_red_id", "fighter_blue_id",
        "weight_class", "is_title_fight", "scheduled_rounds", "status",
        "winner_id", "method", "method_detail", "ending_round",
        "ending_time_seconds", "source_url",
        "event_name", "bout_matchup",
    ]
    return bouts[keep_cols]


def build_bout_stats_table(bouts: pd.DataFrame, fighter_lookup) -> pd.DataFrame:
    """
    Resolve fight_stats.csv rows to real bout_id/fighter_id FKs. INNER
    join to bouts on (event_name, bout_matchup) — required, not a left
    join: bouts already excludes 215 pre-Unified-Rules rows plus any
    unresolved-name rows, so orphaned stats rows must drop too.
    """
    stats = read_fight_stats()

    stats["fighter_id"] = stats["fighter_name"].map(fighter_lookup)

    merged = stats.merge(
        bouts[["bout_id", "event_name", "bout_matchup"]],
        on=["event_name", "bout_matchup"],
        how="inner",
    )

    unresolved_mask = merged["fighter_id"].isna()
    if unresolved_mask.any():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        merged[unresolved_mask].to_csv(
            LOG_DIR / "unresolved_bout_stats_fighters.csv", index=False
        )
        print(f"build_bout_stats_table: {unresolved_mask.sum()} rows "
              f"unresolved — logged, excluded.")

    merged = merged[~unresolved_mask].reset_index(drop=True)

    landed_attempted_cols = [
        col for pair in [
            ("sig_strikes_landed", "sig_strikes_attempted"),
            ("total_strikes_landed", "total_strikes_attempted"),
            ("takedowns_landed", "takedowns_attempted"),
            ("head_strikes_landed", "head_strikes_attempted"),
            ("body_strikes_landed", "body_strikes_attempted"),
            ("leg_strikes_landed", "leg_strikes_attempted"),
            ("distance_strikes_landed", "distance_strikes_attempted"),
            ("clinch_strikes_landed", "clinch_strikes_attempted"),
            ("ground_strikes_landed", "ground_strikes_attempted"),
        ] for col in pair
    ]
    keep_cols = [
        "bout_id", "fighter_id", "round_number", "knockdowns",
        "sub_attempts", "reversals", "control_time_seconds",
    ] + landed_attempted_cols

    return merged[keep_cols]