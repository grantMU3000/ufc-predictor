from pathlib import Path

import pandas as pd

from data.ingestion.parsers import (
    read_events,
    read_fight_results,
    read_fight_stats,
    read_fighters,
)

LOG_DIR = Path("data/ingestion/logs")

# transform.py — name corrections for known typos/formatting mismatches
# between fight_results.csv/fight_stats.csv and fighter_details.csv.
# Each of these is a confirmed 1:1 match, not an ambiguous collision.
NAME_OVERRIDES = {
    "Kai Kamaka": "Kai Kamaka III",
    "Bibulatov Magomed": "Magomed Bibulatov",
    "Rafael Cerquiera": "Rafael Cerqueira",
    "Patricio Freire": "Patricky Freire",
}

# Manually resolved fighter-name collisions. Each maps a colliding
# real_name to either a single default source_url (every appearance in
# this dataset belongs to one specific person), or a {weight_class:
# source_url} map when bout weight class disambiguates the two real
# people sharing the name. Determined 2026-08-06 by cross-referencing
# each candidate's listed weight (ufc_fighter_tott.csv) against every
# bout row's weight class. One manual judgment call: Bruno Silva's single
# Bantamweight row (UFC 243) is assigned to the Flyweight fighter as an
# early-career bout before he settled into his current division.
COLLISION_RESOLUTIONS: dict[str, dict[str, str]] = {
    "Bruno Silva": {
        "Flyweight Bout": "http://ufcstats.com/fighter-details/294aa73dbf37d281",      # Bulldog
        "Middleweight Bout": "http://ufcstats.com/fighter-details/12ebd7d157e91701",   # Blindado
        "Bantamweight Bout": "http://ufcstats.com/fighter-details/294aa73dbf37d281",   # Bulldog — early-career judgment call
    },
    "Jean Silva": {
        "__default__": "http://ufcstats.com/fighter-details/52ef95b5860fb28c",          # Lord
    },
    "Mike Davis": {
        "__default__": "http://ufcstats.com/fighter-details/fb3e61720be4690c",         # Beast Boy
    },
    "Michael McDonald": {
        "__default__": "http://ufcstats.com/fighter-details/d0314416a7f26527",         # Mayday
    },
    "Victor Valenzuela": {
        "__default__": "http://ufcstats.com/fighter-details/078695e385ec2f57",         # Psicosis
    },
    "Joey Gomez": {
        "__default__": "http://ufcstats.com/fighter-details/0778f94eb5d588a5",         # KO King
    },
}


def _build_source_url_lookup(fighters_df: pd.DataFrame) -> dict[str, int]:
    """Map source_url -> fighter_id, used to resolve COLLISION_RESOLUTIONS
    entries (kept as source_url since fighter_id only exists after
    build_fighters_table runs)."""
    return dict(zip(fighters_df["source_url"], fighters_df["fighter_id"]))


def _resolve_collision(
    name: str, weight_class: str, source_url_lookup: dict[str, int]
) -> int | None:
    """
    Check COLLISION_RESOLUTIONS for a manually-resolved name. Returns the
    correct fighter_id, or None if this isn't a known collision (caller
    should fall back to the plain fighter_lookup) or if it's a known
    collision with no matching weight class entry (don't guess).
    """
    resolution = COLLISION_RESOLUTIONS.get(name)
    if resolution is None:
        return None

    source_url = resolution.get(weight_class, resolution.get("__default__"))
    if source_url is None:
        return None

    return source_url_lookup.get(source_url)

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

    # Apply known typo/formatting corrections — these map to a name that's
    # already in `lookup` (assuming it's unambiguous itself).
    for wrong_name, correct_name in NAME_OVERRIDES.items():
        if correct_name in lookup and lookup[correct_name] is not None:
            lookup[wrong_name] = lookup[correct_name]

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


def build_bouts_table(events: pd.DataFrame, fighter_lookup, source_url_lookup: dict) -> pd.DataFrame:
    """
    Resolve fight_results.csv's free-text names into real FKs. Checks
    known collision resolutions (weight-class-based) before falling back
    to the plain name lookup. Rows still ambiguous or unmatched are
    logged to data/ingestion/logs/unresolved_bout_fighters.csv.
    """
    bouts = read_fight_results()

    event_lookup = dict(zip(events["name"], events["event_id"]))

    def _resolve_fighter(name: str, weight_class: str) -> int | None:
        resolved = _resolve_collision(name, weight_class, source_url_lookup)
        return resolved if resolved is not None else fighter_lookup.get(name)

    # pandas-stubs' apply() overloads don't have a variant for a row-function
    # returning `int | None` (they expect pandas' NAType for missing values,
    # not plain None) -- these are correct at runtime.
    bouts["fighter_red_id"] = bouts.apply(
        lambda row: _resolve_fighter(row["fighter_red_name"], row["weight_class"]),
        axis=1,
    )  # type: ignore[call-overload]
    bouts["fighter_blue_id"] = bouts.apply(
        lambda row: _resolve_fighter(row["fighter_blue_name"], row["weight_class"]),
        axis=1,
    )  # type: ignore[call-overload]
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


def build_bout_stats_table(bouts: pd.DataFrame, fighter_lookup, source_url_lookup: dict) -> pd.DataFrame:
    """
    Resolve fight_stats.csv rows to real bout_id/fighter_id FKs. INNER
    join to bouts happens FIRST now (not after fighter resolution, as
    before) — weight_class only exists on the bouts side, and known
    collisions need it to resolve correctly, same as build_bouts_table.
    """
    stats = read_fight_stats()

    stats["fighter_id"] = stats["fighter_name"].map(fighter_lookup)

    merged = stats.merge(
        bouts[["bout_id", "event_name", "bout_matchup", "weight_class"]],
        on=["event_name", "bout_matchup"],
        how="inner",
    )

    def _resolve_fighter(name: str, weight_class: str) -> int | None:
        resolved = _resolve_collision(name, weight_class, source_url_lookup)
        return resolved if resolved is not None else fighter_lookup.get(name)

    # Same pandas-stubs limitation as above (Optional-returning row apply).
    merged["fighter_id"] = merged.apply(
        lambda row: _resolve_fighter(row["fighter_name"], row["weight_class"]), axis=1
    )  # type: ignore[call-overload]

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