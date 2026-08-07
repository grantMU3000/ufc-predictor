"""
Transferring Historical Odds into my database
"""

from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
import pandas as pd

# Reused rather than duplicated -- same low-level helpers the Greco
# pipeline already uses (engine creation, NaN-safe record cleaning).
from data.ingestion.loaders import _get_engine, _clean_for_insert
from data.odds.matcher import resolve_fighter_name, match_to_bout, build_bout_lookup

def _moneyline_to_implied_prob(price: int) -> float:
    """American odds -> implied probability, e.g. -270 -> 0.7297, +220 -> 0.3125."""
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)

def build_real_name_lookup(engine) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, real_name FROM fighters")).fetchall()
    return {row.real_name: row.id for row in rows}

def build_alias_lookup(engine) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT fighter_id, alias_name FROM fighter_aliases")).fetchall()
    return {row.alias_name: row.fighter_id for row in rows}

def build_bout_lookup_from_db(engine) -> dict[frozenset, list[dict]]:
    with engine.begin() as conn:
        # Could this statement have a WHERE clause that only gets fights from 2020 & later?
        rows = conn.execute(text("""
            SELECT b.id AS bout_id, b.fighter_red_id, b.fighter_blue_id, e.event_date
            FROM bouts b JOIN events e ON b.event_id = e.id
             WHERE e.event_date >= '2020-06-06'
        """)).fetchall()
    return build_bout_lookup([dict(r._mapping) for r in rows])

def resolve_and_prepare_snapshot_rows(
    filtered_odds_data: list[dict],
    real_name_lookup: dict[str, int],
    alias_lookup: dict[str, int],
    bout_lookup: dict[frozenset, list[dict]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Resolves each entry's two fighters and matches to a real bout ONCE per
    entry (not per bookmaker row -- home_team/away_team are fixed for the
    whole entry, no reason to re-resolve them per sportsbook/outcome).

    Returns (ready_snapshot_rows, new_confirmed_aliases, unresolved_log).
    new_confirmed_aliases only includes 'fuzzy' matches -- 'exact' matches
    need no alias (the name already IS the canonical real_name), and
    'alias' matches are already in the table.
    """
    ready_rows, new_aliases, unresolved_log = [], [], []

    for entry in filtered_odds_data:
        home_id, home_method = resolve_fighter_name(entry["home_team"], real_name_lookup, alias_lookup)
        away_id, away_method = resolve_fighter_name(entry["away_team"], real_name_lookup, alias_lookup)

        bout_id = None
        if home_id is not None and away_id is not None:
            bout_id = match_to_bout(home_id, away_id, entry["commence_time"], bout_lookup)

        if bout_id is None:
            unresolved_log.append({
                "commence_time": entry["commence_time"],
                "home_team": entry["home_team"], "away_team": entry["away_team"],
                "home_resolved": home_id is not None, "away_resolved": away_id is not None,
            })
            continue

        for name, fighter_id, method in [
            (entry["home_team"], home_id, home_method),
            (entry["away_team"], away_id, away_method)
        ]:
            if method == "fuzzy":
                new_aliases.append({"fighter_id": fighter_id, "alias_name": name})

        entry_fighter_map = {entry["home_team"]: home_id, entry["away_team"]: away_id}

        for bookmaker in entry.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                collected_at = market["last_update"]
                for outcome in market["outcomes"]:
                    fighter_id = entry_fighter_map.get(outcome["name"])
                    if fighter_id is None:
                        continue  # shouldn't happen -- outcome names should always be home/away
                    moneyline = outcome["price"]
                    ready_rows.append({
                        "bout_id": bout_id,
                        "fighter_id": fighter_id,
                        "sportsbook": bookmaker["title"],
                        "moneyline": moneyline,
                        "implied_prob": _moneyline_to_implied_prob(moneyline),
                        "collected_at": collected_at,
                    })

    return ready_rows, new_aliases, unresolved_log

def write_fighter_aliases(engine, new_aliases: list[dict]) -> int:
    """Upserts on (fighter_id, alias_name) -- the constraint already exists
    from Week 1 (uq_fighter_aliases_fighter_alias). DO NOTHING on conflict:
    an alias, once confirmed, never needs updating, only insertion."""
    if not new_aliases:
        return 0

    metadata = MetaData()
    aliases_tbl = Table("fighter_aliases", metadata, autoload_with=engine)

    records = _clean_for_insert(pd.DataFrame(new_aliases), ["fighter_id", "alias_name"])
    stmt = pg_insert(aliases_tbl).values(records)
    stmt = stmt.on_conflict_do_nothing(index_elements=["fighter_id", "alias_name"])

    with engine.begin() as conn:
        result = conn.execute(stmt)
        return result.rowcount

def load_odds_snapshots(engine, ready_rows: list[dict]) -> int:
    if not ready_rows:
        return 0

    # ON CONFLICT DO UPDATE can't affect the same target row twice in one
    # statement. The Odds API occasionally lists the same fight under more
    # than one entry in a single snapshot (same pattern as Greco's own
    # duplicate-bout listing) -- dedupe on the exact constraint key before
    # inserting. Flag loudly if two "duplicates" actually disagree on
    # price -- that's a real inconsistency, not a harmless repeat.
    key_cols = ("bout_id", "fighter_id", "sportsbook", "collected_at")
    seen = {}
    for row in ready_rows:
        key = tuple(row[c] for c in key_cols)
        if key in seen and seen[key]["moneyline"] != row["moneyline"]:
            print(f"load_odds_snapshots: conflicting duplicate for {key} -- "
                  f"{seen[key]['moneyline']} vs {row['moneyline']}, keeping latest")
        seen[key] = row
    ready_rows = list(seen.values())

    metadata = MetaData()
    snapshots_tbl = Table("odds_snapshots", metadata, autoload_with=engine)

    insert_cols = ["bout_id", "fighter_id", "sportsbook", "moneyline", "implied_prob", "collected_at"]
    records = _clean_for_insert(pd.DataFrame(ready_rows), insert_cols)

    stmt = pg_insert(snapshots_tbl).values(records)
    update_cols = {c: stmt.excluded[c] for c in ("moneyline", "implied_prob")}
    stmt = stmt.on_conflict_do_update(
        index_elements=["bout_id", "fighter_id", "sportsbook", "collected_at"],
        set_=update_cols,
    ).returning(snapshots_tbl.c.id)

    with engine.begin() as conn:
        result = conn.execute(stmt)
        return len(result.fetchall())