"""
Transferring Historical Odds into my database
"""

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Reused rather than duplicated -- same low-level helpers the Greco
# pipeline already uses (engine creation, NaN-safe record cleaning).
from data.ingestion.loaders import _clean_for_insert
from data.odds.matcher import build_bout_lookup, match_to_bout, resolve_fighter_name


def _moneyline_to_implied_prob(price: int) -> float:
    """American odds -> implied probability, e.g. -270 -> 0.7297, +220 -> 0.3125."""
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)


def build_real_name_lookup(engine) -> dict[str, int | list[int]]:
    """
    Map real_name -> fighter_id, collision-safe.

    Names shared by more than one real fighter (e.g. two different
    people both named "Bruno Silva") map to a LIST of candidate ids
    instead of silently keeping whichever row the SQL query happened to
    return last -- that's what the old plain dict comprehension did,
    and it's what was misattributing roughly half of Bruno Silva's real
    fights to the wrong person.

    Odds API responses carry no weight_class field, so this can't reuse
    Greco's COLLISION_RESOLUTIONS approach directly. Disambiguation for
    these names has to happen downstream, in
    resolve_and_prepare_snapshot_rows, by trying each candidate id
    against match_to_bout and keeping whichever one actually resolves
    to a real bout.
    """
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, real_name FROM fighters")).fetchall()

    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(row.real_name, []).append(row.id)

    return {name: ids[0] if len(ids) == 1 else ids for name, ids in grouped.items()}


def build_alias_lookup(engine) -> dict[str, int]:
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT fighter_id, alias_name FROM fighter_aliases")
        ).fetchall()
    return {row.alias_name: row.fighter_id for row in rows}


def build_bout_lookup_from_db(engine) -> dict[frozenset, list[dict]]:
    with engine.begin() as conn:
        # Could this statement have a WHERE clause that only gets fights from 2020 & later?
        rows = conn.execute(
            text("""
            SELECT b.id AS bout_id, b.fighter_red_id, b.fighter_blue_id, e.event_date
            FROM bouts b JOIN events e ON b.event_id = e.id
             WHERE e.event_date >= '2020-06-06'
        """)
        ).fetchall()
    return build_bout_lookup([dict(r._mapping) for r in rows])


def resolve_and_prepare_snapshot_rows(
    filtered_odds_data: list[dict],
    real_name_lookup: dict[str, int | list[int]],
    alias_lookup: dict[str, int],
    bout_lookup: dict[frozenset, list[dict]],
) -> tuple[list[dict], list[dict], list[dict]]:
    ready_rows, new_aliases, unresolved_log = [], [], []

    for entry in filtered_odds_data:
        home_result, home_method = resolve_fighter_name(
            entry["home_team"], real_name_lookup, alias_lookup
        )
        away_result, away_method = resolve_fighter_name(
            entry["away_team"], real_name_lookup, alias_lookup
        )

        home_candidates = (
            home_result
            if isinstance(home_result, list)
            else ([home_result] if home_result is not None else [])
        )
        away_candidates = (
            away_result
            if isinstance(away_result, list)
            else ([away_result] if away_result is not None else [])
        )

        # Try every (home, away) candidate pair against real bouts -- for
        # non-ambiguous names this is just one pair, same as before. For
        # a collision name (e.g. either real Bruno Silva), this is what
        # actually picks the CORRECT one: whichever candidate produces a
        # real bout match wins.
        bout_id = home_id = away_id = None
        for h in home_candidates:
            for a in away_candidates:
                matched = match_to_bout(h, a, entry["commence_time"], bout_lookup)
                if matched is not None:
                    bout_id, home_id, away_id = matched, h, a
                    break
            if bout_id is not None:
                break

        if bout_id is None:
            unresolved_log.append(
                {
                    "commence_time": entry["commence_time"],
                    "home_team": entry["home_team"],
                    "away_team": entry["away_team"],
                    "home_resolved": len(home_candidates) > 0,
                    "away_resolved": len(away_candidates) > 0,
                }
            )
            continue

        # Only an UNAMBIGUOUS fuzzy match is safe to alias. An ambiguous
        # name resolving correctly for THIS fight (via bout-matching)
        # says nothing about which real person that string means the
        # NEXT time it appears -- it could legitimately be the other one.
        for name, method, fid in [
            (entry["home_team"], home_method, home_id),
            (entry["away_team"], away_method, away_id),
        ]:
            if method == "fuzzy":
                new_aliases.append({"fighter_id": fid, "alias_name": name})

        entry_fighter_map = {entry["home_team"]: home_id, entry["away_team"]: away_id}

        for bookmaker in entry.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                collected_at = market["last_update"]
                for outcome in market["outcomes"]:
                    fighter_id = entry_fighter_map.get(outcome["name"])
                    if fighter_id is None:
                        continue
                    moneyline = outcome["price"]
                    ready_rows.append(
                        {
                            "bout_id": bout_id,
                            "fighter_id": fighter_id,
                            "sportsbook": bookmaker["title"],
                            "moneyline": moneyline,
                            "implied_prob": _moneyline_to_implied_prob(moneyline),
                            "collected_at": collected_at,
                        }
                    )

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
    seen: dict[tuple, dict] = {}
    for row in ready_rows:
        key = tuple(row[c] for c in key_cols)
        if key in seen and seen[key]["moneyline"] != row["moneyline"]:
            print(
                f"load_odds_snapshots: conflicting duplicate for {key} -- "
                f"{seen[key]['moneyline']} vs {row['moneyline']}, keeping latest"
            )
        seen[key] = row
    ready_rows = list(seen.values())

    metadata = MetaData()
    snapshots_tbl = Table("odds_snapshots", metadata, autoload_with=engine)

    insert_cols = [
        "bout_id",
        "fighter_id",
        "sportsbook",
        "moneyline",
        "implied_prob",
        "collected_at",
    ]
    records = _clean_for_insert(pd.DataFrame(ready_rows), insert_cols)

    stmt = pg_insert(snapshots_tbl).values(records)
    update_cols = {c: stmt.excluded[c] for c in ("moneyline", "implied_prob")}
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=["bout_id", "fighter_id", "sportsbook", "collected_at"],
        set_=update_cols,
    ).returning(snapshots_tbl.c.id)

    with engine.begin() as conn:
        result = conn.execute(upsert_stmt)
        return len(result.fetchall())
