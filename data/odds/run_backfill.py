"""
Historical odds backfill: one snapshot per UFC event since June 2020,
matched to real bouts, loaded into odds_snapshots.

Run with: uv run python -m data.odds.run_backfill
"""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from data.ingestion.loaders import _get_engine
from data.odds.client import fetch_historical_odds
from data.odds.loaders import (
    build_alias_lookup,
    build_bout_lookup_from_db,
    build_real_name_lookup,
    load_odds_snapshots,
    resolve_and_prepare_snapshot_rows,
    write_fighter_aliases,
)
from data.odds.parsers import filter_by_event_date

LOG_PATH = Path("data/odds/logs/unresolved_odds_entries.csv")


def get_target_events(engine, limit: int | None = None) -> list[dict]:
    query = """
        SELECT id, name, event_date FROM events
        WHERE event_date >= '2020-06-06'
        ORDER BY event_date DESC
    """
    if limit:
        query += f" LIMIT {limit}"
    with engine.begin() as conn:
        rows = conn.execute(text(query)).fetchall()
    return [dict(r._mapping) for r in rows]


def run():
    engine = _get_engine()

    print("Building lookups...")
    real_name_lookup = build_real_name_lookup(engine)
    alias_lookup = build_alias_lookup(engine)
    bout_lookup = build_bout_lookup_from_db(engine)

    events = get_target_events(engine)
    print(f"{len(events)} target events (since June 2020)\n")

    totals = {"snapshots": 0, "aliases": 0, "unresolved": 0}
    unresolved_log_rows = []

    for i, event in enumerate(events, 1):
        query_date = (
            datetime.combine(event["event_date"], datetime.min.time(), tzinfo=UTC)
            .replace(hour=12)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        print(f"[{i}/{len(events)}] {event['name']} ({query_date})...", end=" ")

        raw = fetch_historical_odds(query_date)
        if raw is None:
            print("FETCH FAILED, skipped")
            continue

        filtered = filter_by_event_date(raw["data"], event["event_date"])
        ready_rows, new_aliases, unresolved = resolve_and_prepare_snapshot_rows(
            filtered, real_name_lookup, alias_lookup, bout_lookup
        )

        if new_aliases:
            write_fighter_aliases(engine, new_aliases)
            # keep the in-memory lookup current so later events this same
            # run benefit immediately, not just on the next full run
            for a in new_aliases:
                alias_lookup[a["alias_name"]] = a["fighter_id"]

        snapshots_written = load_odds_snapshots(engine, ready_rows)

        for u in unresolved:
            u["event_name"] = event["name"]
        unresolved_log_rows.extend(unresolved)

        totals["snapshots"] += snapshots_written
        totals["aliases"] += len(new_aliases)
        totals["unresolved"] += len(unresolved)

        print(
            f"{snapshots_written} snapshots, {len(new_aliases)} new aliases, "
            f"{len(unresolved)} unresolved"
        )

    print(
        f"\nDone. {totals['snapshots']} snapshots loaded, "
        f"{totals['aliases']} aliases confirmed, "
        f"{totals['unresolved']} unresolved."
    )

    if unresolved_log_rows:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unresolved_log_rows).to_csv(LOG_PATH, index=False)
        print(f"Unresolved entries -> {LOG_PATH}")


if __name__ == "__main__":
    run()
