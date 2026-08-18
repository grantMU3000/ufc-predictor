"""
Orchestrates the Greco1899 ingestion pipeline: parse -> transform -> load.

Run with: uv run python -m data.ingestion.run_ingest
"""

import shutil

from data.ingestion.loaders import load_all
from data.ingestion.transform import (
    LOG_DIR,
    _build_fighter_lookup,
    _build_source_url_lookup,
    build_bout_stats_table,
    build_bouts_table,
    build_events_table,
    build_fighters_table,
)


def run() -> tuple:
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)

    print("=== Step 1: fighters ===")
    fighters = build_fighters_table()
    print(f"fighters: {len(fighters)} rows")

    fighter_lookup = _build_fighter_lookup(fighters)
    source_url_lookup = _build_source_url_lookup(fighters)

    print("\n=== Step 2: events ===")
    events = build_events_table()
    print(f"events: {len(events)} rows")

    print("\n=== Step 3: bouts ===")
    bouts = build_bouts_table(events, fighter_lookup, source_url_lookup)
    print(f"bouts: {len(bouts)} rows")

    print("\n=== Step 4: bout_stats ===")
    bout_stats = build_bout_stats_table(bouts, fighter_lookup, source_url_lookup)
    print(f"bout_stats: {len(bout_stats)} rows")

    print("\n=== Step 5: load into Postgres ===")
    load_all(fighters, events, bouts, bout_stats)

    return fighters, events, bouts, bout_stats


if __name__ == "__main__":
    run()
