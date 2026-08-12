"""
One-off manual test: run the full upcoming-events pipeline against a single
known event (UFC 330) before trusting it against the full list. Delete or
fold into a real pytest file once the integration is confirmed working.
"""

import pandas as pd

from data.ingestion.fighter_resolution import (
    FighterRoster,
    create_stub_fighter,
    resolve_fighter,
)
from data.ingestion.loaders import _get_engine
from data.ingestion.upcoming_events_loader import load_bout, load_upcoming_events
from data.scraping.wiki_api import (
    get_page_info,
    get_section_index,
    get_section_wikitext,
)
from data.scraping.wiki_parsers import parse_fight_card

USE_CACHE = True  # iterating locally — flip to False for a final live check


def main():
    engine = _get_engine()
    roster = FighterRoster.load(engine)

    # 1. Resolve the event itself
    info = get_page_info("UFC 330", use_cache=USE_CACHE)
    print(f"Resolved: {info}")

    event = {
        "name": info["title"],
        "event_date": "2026-08-15",  # from the list-page snapshot, hardcoded for this test only
        "venue": "Xfinity Mobile Arena",
        "location": "Philadelphia, Pennsylvania, U.S.",
        "wikipedia_pageid": info["pageid"],
    }
    events_df = pd.DataFrame([event])
    pageid_to_event_id = load_upcoming_events(engine, events_df)
    event_id = pageid_to_event_id[info["pageid"]]
    print(f"events.id = {event_id}")

    # 2. Fight card
    card_index = get_section_index(info["title"], "Fight card", use_cache=USE_CACHE)
    card_wikitext = get_section_wikitext(info["title"], card_index, use_cache=USE_CACHE)
    bouts = parse_fight_card(card_wikitext)
    print(f"Parsed {len(bouts)} bouts")

    # 3. Fighter resolution + bout load
    for bout in bouts:
        red = resolve_fighter(
            engine, roster, bout["fighter_red"],
            wikipedia_link_target=bout["fighter_red_link_target"],
        )
        blue = resolve_fighter(
            engine, roster, bout["fighter_blue"],
            wikipedia_link_target=bout["fighter_blue_link_target"],
        )
        print(f"  {bout['fighter_red']} [{red.match_type}] vs "
              f"{bout['fighter_blue']} [{blue.match_type}]")

        if red.match_type == "collision" or blue.match_type == "collision":
            print("    -> skipped, needs manual collision resolution")
            continue

        red_id = red.fighter_id or create_stub_fighter(engine, roster, bout["fighter_red"])
        blue_id = blue.fighter_id or create_stub_fighter(engine, roster, bout["fighter_blue"])

        load_bout(engine, event_id, bout, red_id, blue_id)

    # 4. Verify what actually landed in the DB
    print("\n--- events row ---")
    print(pd.read_sql(f"SELECT * FROM events WHERE id = {event_id}", engine))

    print("\n--- bouts rows ---")
    print(pd.read_sql(f"SELECT * FROM bouts WHERE event_id = {event_id}", engine))


if __name__ == "__main__":
    main()