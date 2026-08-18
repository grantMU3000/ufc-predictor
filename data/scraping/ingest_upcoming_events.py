"""
End-to-end run: list page -> parsed events -> pageid resolution ->
per-event fight card -> fighter resolution -> DB load.
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
from data.scraping.wiki_parsers import parse_fight_card, parse_scheduled_events


def run(use_cache: bool = False) -> None:
    engine = _get_engine()
    roster = FighterRoster.load(engine)

    index = get_section_index(
        "List_of_UFC_events", "Scheduled events", use_cache=use_cache
    )
    wikitext = get_section_wikitext("List_of_UFC_events", index, use_cache=use_cache)
    events = parse_scheduled_events(wikitext)

    resolved_events = []
    for event in events:
        info = get_page_info(event["event_title"], use_cache=use_cache)
        resolved_events.append(
            {
                "name": info["title"],
                "event_date": event["date"],
                "venue": event["venue"],
                "location": event["location"],
                "wikipedia_pageid": info["pageid"],
            }
        )

    events_df = pd.DataFrame(resolved_events)
    pageid_to_event_id = load_upcoming_events(engine, events_df)
    print(f"  {len(pageid_to_event_id)} upcoming events upserted")

    for event in resolved_events:
        event_id = pageid_to_event_id[event["wikipedia_pageid"]]
        try:
            card_index = get_section_index(
                event["name"], "Fight card", use_cache=use_cache
            )
        except Exception as e:  # noqa: BLE001 — one bad event page shouldn't kill the whole run
            print(f"Skipping fight card for '{event['name']}': {e}")
            continue

        card_wikitext = get_section_wikitext(
            event["name"], card_index, use_cache=use_cache
        )
        for i, bout in enumerate(parse_fight_card(card_wikitext)):
            red = resolve_fighter(
                engine,
                roster,
                bout["fighter_red"],
                wikipedia_link_target=bout["fighter_red_link_target"],
            )
            blue = resolve_fighter(
                engine,
                roster,
                bout["fighter_blue"],
                wikipedia_link_target=bout["fighter_blue_link_target"],
            )

            if red.match_type == "collision" or blue.match_type == "collision":
                continue  # already logged — needs manual resolution, don't insert a bad row

            red_id = red.fighter_id or create_stub_fighter(
                engine, roster, bout["fighter_red"]
            )
            blue_id = blue.fighter_id or create_stub_fighter(
                engine, roster, bout["fighter_blue"]
            )

            load_bout(engine, event_id, bout, red_id, blue_id, is_main_event=(i == 0))


run()
