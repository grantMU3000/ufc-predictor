from .wiki_api import get_section_index, get_section_wikitext, resolve_event_pageids
from .wiki_parsers import parse_scheduled_events

# 1. Scheduled events — should return 11 rows against today's list-page snapshot
index = get_section_index("List_of_UFC_events", "Scheduled events", use_cache=False)
wikitext = get_section_wikitext("List_of_UFC_events", index, use_cache=False)
events = parse_scheduled_events(wikitext)
print(f"Parsed {len(events)} scheduled events")
assert len(events) == 11, f"expected 11, got {len(events)}"

"""
# spot-check the two piped-link rows specifically
by_title = {e["event_title"]: e for e in events}
assert by_title["UFC 331"]["event_display_name"] == "UFC 331: Van vs. Pantoja 2"
assert by_title["UFC 330"]["event_display_name"] == "UFC 330: Makhachev vs. Machado Garry"

# 2. pageid resolution — run it on one event, confirm it doesn't blow up post-fix
info = get_page_info("UFC 330", use_cache=False)
print(info)
assert isinstance(info["pageid"], int)

# 3. fight card parsing on the same event
card_index = get_section_index("UFC 330", "Fight card", use_cache=False)
card_wikitext = get_section_wikitext("UFC 330", card_index, use_cache=False)
bouts = parse_fight_card(card_wikitext)
print(f"Parsed {len(bouts)} bouts")
assert len(bouts) == 11  # 4 main card + 4 prelim + 3 early prelim, per what you pasted earlier
assert bouts[0]["fighter_red"] == "Islam Makhachev"
assert bouts[0]["fighter_red_is_champion"] is True

print("All checks passed")
"""
events = parse_scheduled_events(wikitext)
enriched = resolve_event_pageids(events, use_cache=False)

failed = [e for e in enriched if e["pageid"] is None]
print(f"Resolved {len(enriched) - len(failed)}/{len(enriched)} events")
