import re
from datetime import date, datetime

import mwparserfromhell

EXPECTED_HEADERS = ["Event", "Date", "Venue", "Location", "Ref."]


def _first_wikilink(cell_wikicode) -> dict | None:
    """Return {'target': ..., 'display': ...} for the first wikilink in a cell, or None."""
    links = cell_wikicode.filter_wikilinks()
    if not links:
        return None
    link = links[0]
    target = str(link.title).strip()
    display = str(link.text).strip() if link.text is not None else target
    return {"target": target, "display": display}


def _parse_dts(cell_wikicode) -> date | None:
    """Extract a date from a {{dts|YYYY|Mon|DD}} template."""
    for template in cell_wikicode.filter_templates():
        if template.name.strip().lower() != "dts":
            continue
        params = [str(p.value).strip() for p in template.params if not p.showkey]
        if len(params) < 3:
            continue
        year, month, day = params[:3]
        for fmt in ("%Y %b %d", "%Y %B %d", "%Y %m %d"):
            try:
                # Calendar date only, no time component -- tzinfo doesn't apply.
                return datetime.strptime(f"{year} {month} {day}", fmt).date()  # noqa: DTZ007
            except ValueError:
                continue
    return None


def parse_scheduled_events(wikitext: str) -> list[dict]:
    """
    Parses the "Scheduled events" wikitable (List_of_UFC_events) into a list
    of event dicts: event_title (page title — use for pageid lookups),
    event_display_name, date, venue, location.
    """
    table_match = re.search(r"\{\|.*?\|\}", wikitext, re.DOTALL)
    if not table_match:
        raise ValueError("No wikitable found in 'Scheduled events' wikitext")
    table_text = table_match.group(0)

    # Header block is everything before the first row separator — check it
    # matches what we expect before trusting the column order below.
    header_block, *row_blocks = re.split(r"\n\|-\s*\n?", table_text)
    headers = [
        h.strip()
        for h in re.findall(
            r'^!\s*(?:scope="col"\s*\|\s*)?(.+)$', header_block, re.MULTILINE
        )
    ]
    if headers != EXPECTED_HEADERS:
        raise ValueError(f"Unexpected table columns: {headers}")

    events = []
    for block in row_blocks:
        block = block.strip()
        cell_lines = [
            line[1:].strip()
            for line in block.splitlines()
            if line.startswith("|") and not line.startswith("|}")
        ]
        if len(cell_lines) < 4:
            continue  # malformed/short row — skip rather than guess

        event_cell = mwparserfromhell.parse(cell_lines[0])
        date_cell = mwparserfromhell.parse(cell_lines[1])
        venue_cell = mwparserfromhell.parse(cell_lines[2])
        location_cell = mwparserfromhell.parse(cell_lines[3])

        event_link = _first_wikilink(event_cell)
        if event_link is None:
            continue  # no linked event page — nothing to resolve a pageid from

        events.append(
            {
                "event_title": event_link["target"],
                "event_display_name": event_link["display"],
                "date": _parse_dts(date_cell),
                "venue": venue_cell.strip_code().strip(),
                "location": location_cell.strip_code().strip(),
            }
        )

    return events


def _get_param(template, index: int) -> str:
    """Extract and clean positional param `index` (1-based) from a template, or '' if absent."""
    key = str(index)
    if template.has(key):
        value = str(template.get(key).value)
        return mwparserfromhell.parse(value).strip_code().strip()
    return ""


def _get_param_and_link(template, index: int) -> tuple[str, str | None]:
    """
    Extract positional param `index` (1-based) as (display_text, link_target).
    link_target is the wikilink's target page title if the param is linked
    (e.g. "Islam Makhachev", or a disambiguated title like
    "Bruno Silva (welterweight)"), or None if the fighter's name appears
    as plain unlinked text (e.g. no Wikipedia article yet).
    """
    key = str(index)
    if not template.has(key):
        return "", None

    value_wikicode = template.get(key).value
    links = value_wikicode.filter_wikilinks()
    link_target = str(links[0].title).strip() if links else None

    display_text = mwparserfromhell.parse(str(value_wikicode)).strip_code().strip()
    return display_text, link_target


def parse_fight_card(wikitext: str) -> list[dict]:
    """
    Parses a UFC event page's "Fight card" section wikitext into a list of
    bout dicts, in document order (main card, then prelims, then early prelims).
    """
    parsed = mwparserfromhell.parse(wikitext)
    bouts = []
    current_tier = None

    for template in parsed.filter_templates():
        name = template.name.strip()

        if name == "MMAevent card":
            current_tier = _get_param(template, 1)
            continue

        if name != "MMAevent bout":
            continue

        fighter_red_raw, fighter_red_link = _get_param_and_link(template, 2)
        fighter_blue_raw, fighter_blue_link = _get_param_and_link(template, 4)

        bouts.append(
            {
                "card_tier": current_tier,
                "weight_class": _get_param(template, 1),
                "fighter_red": fighter_red_raw.replace("(c)", "").strip(),
                "fighter_red_link_target": fighter_red_link,
                "fighter_red_is_champion": "(c)" in fighter_red_raw,
                "connector": _get_param(
                    template, 3
                ),  # "vs." = not yet fought, "def." = result recorded
                "fighter_blue": fighter_blue_raw.replace("(c)", "").strip(),
                "fighter_blue_link_target": fighter_blue_link,
                "fighter_blue_is_champion": "(c)" in fighter_blue_raw,
                "method": _get_param(template, 5),
                "round": _get_param(template, 6),
                "time": _get_param(template, 7),
                "notes": _get_param(template, 8),
            }
        )

    return bouts
