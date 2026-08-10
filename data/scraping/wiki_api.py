import json
import requests
from .fetch import fetch

WIKI_API = "https://en.wikipedia.org/w/api.php"

class SectionNotFoundError(Exception):
    pass

def get_page_info(page: str, use_cache: bool = True) -> dict:
    """Returns {'pageid': int, 'title': str} — resolves redirects. Metadata only, no page content"""
    raw = fetch(WIKI_API, params={
        "action": "query", "titles": page, "format": "json", "redirects": 1
    }, use_cache=use_cache)
    pages = json.loads(raw)["query"]["pages"]
    page_data = next(iter(pages.values()))
    return {"pageid": page_data["pageid"], "title":  page_data["title"]}

def get_section_index(page: str, section_title: str, use_cache: bool = True) -> str:
    raw = fetch(WIKI_API, params={
        "action": "parse", "page": page, "prop": "sections", "format": "json", "redirects": 1
    }, use_cache=use_cache)
    sections = json.loads(raw)["parse"]["sections"]
    for s in sections:
        if s["line"] == section_title:
            return s["index"]
    available = [s["line"] for s in sections]
    raise SectionNotFoundError(
        f"No section '{section_title}' on page '{page}'. Available: {available}"
    )

def get_section_wikitext(page: str, section_index: str, use_cache: bool = True) -> str:
    raw = fetch(WIKI_API, params={
        "action": "parse", "page": page, "prop": "wikitext",
        "section": section_index, "format": "json", "redirects": 1
    }, use_cache=use_cache)
    return json.loads(raw)["parse"]["wikitext"]["*"]

def resolve_event_pageids(events: list[dict], use_cache: bool = True) -> list[dict]:
    """
    Enrich parsed scheduled-event dicts (from parse_scheduled_events) with a
    stable Wikipedia pageid, resolved via get_page_info().

    Each input dict must have an 'event_title' key (the wikilink target
    pulled from the Scheduled events table). Returns new dicts with two
    additional keys:
      - 'pageid': int, or None if resolution failed
      - 'resolved_title': str, the current title after following any
        redirect — may differ from 'event_title' if the page has since
        been renamed

    Resolution failures (e.g. a linked page that no longer exists, or a
    transient network error) are caught per-event so one bad row doesn't
    abort the whole batch — the event is still returned, just with
    pageid=None. Callers must check for None before using an event
    downstream. (Real failure logging, matching the fighter-match-failure
    convention, is wired in with the loader in a later step — this just
    prints for now.)
    """
    enriched = []
    for event in events:
        try:
            info = get_page_info(event["event_title"], use_cache=use_cache)
            pageid = info["pageid"]
            resolved_title = info["title"]
        except (KeyError, requests.exceptions.RequestException) as e:
            print(f"Warning: could not resolve pageid for '{event['event_title']}': {e}")
            pageid = None
            resolved_title = None

        enriched.append({
            **event,
            "pageid": pageid,
            "resolved_title": resolved_title,
        })

    return enriched

