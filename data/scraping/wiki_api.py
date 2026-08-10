import json

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

