import json
from .fetch import fetch

WIKI_API = "https://en.wikipedia.org/w/api.php"

class SectionNotFoundError(Exception):
    pass

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
    }, use_cache=False)
    return json.loads(raw)["parse"]["wikitext"]["*"]