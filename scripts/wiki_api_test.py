import requests

HEADERS = {"User-Agent": "ufc-predictor/0.1 (robinsonjg64@gmail.com)"}
BASE = "https://en.wikipedia.org/w/api.php"

# 1. Get section index
r = requests.get(BASE, params={
    "action": "parse", "page": "UFC_330", "prop": "sections", "format": "json"
}, headers=HEADERS)
sections = r.json()["parse"]["sections"]
fight_card = next(s for s in sections if s["line"] == "Fight card")
print(fight_card["index"])

# 2. Get that section's wikitext
r = requests.get(BASE, params={
    "action": "parse", "page": "UFC_330", "prop": "wikitext",
    "section": fight_card["index"], "format": "json"
}, headers=HEADERS)
wikitext = r.json()["parse"]["wikitext"]["*"]
print(wikitext)