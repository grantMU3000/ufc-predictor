import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["ODDS_API_KEY"]
url = "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/odds"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "h2h",
    "oddsFormat": "american",
    "date": "2026-07-11T12:00:00Z",  # near one of your known event dates
}
resp = requests.get(url, params=params)
print("Status:", resp.status_code)
print("Credits used this call:", resp.headers.get("x-requests-used"))
print("Credits remaining:", resp.headers.get("x-requests-remaining"))

if resp.status_code == 200:
    cache_dir = Path("data/raw/odds")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{params['date']}.json".replace(":", "-")
    cache_path.write_text(json.dumps(resp.json(), indent=2))
    print("Cached to:", cache_path)