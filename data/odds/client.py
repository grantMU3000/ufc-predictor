"""
This module gathers historical MMA match odds from Odds API. These odds are 
cached and will be filtered to just UFC bouts in subsequent modules. 
"""
import json, os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ODDS_API_KEY"]
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/odds"
CACHE_DIR = Path("data/raw/odds")

def fetch_historical_odds(query_date: str) -> dict | None:
    """
    query_date: ISO8601 string, e.g. "2026-07-11T12:00:00Z".
    Cache-first -- never re-spends credits on a date already fetched
    successfully. Returns the parsed JSON, or None on failure (caller
    logs and continues rather than crashing the whole backfill over one
    bad call, same discipline as fetch.py's retry handling).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{query_date.replace(':', '-')}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    params = {
        "apiKey": API_KEY, "regions": "us", "markets": "h2h",
        "oddsFormat": "american", "date": query_date,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)

    if resp.status_code != 200:
        print(f"fetch_historical_odds: FAILED for {query_date} "
              f"(status {resp.status_code}): {resp.text[:200]}")
        return None

    data = resp.json()
    cache_path.write_text(json.dumps(data))
    return data