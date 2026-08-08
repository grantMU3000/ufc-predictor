"""
Cache-first, rate-limited HTTP fetcher for ufcstats.com.

Every other scraper module (events.py, bouts.py, fighters.py) should call
fetch() instead of making requests directly. This is the ONLY place that
touches the network, which means it's the only place we need to get rate
limiting and caching right.
"""

import hashlib
import time
from pathlib import Path

import requests

# Where cached HTML pages get stored on disk.
RAW_DATA_DIR = Path("data/raw")

MIN_DELAY_SECONDS = 1.0  # Politely scraping


# Labeled traffic is less suspicious
HEADERS = {
    "User-Agent": "ufc-predictor/0.1 (https://github.com/grantMU3000/ufc-predictor; robinsonjg64@gmail.com)"
}

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0

_last_request_time = 0.0

def _cache_path(url: str) -> Path:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return RAW_DATA_DIR / f"{url_hash}.html"

def _respect_rate_limit() -> None:
    """Sleep, if needed, so we never request more than once per second."""
    elapsed = time.monotonic() - _last_request_time
    if elapsed < MIN_DELAY_SECONDS:
        time.sleep(MIN_DELAY_SECONDS - elapsed)

# Status codes worth retrying even though they're 4xx — these represent
# transient conditions (rate limiting, timeouts) rather than a genuinely
# invalid request.
RETRYABLE_STATUS_CODES = {408, 409, 429}

def fetch(url: str, params: dict | None = None, use_cache: bool = True) -> str:
    """
    Fetch a page's HTML/JSON, using a local cache when available.
    Pass use_cache=False for sources that change over time (e.g. Wikipedia
    upcoming-event pages) — the response is still written to cache for
    debugging, just never read back.
    """
    global _last_request_time

    if params:
        url = requests.Request("GET", url, params=params).prepare().url

    cache_file = _cache_path(url)

    if use_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        _respect_rate_limit()

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)

            # 408/409/429 mean "try again," so they fall through to the
            # retry path below instead.
            if 400 <= response.status_code < 500 and response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()

            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            assert e.response is not None  # always set: raised from response.raise_for_status()
            status = e.response.status_code
            if 400 <= status < 500 and status not in RETRYABLE_STATUS_CODES:
                raise  # genuinely invalid request. Not wasting retries
            last_exception = e

            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after is not None:
                    time.sleep(float(retry_after))
                    continue

        except requests.exceptions.RequestException as e:
            last_exception = e

        else:
            _last_request_time = time.monotonic()
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(response.text, encoding="utf-8")
            return response.text

        if attempt < MAX_RETRIES:
            wait = BACKOFF_BASE_SECONDS ** attempt
            time.sleep(wait)

    assert last_exception is not None  # loop only exits here after an exception was recorded
    raise last_exception
