# Progress Log

Daily log of what shipped, what's blocked, and what's next. Written at the end of each work session (see "Wrap" block in the daily rhythm) and read first thing the next day to generate the day's plan.

**Format:** newest entry at the top. Keep entries short — this is a working log, not a report. If it takes more than 5 minutes to write, it's too long.

---

## Template

```
## YYYY-MM-DD (Week N, Day X)

**Planned today:** (copied from yesterday's "tomorrow" line, or agent-generated)

**Shipped:**
-
-

**Blocked / open questions:**
-

**Research (1hr):** topic — link to docs/research/YYYY-MM-DD-topic.md

**Tomorrow's first task:**

**Energy / notes:** (1 line — burnout is a tracked risk in this project, not an afterthought)

**Metrics check (weekly only, Fridays):** exit criteria vs. actual — anything slipping?
```

---

## Log

## 2026-08-10 (Week 1, Day 7)

**Planned today:** Wire `wiki_parsers.py` output into the actual ingestion path — persist parsed scheduled events and fight cards keyed by `pageid`.

**Shipped:**
- Fixed CI, red since Aug 8: ruff import-sort errors (which had been masking mypy not running), assorted real ruff hits, and 10 mypy errors; declared `mwparserfromhell` in `pyproject.toml`/`uv.lock` (`d7db47a`)
- Researched data leakage and reviewed the data split/transformation plan (`docs/research/2026-08-10-DataLeakage.md`) (`e423d3d`)
- Added `wikipedia_pageid` column to `events` via migration, to support keying upcoming events (`a429e0b`)
- Implemented the upcoming-events ingestion path: `data/ingestion/fighter_resolution.py`, `data/ingestion/upcoming_events_loader.py`, `data/scraping/ingest_upcoming_events.py` — wires `wiki_parsers.py` output through to persistence, closing out yesterday's "tomorrow" task. Surfaced fighter name collision bugs in the process, logged to `data/ingestion/logs/` (4 unresolved fighters, 4 alias collisions) (`e7afc59`)

**Blocked / open questions:**
-

**Research (1hr):** Data leakage — `docs/research/2026-08-10-DataLeakage.md`

**Tomorrow's first task:** Resolve the fighter name-collision bugs from today's ingestion run — 4 unresolved fighters (Choi Doo-ho, Osman Diaz, Yoo Joo-sang, Wesley Schultz). Also check whether `alias_collisions.jsonl` double-logging each row is a real dupe-write bug in `fighter_resolution.py`.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-09 (Week 1, Day 6)

**Planned today:** Add `mwparserfromhell` as a dependency and implement the parser that turns `{{MMAevent bout}}`/`{{MMAevent card}}` wikitext (from `get_section_wikitext`) into structured fight-card rows keyed by `pageid`.

**Shipped:**
- Added `mwparserfromhell` dependency (`pyproject.toml`, `uv.lock`)
- Implemented `data/scraping/wiki_parsers.py`: `parse_scheduled_events` (parses the "Scheduled events" wikitable into event dicts with title/display name/date/venue/location) and `parse_fight_card` (parses `{{MMAevent bout}}`/`{{MMAevent card}}` templates into bout dicts — tier, weight class, fighters, champion flags, method/round/time/notes)
- Added `get_page_info` to `data/scraping/wiki_api.py` to resolve a page title to its stable `pageid`, for keying event pages later
- (Uncommitted) `data/scraping/_manual_test.py` — manual smoke test against live Wikipedia data (List_of_UFC_events scheduled-events parsing, pageid resolution, UFC 330 fight-card parsing. This is a throwaway script

**Blocked / open questions:**
-

**Research (1hr):** —

**Tomorrow's first task:** Wire `wiki_parsers.py` output into the actual ingestion path — persist parsed scheduled events and fight cards keyed by `pageid`.

**Energy / notes:**
- Short day today. Will have to do a lot of work tomorrow with the API and data quality suite

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-08 (Week 1, Day 5)

**Planned today:** Audit the 1,855 unresolved odds entries in `data/odds/logs/unresolved_odds_entries.csv` — categorize why each failed to match, resolve or write off as many as possible, then confirm final odds coverage against the ~3,200-fight target.

**Shipped:**
- Resolved name conflicts blocking ~200 bouts from odds matching; final backfilled historical odds now at 2,775 bouts / 59,972 snapshots (`data/odds/loaders.py`, `matcher.py`) — unresolved log down to 1,718 remaining rows
- Researched fuzzy matching / entity resolution for scraped-to-database fighter/event matching — `docs/research/2026-08-08-EntityResolution.md`
- Tested Wikipedia's MediaWiki API response for the UFC 330 fight card (`scripts/wiki_api_test.py`) — found event pages use a `{{MMAevent bout}}` template with a stable `pageid`, not a plain wikitable
- Implemented `data/scraping/wiki_api.py` lookup functions (`get_section_index`, `get_section_wikitext`) and extended `fetch.py` with query-param support + a `use_cache` bypass
- Pivoted upcoming-events/rankings retrieval from HTML scraping to the MediaWiki Action API — logged as ADR-009 in `docs/DECISIONS.md`, updated `docs/PLAN_ADDENDUM.md` accordingly
- (Untracked, not yet committed) `scripts/2020_fight_amount.py`

**Blocked / open questions:**
-

**Research (1hr):** Fuzzy matching / entity resolution — `docs/research/2026-08-08-EntityResolution.md`

**Tomorrow's first task:** Add `mwparserfromhell` as a dependency and implement the parser that turns `{{MMAevent bout}}`/`{{MMAevent card}}` wikitext (from `get_section_wikitext`) into structured fight-card rows keyed by `pageid` — the piece ADR-009 commits to but `wiki_api.py` doesn't have yet. (The 1,718 remaining unresolved odds entries from yesterday are also still open if you'd rather pick that back up.)

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-07 (Week 1, Day 4)

**Planned today:** Start Thursday's odds ingestion — pay for one month of The Odds API historical tier, backfill MMA moneylines to 2020, join to bouts by (date, fighter pair). Expect ~10% match failures — log them, don't silently drop.

**Shipped:**
- Researched Alembic migrations — `docs/research/2026-08-07-Alembic.md`
- Built `data/odds/` module (`client.py`, `matcher.py`, `parsers.py`) and `scripts/odds_api_small.py` to pull MMA odds from The Odds API and narrow results to the exact time window needed for historical bouts
- Discovered The Odds API only covers MMA fights since June 2020 (~3,200 fights, below the original 4,000-bout target) — revised the odds-coverage exit criterion in `docs/DECISIONS.md`; added `scripts/2020_fight_amount.py` to quantify events/bouts available since the June 2020 floor
- Built `data/odds/loaders.py` and `run_backfill.py` and backfilled historical bout odds; added an Alembic migration for a unique constraint on `odds_snapshots`
- Logged 1,855 unresolved odds entries to `data/odds/logs/unresolved_odds_entries.csv` for follow-up (not yet audited)
- (Uncommitted) Backfilled the "How this applies to my project" section and time spent on yesterday's scraping research doc; updated `manager`/`planner` skills to also read `docs/PLAN_ADDENDUM.md`

**Blocked / open questions:**
-

**Research (1hr):** Alembic migrations — `docs/research/2026-08-07-Alembic.md`

**Tomorrow's first task:** Audit the 1,855 unresolved odds entries in `data/odds/logs/unresolved_odds_entries.csv` — categorize why each failed to match (name variants, date mismatches, event not found, etc.) and resolve or explicitly write off as many as possible, then confirm final odds coverage against the revised ~3,200-fight target.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-06 (Week 1, Day 3)

**Planned today:** Finish `loaders.py` (Postgres insert in FK-safe order), resolve the 8 known fighter name collisions, build `run_ingest.py` orchestration.

**Shipped:**
- Resolved the 8 known cross-fighter name collisions and event-name collisions in `transform.py`/`parsers.py`
- Built `run_ingest.py` orchestration (initial version without loader, then wired to `loaders.py`)
- Implemented `loaders.py`: FK-safe Postgres upserts keyed on `source_url`, with small schema tweaks (nullable `control_time_seconds`, unique constraint on `events.source_url`) needed to make it run end to end
- Added `preserve_on_conflict` support to the upsert helper — locks specific columns (e.g. `ufc_debut_date`) so a later backfill process isn't clobbered by re-ingestion
- Backfilled `fighters.ufc_debut_date` from each fighter's earliest bout `event_date`, idempotent and safe to rerun
- Logged ADR-007: excluding "Road to UFC" rows from the bout dataset (2 rows, no matching event — deliberate scope decision, not a data-quality bug)
- Added a `teacher` skill for concept/big-picture explanations alongside the coding work
- Filled in the "How this applies to my project" section of yesterday's scraping research writeup
- Confirmed post-ingest row counts against Postgres — bout count remains above the 8,000-bout Week 1 exit threshold after collision exclusions

**Blocked / open questions:**
-

**Research (1hr):** Idempotent upsert patterns in Postgres — `docs/research/2026-08-06-Idempotent.md`

**Tomorrow's first task:** Start Thursday's odds ingestion — pay for one month of The Odds API historical tier, backfill MMA moneylines to 2020, join to bouts by (date, fighter pair). Expect ~10% match failures — log them, don't silently drop.

**Energy / notes:**
- For UFC debuts, there are missing dates for close to half the fighters. This is because Greco's scraped data, not an issue on my end. It includes fighters that just simply don't appear to have any fights for the promotion.

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-05 (Week 1, Day 2)

**Planned today:** Build a ufcstats.com scraper (events index → event pages → bout pages), rate-limited and resumable, with raw HTML cached to `data/raw/`.

**Shipped:**
- Researched scraping best practices (rate limiting, monitoring/validating responses) — `docs/research/2026-08-05-Scraping.md`
- Built `fetch()`: a cache-first, rate-limited HTTP fetcher for ufcstats.com with retry/backoff
- Discovered ufcstats.com blocks non-JS clients; pivoted to `Greco1899/scrape_ufc_stats` CSVs as the primary data source instead of a custom scraper (decision logged)
- Wrote parsers for the Greco1899 CSVs; extended the schema to capture the missing "reversals" stat and added source URLs to `fighters`/`bouts`
- Resolved parsing bugs and future-bout join issues
- Wrote (but haven't yet run) a transform step joining/reshaping the parsed data — see notes in commit `b38e609` before running it
- Cleared CI: fixed ruff `PLW0602` (unneeded `global`) and `I001` (import sorting), fixed mypy errors (missing `pandas-stubs`, unguarded `Optional` access in `fetch.py`), fixed a pytest collection error (scoped test discovery to `tests/`)
- Added additional plan information in docs/PLAN_ADDENDUM.md
- Updated the planner skill to surface a research topic each day

**Blocked / open questions:**

**Research (1hr):** Scraping best practices — `docs/research/2026-08-05-Scraping.md`

**Tomorrow's first task:** Run the transform step (`data/ingestion/transform.py`) against the parsed Greco1899 data — check the notes from `b38e609` first — then verify the loaded output against Postgres.

**Energy / notes:**
- Diverged from the original plan. I'm going to use Wikipedia's API for future events since it seems more reliable than ESPN API/scraping.
- Also, the Greco1899 repo is the source of truth for UFC fight stats for now. This'll remain the case until I find a way to scrape UFC stats
- Update docs/PLAN.md eventually

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-04 (Week 1, Day 1)

**Planned today:** Commit the pending `pyproject.toml`/`uv.lock` changes (duckdb, pandas, python-dotenv added), then finish the schema design & migration for `events`, `fighters`, `bouts`, `bout_stats`, `predictions`, `prediction_results`.

**Shipped:**
- Decided to exclude draws & no-contests from training data; rationale logged in `docs/DECISIONS.md`
- Designed the initial Postgres schema and applied the Alembic migration to Neon (`events`, `fighters`, `bouts`, `bout_stats`, `predictions`, `prediction_results`)
- Justified the red/blue corner design decision and bout table design in `docs/DECISIONS.md`
- Fixed CI: removed ~69k lines of raw scraped CSVs that had been accidentally committed, adjusted `migrations/env.py`
- Declared `alembic` + `psycopg2-binary` in `pyproject.toml`/`uv.lock` — CI's mypy step was failing because they were resolved locally but never declared

**Blocked / open questions:**
-

**Research (1hr):** Postgres indexing — `docs/research/2026-08-04-Indexing.md`

**Tomorrow's first task:** Write your own ufcstats.com scraper (events index → event page → bout page → fighter page), rate-limited (1 req/sec) and resumable, with raw HTML cached to `/data/raw/`.

**Energy / notes:**
- I plan to do Wednesday tasks while I wait for the scraper to finish doing what it needs to do

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-03 (Week 0, Day 2)

**Planned today:** Repo scaffolding, Neon project, pull Greco1899 CSVs into DuckDB

**Shipped:**
- Repo initialized, MIT license added, `.gitignore` covers `.env`/`.venv`/`__pycache__`
- Neon Postgres project created, connection string in `.env`, sanity-checked with a test connection
- `docs/DECISIONS.md` and `docs/PROGRESS.md` scaffolded
- Agentic workflow scaffolded: `CLAUDE.md` + Planner/Scribe/Manager/Reviewer skills; scribe updated to check logs since 7 AM
- Loaded pre-scraped Greco1899 UFC CSVs into DuckDB, ran a quick analysis script over them
- Researched Postgres schema design & indexing; used it to start the initial database schema

**Blocked / open questions:**
- Will I need to use indexing within my database?
- What are the different indexing methods that are supported by PostgreSQL?
-

**Research (1hr):** Postgres schema design & indexing

**Tomorrow's first task:** Commit the pending `pyproject.toml`/`uv.lock` changes (duckdb, pandas, python-dotenv added), then finish the schema design & migration for `events`, `fighters`, `bouts`, `bout_stats`, `predictions`, `prediction_results`.

**Energy / notes:**
- I think I'll do more research tomorrow on schemas & indexing
- I also added two entities to the schema that I'm working through: odds snapshots & fighter aliases
- Fighter aliases is used to track various names that a fighter goes by & resolve those differences
- Odds snapshots go over the odds of each bout

**Metrics check (weekly only, Fridays):** —

---

<!-- Add new entries above this line, newest first -->