# UFC Fight Predictor — Plan Addendum: Data Source Pivot

**Date:** 2026-08-05
**Applies to:** `docs/PLAN.md`
**Status:** Active — supersedes the sections listed below until merged back into the main plan

This addendum captures changes made during Week 1 (Mon–Tue) that affect the original plan's
assumptions about data sources and ingestion architecture. Nothing about the project's goals,
timeline, or feature/model plan changes — this is entirely about *how data gets in*, and it
changes because of a discovery, not a preference.

---

## 1. Why this changed (short version)

While building the Tuesday scraper (`fetch.py`), requests to `ufcstats.com` consistently
returned a JavaScript proof-of-work challenge page instead of real content. Diagnosis (browser
vs. bare `curl` vs. `curl` with a spoofed browser User-Agent, all from the same IP) confirmed
this is a deliberate, permanent gate against any non-browser client — not an IP flag, not a
header heuristic, not intermittent. Full reasoning and alternatives considered are in
**ADR-006** (`docs/DECISIONS.md`).

Separately, manually spot-checking ESPN's public rankings endpoint turned up an outdated
fighter still listed in current rankings — enough to disqualify it as a default/primary source
without per-use verification.

Net effect: **no first-party scraper against ufcstats.com**, and **ESPN moves from "preferred"
to "fallback, verify before trusting."**

---

## 2. Updated: §1 Data sources table

Replace the original table with:

| Source | What | Status |
|---|---|---|
| **Greco1899 CSVs** | Every event, bout, round-by-round stats, fighter bios | **Primary and current sole source**, until further notice. Pre-scraped from ufcstats.com by a third-party pipeline (their own daily GCP Cloud Run job), refreshed regularly. See §4 for the reliability caveat. |
| **ufcstats.com (direct)** | — | **Not used.** Blocks all non-browser access via an enforced JS proof-of-work challenge (ADR-006). Not a `robots.txt` restriction — an active technical barrier. Revisit only if the site's posture changes. |
| **Wikipedia** | UFC event schedule, official divisional rankings | **Primary** for both upcoming-events schedule and rankings, accessed via the MediaWiki API (ADR-009). Human-maintained, sourced from official UFC.com, not bot-gated. Events are keyed on the stable `wikipedia_pageid`, not page title, since titles get renamed as fight cards are confirmed/reshuffled (ADR-011). |
| **ESPN public MMA API** | Event schedule, rankings, fighter bios | **Fallback only.** Observed to return stale rankings data (a since-departed fighter still listed) as of 2026-08-05 — do not treat as current without spot-checking against a known-correct value first. |
| **The Odds API** | Live + historical MMA moneylines | Unchanged. |

---

## 3. Updated: Week 1 day-by-day tasks

### Tuesday (was: "Your own ufcstats scraper")

**Original text:** *"Your own ufcstats scraper: events index → event page → bout page →
fighter page. Raw HTML cached to `/data/raw/`. Rate-limited, resumable."*

**Replace with:**

> Data ingestion pipeline against Greco1899's six CSVs (`ufc_events.csv`, `ufc_fight_results.csv`,
> `ufc_fight_stats.csv`, `ufc_fighter_details.csv`, `ufc_fighter_tott.csv`; `ufc_fight_details.csv`
> confirmed redundant, not used). Built in `data/ingestion/`:
> - `parsers.py` — per-file read/clean functions (landed/attempted string parsing, height/reach/
>   control-time unit conversion, weight-class and title-fight detection, historical-format
>   filtering)
> - `transform.py` — resolves free-text fighter/event names to real FKs, builds `fighters`/
>   `events`/`bouts`/`bout_stats` DataFrames with explicit IDs, inner-joins `bout_stats` to the
>   filtered `bouts` table
> - `loaders.py` — Postgres insert, still to be built (Wednesday)
>
> `data/scraping/fetch.py` (cache-first, rate-limited, retrying HTTP layer) was built but is
> currently unused against ufcstats.com — it's reusable as-is for Friday's Wikipedia/ESPN
> upcoming-events work, since neither of those sources exhibits the ufcstats.com gate.

### Wednesday (was: "Parsers + loaders... fuzzy fighter-name resolution")

**Scope narrows.** Original text assumed name-matching ambiguity *within* the scraped dataset.
That's mostly not the case: Greco's fighter-bio files share a stable `source_url` per fighter,
so internal identity is resolved deterministically — **except for 8 known real name collisions**
(16 fighter rows out of 4,577; e.g. two different UFC fighters both named "Bruno Silva"),
which are logged to `data/ingestion/logs/` rather than guessed, for manual resolution.

**`rapidfuzz` + `fighter_aliases` are still needed, but now specifically for cross-source
matching** — joining Greco-sourced fighters against The Odds API (Thursday) and
Wikipedia/ESPN (Friday), none of which share Greco's URL scheme.

Remaining Wednesday work: finish `loaders.py` (Postgres insert in FK-safe order, using
explicit IDs — see note below), resolve the 8 name collisions, `run_ingest.py` orchestration.

> **Note on explicit IDs:** `transform.py` assigns `fighter_id`/`event_id`/`bout_id` explicitly
> in pandas before any DB interaction, rather than inserting and reading autogenerated IDs back.
> `loaders.py` needs to insert these explicit values and reset each table's sequence afterward
> (`setval`) so future inserts don't collide with them.

### Friday (was: "Upcoming-events scraper (ESPN + ufcstats + Wikipedia fallback)")

**Replace with:**

>> Upcoming-events and rankings client: **Wikipedia primary**, accessed via the **MediaWiki
> Action API** rather than HTML scraping (see ADR-009). **ESPN fallback only** (verify
> against a known-current value before trusting, per §2). ufcstats.com excluded entirely.
> Extends `fetch.py` from Tuesday — added query-param support, a `use_cache` bypass for
> lookups that must always be live, and a Wikimedia-compliant `User-Agent`. New module
> `data/scraping/wiki_api.py` wraps the API calls; new dependency `mwparserfromhell` parses
> the `{{MMAevent bout}}`/`{{MMAevent card}}` templates on event pages and the wikitable on
> the list page. Upcoming events/bouts keyed by Wikipedia `pageid`, not `source_url`.

### Saturday (data-quality suite) — additions

In addition to the original checks, verify:
- The pre-Unified-Rules-era filter still drops ~215 bouts (±small drift as Greco's data updates)
  — a large swing signals something changed upstream, worth investigating before trusting the load.
- No orphaned `bout_stats` rows (i.e., the inner-join in `transform.py` is doing its job).
- The known-8 name-collision list hasn't grown — new collisions should be logged and reviewed,
  not silently absorbed by the exact-match path.

### Week 1 exit criterion — status check

Original: *"≥ 8,000 bouts with per-round stats."* Current count after filtering historical
formats: **8,595 bouts** (from 8,810, after dropping 215 pre-Unified-Rules-era rows). This still
clears the 8,000 threshold, but Wednesday's name-collision exclusions will subtract a small
additional number — worth confirming the final count stays above 8,000 once `transform.py` runs
end-to-end.

Original:  ≥ 4,000 bouts with odds attached.
Updated: Bouts with odds attached: ~2,800-2,900 (revised down from ≥4,000 — The Odds API's historical coverage begins June 2020, capping the addressable population at 3,209 bouts regardless of match rate). Coverage is concentrated in the 2023+ validation/test window, where it's actually used.

---

## 4. Updated: Week 4 Friday — GitHub Actions automation

**This section needs the most substantial rework** and is flagged here at a high level;
revisit in full when Week 4 actually arrives, since implementation details may shift further
between now and then.

**What changes:** the original "event-driven results scrape (~3hrs post-event, ~12hr fallback
retry)" was designed around triggering a live scrape of ufcstats.com tied to actual fight-card
timing. That no longer applies — there's no live source to trigger against. Greco1899 refreshes
via their own daily automated job, decoupled from your event timing.

**Replacement approach (subject to revision in Week 4):**
- Poll Greco1899's repo on a regular cadence (e.g., daily) rather than an event-driven trigger,
  checking for new/updated CSV content since the last successful ingest.
- Add a **staleness check**: if a known UFC event (from the Wikipedia-sourced schedule) has
  passed and Greco's data hasn't reflected it within a defined threshold, surface a warning
  rather than silently settling nothing. This directly addresses the real gap observed
  May 21–Jul 19, 2026 (6 events, later backfilled) — see ADR-006 and §5 below.
- Re-running the full `parsers.py → transform.py → loaders.py` pipeline should be safe and
  idempotent by construction (via `source_url` uniqueness), so "new data arrived" can simply
  mean "rerun the pipeline," not a specially-built incremental path.
- Upcoming-events + rankings refresh: same fight-week-aware cadence reasoning as originally
  planned (every 3 days normally, every 12 hours during fight week) — just pointed at the
  Wikipedia API (ADR-009) / ESPN instead of ufcstats.com/ESPN/Wikipedia. Each refresh call
  passes `use_cache=False` since these lookups need to reflect live state.
- Odds refresh (6x/day, Odds API): unaffected by any of this.

**Status update (2026-08-09):** the Wikipedia-sourced upcoming-events pipeline itself —
`wiki_api.py`, `wiki_parsers.py`, `upcoming_events_loader.py`, `ingest_upcoming_events.py`
(ADR-009/010/011) — is now built and idempotency-tested. The bullets above still describe
future GitHub Actions cron/event-trigger work, not the ingestion logic those triggers will
call; that logic already exists.

**Greco↔Wikipedia duplicate-event reconciliation:** ADR-011 flagged the merge step between
a Wikipedia-sourced event row and Greco's later-confirmed row as an unbuilt bridge. As of
this update, that gap has **automated detection** — `check_duplicate_events` in
`data/ingestion/quality_checks.py` fuzzy-matches event name + date across the two sources
and flags likely pairs — but **not automated reconciliation**. Merging a flagged pair (or
deciding how to) is still a manual step. The actual bridge remains a prerequisite before the
first of the currently-tracked upcoming events completes and Greco ingests a real result for
it.

---

## 5. Updated: Risk register

**Replace** the original "ufcstats HTML changes mid-project" entry with:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ufcstats.com blocks all non-browser access (proof-of-work gate) | **Realized**, not hypothetical | High — eliminated a planned primary data path | Greco1899 CSVs adopted as sole source (ADR-006). No workaround attempted — see ADR-006 for reasoning. |

**Add:**

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sole reliance on Greco1899 as a third-party pipeline; update cadence not guaranteed to match live event timing | Medium (one ~2-month gap observed: May 21–Jul 19, 2026, 6 events, later backfilled) | Medium — could delay settlement/track-record accuracy | Staleness detection in Week 4 settlement job (§4); idempotent, safely re-runnable loader |
| ESPN public rankings API returns stale data | Confirmed (outdated fighter observed in rankings, 2026-08-05) | Low–Medium, contained to fallback-only usage | Wikipedia as primary; ESPN used only with manual verification against a known-correct value |
| Greco↔Wikipedia duplicate `events` rows once a Wikipedia-sourced event completes and Greco ingests its real result (ADR-011's known follow-up) | Medium — will occur for every currently-tracked upcoming event unless resolved first | Medium — duplicate event/bout rows; risk of orphaned `predictions` rows if not caught | Automated detection via `check_duplicate_events` (`data/ingestion/quality_checks.py`); reconciliation itself (merging the two rows) is still manual — see §4 |

---

## 6. Schema changelog (for reference — already applied, no action needed)

Five migrations applied since the original schema design:

1. `38e3db2c80a5` — initial schema
2. `e14a345445b9` — added `source_url` (unique, nullable) to `fighters` and `bouts`, enabling
   idempotent reloads keyed on Greco's own URLs rather than requiring name-matching on every run
3. `e849068ec219` — added `reversals` to `bout_stats` (present in source data, initially dropped,
   reinstated as a kept stat)
4. added `wikipedia_pageid` (`BigInteger`, nullable, unique) to `events` — the stable key for
   Wikipedia-sourced upcoming events, kept separate from `source_url` since it applies before
   Greco has any data for the event and survives page-title renames (ADR-011)
5. added `rounds_confirmed` (`Boolean`, `NOT NULL`, default `false`) to `bouts` — guards a
   manually-corrected `scheduled_rounds`/`is_title_fight` from being silently reverted on the
   next Wikipedia ingest rerun (ADR-010)

**Open item, not yet acted on:** if a `CHECK` constraint on `fighters.stance` is added in the
future, it must include `'open stance'` and `'sideways'` in addition to `'orthodox'`/
`'southpaw'`/`'switch'` — confirmed via real data (`ufc_fighter_tott.csv`), not yet enforced at
the DB level since no such constraint currently exists.

---

## 7. New: dataset scope note

215 of 8,810 bouts (~2.4%) — pre-Unified-Rules-era UFC events using 1/2-round or no-time-limit
formats — are **permanently excluded** at ingestion time, not held out as part of the Week 2
temporal train/test split. This is a deliberate data-scope boundary, distinct from and prior to
any modeling-related splitting. All excluded bouts predate 2022 by a wide margin, so this has no
interaction with the temporal split's train (≤2022) / validation (2023–24) / test (2025+)
boundaries.

---

## 8. Cross-references

- Full reasoning for the ufcstats.com decision: **ADR-006**, `docs/DECISIONS.md`
- This addendum should be merged into `docs/PLAN.md` §1 (data sources), the Week 1 day-by-day
  table, Week 4 Friday's task list, and §5 (risk register) at your convenience — flagged here
  as an addendum rather than an in-place edit so you can review before it overwrites anything.
- Full reasoning for the upcoming-events API choice: **ADR-009**, `docs/DECISIONS.md`
- Full reasoning for reusing `events`/`bouts` with `status='scheduled'` instead of a staging
  table: **ADR-010**, `docs/DECISIONS.md`
- Full reasoning for `wikipedia_pageid` as a dedicated identity column: **ADR-011**,
  `docs/DECISIONS.md`