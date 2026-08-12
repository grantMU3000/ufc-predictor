# Decisions Log (ADR)

This file records significant technical and product decisions for the UFC Predictor project — what was chosen, what alternatives were considered, and why. It is not a changelog. It's a record of reasoning at the moment a decision was made, so it doesn't need to be reconstructed later (in a PR review, a debugging session, or an interview).

**When to add an entry:** any time you choose between two real alternatives and the choice isn't obvious in hindsight — a library/tool choice, a schema design, a scope cut, a modeling approach, a "we tried X and reverted" moment. Skip it for trivial stuff (variable names, formatting).

**Format:** copy the template below, fill it in, add it to the top of the log (newest first), and never edit an old entry's reasoning after the fact — if a decision is reversed, add a *new* entry that supersedes it and link back.

---

## Template

```
## [ADR-NNN] Short title of the decision

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-XXX

### Context
What problem or question forced this decision? What constraints were in play (budget, time, skill level, prior decisions)?

### Options considered
1. **Option A** — pros / cons
2. **Option B** — pros / cons
3. **Option C** — pros / cons

### Decision
Which option was chosen.

### Why
The actual reasoning. Be specific about tradeoffs accepted, not just benefits gained.

### Consequences
What this makes easier, what this makes harder, what it forecloses or defers.
```
---
## [ADR-012] Reuse `rounds_confirmed` to suppress quality-check false positives on legitimate 3-round title fights

**Date:** 2026-08-13
**Status:** Accepted

### Context
The `title_fights_have_five_rounds` quality check (`data/ingestion/quality_checks.py`)
flagged ~70 bouts where `is_title_fight = true` but `scheduled_rounds = 3`. Investigation
showed these are not data errors:

- The overwhelming majority are tournament-finale "Title Bout"s from *The Ultimate Fighter*
  and *Road to UFC* franchises — these crown a tournament winner and have always been
  scheduled for 3 rounds, unlike a standard 5-round UFC title bout. `is_title_fight` is
  correctly `true` (real_name string contains "Title Bout"/"Championship"), but the
  5-round assumption baked into the check doesn't hold for this bout type.
- One additional case (`bouts.id=8559`, event_id=640, "UFC Welterweight Title Bout") is a
  legitimate standard UFC title bout from an older event, predating the modern 5-round
  title-bout convention — a real historical exception, not a parsing bug.

All ~70 rows were individually confirmed as legitimate before this decision was made, not
bulk-suppressed on assumption.

### Options considered
1. **Leave the check failing, treat it as expected noise** — simplest, but a check that's
   expected to always show ~70 failures stops being useful as a signal; a *new*, real
   regression (e.g. a genuine 3/5-round data bug introduced later) would be lost in the
   same noise.
2. **Change the check's query to exclude bouts by event-name pattern** (e.g. name contains
   "Ultimate Fighter" or "Road to UFC") — mirrors ADR-007's existing pattern for excluding
   Road to UFC rows, but would need to also special-case the one non-tournament historical
   outlier separately, and doesn't leave a durable, per-row record of *which* bouts were
   actually reviewed and confirmed correct.
3. **Reuse the existing `rounds_confirmed` column**: set it `true` on exactly these ~70
   confirmed rows, and add `AND NOT rounds_confirmed` to the check's `WHERE` clause.

### Decision
Option 3. `rounds_confirmed` was set `true` on the ~70 confirmed-legitimate 3-round title
bouts, and `check_title_fights_have_five_rounds` was updated to exclude
`rounds_confirmed = true` rows.

### Why
`rounds_confirmed` already exists on `bouts` and already means, informally, "a human has
looked at this row's round count and it's correct" — reusing it avoids adding a second
column with near-identical meaning, and leaves a durable, queryable record of exactly which
rows were reviewed (versus a name-pattern filter, which encodes "why we think this is fine"
in the query rather than on the data itself).

**Important distinction from this column's original purpose (ADR-010):** on the Wikipedia
ingestion path, `upcoming_events_loader.py`'s `load_bout()` uses `rounds_confirmed` as an
active guard — a literal `CASE WHEN rounds_confirmed THEN scheduled_rounds ELSE :rounds END`
in the UPDATE statement, stopping a human correction from being silently reverted on rerun.
**The Greco loader (`loaders.py`) has no such guard.** `rounds_confirmed` is not in its
`insert_cols` list for `bouts`, so a Greco reload will never flip the flag back to `false`
(the flag itself is safe) — but `is_title_fight` and `scheduled_rounds` *are* in
`insert_cols`, meaning Greco unconditionally re-derives and rewrites both on every reload,
with no check of `rounds_confirmed` at all. In practice this doesn't threaten these specific
rows, since Greco's parser deterministically re-derives "3 rounds" from the same source CSV
text every time — but that's a property of the source data being stable, not protection
this column is actually providing on the Greco side. Here, `rounds_confirmed` is doing a
narrower job than its original one: marking "human-verified, suppress the quality check,"
not "protected from being overwritten."

### Consequences
**Easier:** `title_fights_have_five_rounds` goes back to being a clean signal — a future
failure means a genuinely new, unreviewed round-count mismatch, not one of these ~70 known
cases.

**Requires care going forward:** any *new* legitimate 3-round title bout (e.g. a future TUF
or Road to UFC finale ingested by Greco) will need `rounds_confirmed` set manually the same
way, or it will correctly reappear in the check's output — this is expected, not a bug.

**Worth remembering:** `rounds_confirmed = true` means two different things depending on
which loader touches the row — "protected from being overwritten" (Wikipedia path) vs.
"human-verified, quality-check suppressed, but not actually write-protected" (Greco path).
Anyone editing `loaders.py`'s `insert_cols` or this quality check in the future should be
aware the column isn't uniformly enforced across both ingestion paths.

---
## [ADR-011] Track Wikipedia identity via a dedicated `wikipedia_pageid` column, separate from `source_url`

**Date:** 2026-08-09
**Status:** Accepted

### Context
Upcoming events are ingested from Wikipedia before Greco/ufcstats has any data for them — `events.source_url` starts NULL and gets filled in later, once Greco's daily job picks up the completed event. Wikipedia article titles for upcoming events are not stable: they get renamed as fights are confirmed, reshuffled, or drop out (e.g. a Fight Night page titled after a headliner who is later replaced). Something had to serve as the durable key for upserting these rows across reruns.

### Options considered
1. **Store the Wikipedia page title/URL in `source_url`, same column Greco writes to** — reuses existing infrastructure, but conflates two different identity schemes in one column (a ufcstats URL that never changes vs. a Wikipedia title that can), and breaks the moment a page gets renamed between scrapes.
2. **Add a new `wikipedia_pageid` column, populated from Wikipedia's own stable numeric page ID, kept independent of `source_url`** — small schema cost (one migration), but each column keeps one clear meaning for its whole lifetime, and `pageid` is immune to title renames by construction.

### Decision
Option 2 — added `wikipedia_pageid` (nullable, unique) to `events`. `source_url` keeps its existing meaning (a ufcstats.com URL, set once Greco ingests the event) and starts NULL for Wikipedia-originated rows; `wikipedia_pageid` is set immediately and never cleared, even after `source_url` is eventually populated.

### Why
Overloading one column with two identity schemes that apply at different times was the same mistake already avoided once with the fighter `source_url` convention. Two columns, two fixed meanings, both nullable independently, means the Wikipedia-sourced ingestion path and the Greco ingestion path can both write to the same `events` row without ever needing to agree on what "the identity" of that row is.

### Consequences
Upcoming-events upserts key on `wikipedia_pageid`, resolved via the MediaWiki API's `action=query` metadata lookup (which follows redirects) rather than trusting whatever title a link happened to show at scrape time. Requires a follow-up reconciliation step in the Greco loader path: when Greco later ingests a real result for an event that already has a Wikipedia-sourced row (`source_url IS NULL`, `wikipedia_pageid` set), the loader must match on `event_date` (+ fuzzy name as a second check) and update that row in place rather than inserting a duplicate — this bridge logic doesn't exist yet as of this session and is a known follow-up, not yet built.

---
## [ADR-010] Store upcoming events/bouts in the existing `events`/`bouts` tables with `status='scheduled'`, not a separate staging table

**Date:** 2026-08-09
**Status:** Accepted

### Context
Upcoming-events data (from Wikipedia, per ADR-009) needs a landing spot in the schema. `predictions.bout_id` is a foreign key directly into `bouts` — the project's core goal of logging a prediction before a fight happens requires that whatever table holds "an upcoming fight" is the same table a prediction can point at from day one, without a later migration/promotion step.

### Options considered
1. **A separate staging table for unresolved/upcoming bouts, promoted into `bouts` once confirmed** — cleanly isolates churny, pre-fight data from the stable historical table, but requires a "promote from staging" step before any prediction could ever be logged against it, adding a reconciliation job for something that should be a plain insert.
2. **Reuse `bouts`/`events` directly, with `status='scheduled'` and null result columns** — `bouts.status` was already designed to support this (`scheduled`/`completed`/`cancelled`, from the original schema). A prediction logged against a `scheduled` bout keeps the same `bout_id` for its entire lifecycle, through completion.

### Decision
Option 2. Upcoming events insert into `events` (keyed on `wikipedia_pageid`, see ADR-011). Upcoming bouts insert into `bouts` with `status='scheduled'` and all result columns NULL, resolved through the same `fighter_red_id`/`fighter_blue_id` FKs as historical bouts.

### Why
The staging-table approach solves a data-hygiene concern (keep churny data separate) at the cost of breaking the one hard constraint that actually matters — predictions must be able to reference a bout before it happens, permanently, without ever needing to know if that bout was "promoted." The existing schema had already anticipated this exact case via the `status` column; no schema redesign was needed, only three additive behaviors layered on top (below).

### Consequences
Three new mechanisms were required to make bout-level data reliable under this reuse, since Wikipedia doesn't hand any of them over directly:

- **Stub fighters:** a Wikipedia fighter name with no match in `fighters` (via alias → exact → blocked fuzzy match, WRatio@90) gets a minimal stub row (`real_name` only, `source_url` left NULL — an unverified ufcstats guess was judged worse than no link, since a wrong guess would silently misattribute a future Greco fighter's real stats to the wrong row). Verified links are added manually later via the same override-dict process used for Greco's collisions, never inferred automatically.
- **Cancel-and-reinsert for fighter swaps:** since there's no stable natural key for an upcoming bout the way `source_url` serves historical ones, a fighter replacement is detected as "an active bout on this event involving one of the two fighters, but a different pairing" — the stale row is marked `status='cancelled'` (not deleted, preserving what was originally booked) and a new row inserted for the current pairing.
- **Inferred fields, explicitly flagged as inferred:** `is_title_fight` (from champion markers/notes text) and `scheduled_rounds` (5 for title fights and main events, 3 otherwise) aren't provided by the Wikipedia template and are derived. A `rounds_confirmed` boolean column was added after a manual correction was found to silently revert on the next ingest run — the inference now only overwrites `scheduled_rounds`/`is_title_fight` when that flag is false, so a human correction persists across reruns.

**Deliberately out of scope for this session, logged as known gaps rather than fixed:** events whose fight card is still in an "Announced bouts" prose section (pre-table stage) get their `events` row but zero parsed bouts — parsing prose-announced matchups was judged a meaningfully messier problem than the clean `{{MMAevent bout}}` template case and left for later. Wikipedia's own live-edited result fields (method/round/time, if a fast editor fills them in before Greco's daily job runs) are never trusted — `status` stays `'scheduled'` until Greco provides a confirmed result through the normal ingestion path, preserving the prediction ledger's single source of truth.

---
## [ADR-009] Switch upcoming-events data collection from HTML scraping to Wikipedia's API

**Date:** 2026-08-08
**Status:** Accepted

### Context
Per `PLAN_ADDENDUM.md` §3 (Friday), the upcoming-events/rankings task was planned as HTML
scraping against Wikipedia, reusing `fetch.py` unmodified. Inspecting the actual wikitext via
the MediaWiki API showed event fight-card data is built from a `{{MMAevent bout}}` template —
a fixed positional structure — rather than a plain wikitable, and the API additionally exposes
a stable `pageid` that survives page renames (a real risk for upcoming events, which get
retitled as cards are confirmed/reshuffled).

### Options considered
1. **HTML scraping** — original plan, reusing `fetch.py` unmodified. Works, but requires
   parsing rendered HTML (BeautifulSoup) and offers no built-in way to get a page's stable ID.
2. **MediaWiki Action API** (`action=parse`, `prop=wikitext`/`sections`) — returns raw wikitext;
   combined with `mwparserfromhell`, the `{{MMAevent bout}}` template parses into clean
   positional fields, and `pageid` comes back in the same response, for free.

### Decision
Use the MediaWiki Action API for both the "List of UFC events" schedule and individual event
pages, instead of scraping rendered HTML.

### Why
Same cost (free, no auth) as scraping, but more reliable given the actual template-based page
structure, and it solves the title-stability problem for upcoming events without extra work —
`pageid` becomes the natural key instead of `source_url`.

### Consequences
`fetch.py` extended, not reused unmodified — added query-param support, a `use_cache` bypass
(upcoming-event lookups must be live, not served stale from cache), and a Wikimedia-compliant
`User-Agent`. New dependency: `mwparserfromhell`. Upcoming events/bouts will be keyed by
Wikipedia `pageid` rather than `source_url` — a deliberate departure from the convention used
for Greco/Odds data, worth noting when the upcoming-events table is built.

---
## [ADR-008] Revise odds-coverage exit criterion — The Odds API's June 2020 floor

**Date:** 2026-08-07
**Status:** Resolved

### Context
The original exit criterion (`docs/PLAN.md` §6) targets "≥4,000 bouts with odds attached."
The Odds API's historical MMA data begins June 2020 — a hard coverage floor on any paid tier,
not a budget constraint. Querying the loaded dataset directly (Postgres, not the raw CSVs)
confirmed 3,209 bouts fall on or after June 2020, out of 8,593 total loaded bouts. This is the
maximum possible population that could ever have real odds attached, before any name-matching
loss is applied (the plan already budgets ~10% match failure on top of this).

### Options considered
1. **Keep the ≥4,000 target and treat it as at-risk** — no change to the stated goal, revisit
   only if the shortfall becomes a problem later. Rejected: the ceiling is structural (a fixed
   API limitation), not a matching-quality or effort problem, so "trying harder" cannot close
   this gap. Leaving the number as-is risks an unexplained shortfall in the final README.
2. **Pay for a different/additional odds provider with deeper historical coverage** — could
   theoretically raise the ceiling. Rejected for now: adds cost and integration complexity
   disproportionate to the benefit, given odds are explicitly excluded from model features
   (ADR-002) and used only for baseline/backtest/calibration purposes.
3. **Revise the exit criterion to reflect the real achievable population, reframed around
   where odds are actually used** — the criterion becomes honest and defensible rather than
   quietly missed.

### Decision
Option 3 — revise the exit criterion to ~2,800–2,900 bouts with odds attached (accounting for
expected match-failure loss on top of the 3,209 ceiling), and explicitly note that coverage
concentrates in the 2023+ validation/test window, where odds are actually consumed
(baseline comparison, ROI backtest, calibration — never as a model feature). 

### Why
Odds are evaluation-stage tooling, not training input (ADR-002), and evaluation only touches
the validation (2023–24) and test (2025+) windows. Year-by-year bout counts confirm those
years alone (~1,871 bouts) already provide a strong sample for the metrics that actually get
reported. The pre-2023 portion of the 3,209-bout ceiling is a bonus for context, not a
requirement — so the original 4,000 target was measuring the wrong thing: total odds-tagged
rows, rather than odds coverage where it's actually consumed. Revising the number is more
honest than either quietly missing it or spending disproportionately to chase it.

### Consequences
**Easier:** The odds-ingestion scope stays bounded to a single, affordable API tier and a
single historical backfill pass — no need to justify a second data source or an expanded
budget purely to hit an arbitrary total.

**Forecloses:** Bouts before June 2020 (5,384 of 8,593 loaded bouts) will permanently have no
real odds data from this source. The favorite-baseline and ROI backtest, by design, only ever
needed the post-2023 window anyway, so this forecloses nothing that was actually load-bearing.

**Requires a follow-up edit:** `docs/PLAN_ADDENDUM.md` §6 (exit checklist) needs its odds
line item updated to reference this ADR and the revised number, once that section is written.

---
## [ADR-007] Exclude "Road to UFC" rows from bout dataset

**Date:** 2026-08-06
**Status:** Accepted

### Context
2 rows in `ufc_fight_results.csv` belong to "UFC - Road to UFC 4.6," a contract/prospect
elimination show, not an event involving fighters already on the UFC roster. These rows also
had no matching entry in `ufc_events.csv`, surfacing as unresolved `event_id` rows during
ingestion.

### Options considered
1. **Keep the rows, add the missing event to `events`** — technically possible, but the
   fighters/bouts don't represent roster-level UFC competition, which is what this dataset
   models.
2. **Exclude "Road to UFC" rows at ingestion**, matched by event-name pattern — deliberate,
   documented scope boundary rather than an incidental drop.
3. **Leave exclusion implicit** (rely on the existing unresolved-name/event log) — works
   today, but conflates a deliberate scope decision with genuine data-quality bugs, risking
   confusion if more such rows appear in future Greco refreshes.

### Decision
Option 2 — explicit filter in `read_fight_results()`, matching on `event_name` containing
"Road to UFC," logged with a row count on every run.

### Why
Only 2 rows / 1 event affected today, but the exclusion is about what the dataset should
represent (roster-level UFC bouts), not about these specific rows — an explicit, named filter
makes that reasoning visible and durable if similar shows (e.g. Contender Series) appear later,
rather than relying on an incidental side effect of FK resolution.

### Consequences
**Easier:** `unresolved_bout_fighters.csv` stays reserved for genuine data-quality issues,
not conflated with intentional scope exclusions. Future Road to UFC rows are caught and
logged automatically rather than silently piling into the unresolved log.

**Forecloses:** Prospect/contract-show fights (Road to UFC, and similar if found later) are
permanently out of scope for this dataset, independent of any modeling-stage decisions.

---
## [ADR-006] Primary data source: Greco1899 CSVs over a self-built ufcstats.com scraper

**Date:** 2026-08-05
**Status:** Accepted

### Context
Week 1 Tuesday's original task (per the execution plan) was to build a first-party
scraper against ufcstats.com: events index → event page → bout page → fighter
page, with raw HTML cached locally, rate-limited and resumable. `Greco1899/scrape_ufc_stats`
CSVs were meant to be a Day-1 bootstrap dataset, scraped in parallel with — not
instead of — an owned scraper, specifically so the project wouldn't be solely
dependent on a third party's pipeline.

While building `fetch.py` (cache-first, rate-limited, retry-with-backoff), test
requests against `ufcstats.com/statistics/events/completed` consistently returned
a JavaScript proof-of-work challenge page (SHA-256-based, similar to Anubis-style
bot gates) instead of the real page content. This was diagnosed methodically:
- The page loads normally in a real browser.
- A bare `curl` request with no custom headers, from the same IP as the browser,
  still receives the challenge page.
- A `curl` request spoofing a full Chrome `User-Agent` string, same IP, still
  receives the challenge page.

This rules out IP reputation and User-Agent heuristics as the cause. The gate is
unconditional for any client that doesn't execute JavaScript — i.e., any client
that doesn't run and answer the proof-of-work challenge, regardless of how
"normal" the rest of the request looks.

### Options considered
1. **Build a JS-challenge solver and continue with a first-party scraper** —
   would technically unblock data collection and preserve the original Week 1
   plan as written. Rejected outright: replicating the proof-of-work check and
   submitting the answer is circumventing an access control the site has
   deliberately and actively put in place for non-browser clients. This isn't
   ambiguous like a missing `robots.txt` — it's an enforced technical barrier,
   which is a clear signal of intent from the site owner. Not something to
   engineer around regardless of the reasonableness of the end goal.
2. **Headless-browser automation (e.g. Playwright/Selenium) to execute the
   challenge like a real browser would** — would likely work technically, but
   is functionally the same as option 1: deliberately automating past a check
   whose entire purpose is to distinguish browsers from automated clients.
   Rejected for the same reason.
3. **Rely on Greco1899 CSVs as the primary and, for now, sole ingestion
   source** — pre-scraped, regularly updated by a third party, already cloned
   and loaded into DuckDB from Week 0. Con: introduces a hard dependency on
   someone else's pipeline, including its reliability and update cadence,
   which the original plan explicitly tried to avoid by building an owned
   scraper in parallel.
4. **ESPN's public API as a supplementary source** — already preferred in the
   plan for rankings/schedules due to being less bot-protected than UFC.com.
   Doesn't cover the full historical bout-level stats Greco provides, so
   viable as a supplement, not a replacement.

### Decision
Use Greco1899 CSVs as the primary and current sole source for historical
events/bouts/fighter-stats ingestion. Do not build or attempt to bypass
ufcstats.com's bot-detection gate. ESPN's public API remains in use as a
supplementary source for schedules/rankings, per prior decisions. A first-party
ufcstats.com scraper is off the table unless the site's posture changes.

### Why
The diagnostic process (browser vs. bare curl vs. spoofed-UA curl, same IP for
all three) isolated the cause precisely: this is not an IP flag, not a header
heuristic, not intermittent load-based gating — it's a deliberate, consistently
enforced barrier against non-browser clients. That's a materially different
situation from the "ufcstats HTML changes mid-project" risk the original risk
register anticipated (parser breakage from a markup change), which is why this
decision supersedes that mitigation rather than simply exercising it.

The tradeoff accepted here is real: this project no longer owns its primary
data-acquisition layer end-to-end, which was one of the explicit learning and
resilience goals of building a first-party scraper. What's kept is everything
downstream of acquisition — parsing, fuzzy fighter-name resolution, the
`fighter_aliases` table, loading into the Postgres schema, idempotent/resumable
ingestion — all of which is unaffected by this change and remains the owned,
defensible engineering surface for this part of the project.

Verified before accepting this dependency: Greco1899 has updated within a day
of nearly every UFC event over the past year, with one notable gap (May 21 –
July 19, 2026, spanning 6 events, later backfilled). All data currently appears
up to date as of this decision.

### Consequences
**Easier:** Today's ingestion work simplifies from "fetch + parse HTML" to
"parse CSV," letting Tuesday's remaining time go toward the loader, fuzzy name
matching, and schema population rather than a blocked network layer. Removes
an entire class of scraping-etiquette and access-control concerns.

**Harder / deferred:** Week 4's event-driven results/odds automation (originally
designed around hitting ufcstats.com directly, ~3hrs post-event with a ~12hr
fallback retry) now depends on Greco's own update cadence instead, which is not
guaranteed to match that timeline — the May–July gap is proof this can happen.
Mitigations planned for Week 4: (1) the settlement job checks data freshness
against the known UFC event schedule and surfaces a warning rather than
silently settling against stale data; (2) the loader is built idempotent and
safely re-runnable from the start, so a delayed Greco update can be picked up
later without manual cleanup.

**Foreclosed, for now:** A first-party historical scraper against ufcstats.com.
Revisit only if the site's access posture changes, or if Greco1899 stops being
reliably maintained.

**`fetch.py`'s caching/rate-limiting/retry infrastructure is not wasted** — it
remains the right tool for Friday's upcoming-events scraper against ESPN and
Wikipedia, neither of which exhibited this gating behavior.

---
## [ADR-005] Bouts table design — corner FKs, unified status, winner tracking

**Date:** 2026-08-04
**Status:** Accepted

### Context
A bout involves exactly two fighters (no more, no less), and needs to represent
both upcoming (unresolved) and completed (resolved) fights in a single queryable
shape, since `predictions` must be able to reference a bout before it happens.

### Options considered
- **Junction table (`bout_fighters`)** — rejected. Cardinality is fixed at
  exactly 2, so a junction table adds a join with no benefit and doesn't
  enforce "exactly two" without additional constraints anyway.
- **Separate `bout_results` table** — rejected for now. Splitting outcome
  data out would mean joining on every query that needs both matchup and
  result, for a table that's never actually queried independently. Revisit
  if the outcome columns grow substantially.

### Decision
- Model the two participants as two dedicated FK columns on `bouts`
  (`fighter_red_id`, `fighter_blue_id`) rather than a fighter↔bout junction table.
- Track the outcome (`winner_id`, `method`, `method_detail`, `ending_round`,
  `ending_time_seconds`) as nullable columns on the same row, gated by a
  `status` enum-like text field (`scheduled` / `completed` / `cancelled`).
- Add two same-row CHECK constraints:
  - `fighter_red_id <> fighter_blue_id`
  - `winner_id IS NULL OR winner_id IN (fighter_red_id, fighter_blue_id)`

### Consequences
- Querying "who fought" requires two joins to `fighters` (aliased as
  `f_red`/`f_blue`), not one — acceptable, this is a read-time cost only.
- `winner_id` referencing a non-participant is now impossible at the DB
  level; this class of bug can't reach production data.
- Cross-table validation (e.g. `predictions.predicted_winner_id` matching
  the bout's actual participants) is explicitly out of scope for schema
  constraints and is deferred to the application/ingestion layer.

---

## Log

## [ADR-004] Keep red/blue corner explicit in schema and features, rather than anonymous fighter_a/fighter_b

**Date:** 2026-08-04
**Status:** Accepted

### Context
ufcstats.com (and most UFC data sources) always lists bouts as a red-corner fighter vs. a blue-corner fighter. Corner assignment isn't random: the red corner is conventionally given to the higher-profile or higher-ranked fighter (defending champion, betting favorite, promotional priority), so red-corner fighters win more often than blue-corner fighters in the raw historical record. `docs/PLAN.md` §0.2 flags this directly as "corner/ordering leakage" — a model trained on `(fighter_A, fighter_B)` in dataset order can learn to just predict whichever slot wins more often, independent of who the fighters actually are, and look artificially accurate while having learned nothing about fight outcomes.

### Options considered
1. **Anonymize corners into generic `fighter_a`/`fighter_b` columns at ingestion, discarding which was red/blue** — removes the temptation to leak on corner identity by construction, but throws away real information (corner assignment does correlate with promotional confidence in a fighter) and makes it harder to reconstruct the original bout record for debugging or display.
2. **Keep `fighter_red_id` / `fighter_blue_id` explicit in the schema (as built — see `fighter_red_id`/`fighter_blue_id` in the `bouts` table and `predicted_prob_red` in `predictions`), but defend against the leak downstream at the feature/training layer** — via symmetrization (emit both orderings with flipped labels) or purely differential features (`reach_diff`, `slpm_diff`, computed as red-minus-blue) so the model can't key off corner position alone, combined with measuring and logging the raw red-corner win rate as a known baseline to test against.

### Decision
Keep red/blue corner explicit everywhere in the schema and raw data (`fighter_red_id`, `fighter_blue_id`, `predicted_prob_red`), and handle the bias at the feature-engineering and training layer instead of by anonymizing the source data.

### Why
Anonymizing at ingestion would hide the bias rather than defend against it — you can't audit a leak you've already erased the evidence of. Keeping corner explicit means the red-corner win rate is a number that can be computed, written down, and tracked as a known bias to test the pipeline against: if a trained model's implied "red advantage" ever approaches that raw win rate, that's a signal the symmetrization or differential-feature defense has failed and the model is leaking corner position instead of learning fighter-specific signal. This makes the leak testable (`LEAKAGE_LOG.md`) instead of just assumed away, at the cost of requiring discipline downstream — every training-matrix build must symmetrize or difference the features rather than feeding `fighter_red_id`/`fighter_blue_id` (or red/blue-keyed raw stats) into the model directly.

### Consequences
The training pipeline must never use raw red/blue-keyed features as model inputs directly — only symmetrized (dual-row, flipped-label) or differential (red-minus-blue) versions. This needs a unit test / leakage check (Week 2 audit day) that specifically verifies the model's accuracy doesn't collapse toward the raw red-corner win rate when corner is the only signal available. It also keeps the raw schema faithful to the source data, which matters for debugging, display (the frontend needs to know who was actually in which corner for a given historical bout), and for computing the red-corner win rate as a standing baseline number rather than a one-time throwaway stat.

---

## [ADR-003] Exclude draws and no-contests from training data

**Date:** 2026-08-04
**Status:** Accepted

### Context
Bout outcomes in the raw data aren't purely binary: alongside win/loss, a small number of bouts end in a draw (judges' scorecards split with no majority winner) or a no-contest (result overturned, e.g. for a doping failure or accidental foul). The model as planned is a binary win-probability classifier — it needs a decision on how these outcomes are represented in the label.

### Options considered
1. **Model draws/no-contests as a third class (multiclass, or a predicted draw probability)** — more "complete" in that it doesn't discard information, but draws/NCs are rare enough in UFC (well under 2% of bouts) that there's too little signal to learn a reliable draw probability. It also complicates every downstream piece: loss function, calibration, the ROI backtest (books don't price draws the same way as moneylines), and the frontend probability bar.
2. **Exclude draws/no-contests from training data; keep the label strictly binary (win/loss)** — the common approach for this problem type. Loses the handful of draw/NC rows entirely, but keeps the model, calibration, and backtest simple and matches how the betting market itself is structured (moneylines are effectively win/loss with draws refunded or handled as a separate prop).

### Decision
Exclude draws and no-contests from the training data. The label stays strictly binary (fighter A wins / fighter B wins).

### Why
Draws and no-contests are too rare in the dataset to support a reliable third class, and this is the standard treatment in UFC prediction models for that reason. Keeping the label binary also keeps every downstream piece simpler — loss function, calibration (Platt/isotonic expects a binary target), and the backtest, since sportsbook moneylines are themselves structured around a win/loss outcome. The tradeoff is explicitly accepted: those bouts contribute nothing to training, and the model will never be asked to output a draw probability.

### Consequences
Simplifies the label, the loss function, and calibration — no multiclass handling needed anywhere in the pipeline. It forecloses ever surfacing a "predicted draw" probability in the UI without a separate model change later. It also means the bout-level schema needs an explicit way to mark a row as excluded from training (rather than silently mis-encoding a draw as a loss for one side) — worth calling out during schema design so the training-matrix query can filter on it directly instead of inferring it. Deferred: if the dataset grows enough that draws stop being statistically negligible, or if the project later wants a genuine 3-way market (some books do offer draw props), this decision would need revisiting.
---

## [ADR-002] Exclude betting odds from model features

**Date:** 2026-08-01
**Status:** Accepted

### Context
Odds are available from The Odds API and correlate strongly with fight outcomes. Including them as a feature would likely boost measured accuracy.

### Options considered
1. **Include odds as a feature** — highest raw accuracy, but the model would mostly be learning to parrot the market rather than predicting the fight itself.
2. **Exclude odds entirely from features; use only as a baseline/backtest input** — lower raw accuracy ceiling, but the accuracy that remains is actually the model's own signal.

### Decision
Exclude odds from all model features. Use odds only as (a) a baseline to beat and (b) an input to the ROI backtest.

### Why
The project's actual value proposition is "does the model find real signal," not "can it copy Vegas." A model fed the odds would report inflated accuracy that means nothing and would collapse the moment odds were removed — the classic leakage failure mode for sports-prediction hobby projects.

### Consequences
Accuracy ceiling is capped near the favorite-baseline rate (~65–68%) rather than looking artificially higher. This is explicitly the honest tradeoff described in the project plan.

---

## [ADR-001] Use gradient-boosted trees (LightGBM) instead of a neural network

**Date:** 2026-08-01
**Status:** Accepted

### Context
~8,000 usable historical UFC fights, tabular feature set (physical stats, career rates, Elo, style clusters). Need a model that's fast to iterate on, debuggable solo, and appropriate for the dataset size.

### Options considered
1. **Neural network (MLP or similar)** — trendier, but needs far more data to beat tree-based methods on tabular data, harder to debug, slower iteration loop.
2. **Gradient-boosted decision trees (LightGBM)** — well-suited to small/medium tabular datasets, fast training, built-in feature importance, easier to reason about failures.
3. **Plain logistic regression only** — fine as a baseline, likely leaves accuracy on the table vs. GBDT.

### Decision
LightGBM as the primary model, logistic regression kept as a baseline, blended in an ensemble in Week 3.

### Why
At this dataset size, GBDTs consistently outperform neural nets on tabular data and iterate 10x faster, which matters directly given the 6-week timeline.

### Consequences
Forecloses (for now) any deep-learning learning goal from the original wishlist — explicitly deferred to "long-term, if the dataset grows or sequence data is added."

---

<!-- Add new entries above this line, newest first -->