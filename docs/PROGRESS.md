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

## 2026-08-20 (Week 3, Day 1)

**Planned today:** Tuesday's plan deliverable — LightGBM research/exploration, Tier 3 features (SoS, style clustering, short-notice, layoff interactions), measure each addition's delta, and the ADR-014 calibration-bucket gate check for Glicko-2 RD.

**Shipped:**
- Researched gradient boosting internals / LightGBM (histogram training, GOSS, leaf-wise growth vs. XGBoost's depth-wise growth) — `docs/research/2026-08-20-LightGBM.md`; logged three follow-up ideas to `docs/IDEAS.md` (higher learning rate, more Optuna trials, try XGBoost) (`475fd69`)
- Ran the ADR-014 calibration-bucket gate check for Glicko-2 RD (`models/calibration_buckets.py`): the pre-registered rule initially fired (debut and 365-730d-layoff buckets showed ECE ~2.9x the full-val baseline), but a permutation test against a size-matched null showed every triggered bucket lands well inside its own null distribution (best case: 53rd percentile, p=0.47) — the pre-registered rule was measuring bucket *size*, not miscalibration. **Gate closed**, Glicko-2 RD not built; logged as ADR-015 with the full numbers, and as a revisit-at-test-unlock item in `IDEAS.md` (`9b9efd1`)
- Built and hand-verified Tier 3 `strength_of_schedule` (rolling mean of opponents' pre-fight Elo, leak-guarded via `.shift(1)`) against Khabib Nurmagomedov's real 13-fight career — 4 tests in `tests/test_tier3.py`, including a dedicated test that the current opponent's own Elo never enters their own SoS window (`9b9efd1`)
- Built and hand-verified `recent_damage_absorbed` (24-month trailing sum of strikes absorbed) against Joshua Van's career, including a boundary test that a fight 4 days past the 24-month cutoff is correctly excluded — 3 tests in `tests/test_tier3.py` (`e9e259f`)
- Added the weight-class-change feature (`build_weight_class_change_by_bout`) and wired all three new Tier 3 features (SoS, recent damage, weight class) plus two interaction terms (`layoff_x_age`, `age_x_experience`) into `build_train_val_with_elo()` via new `include_damage`/`include_weight` flags (`e9e259f`)
- (Uncommitted) Deleted three one-off/throwaway scripts no longer needed (`scripts/load_first_ufc_stats.py`, `scripts/odds_api_small.py`, `scripts/wiki_api_test.py`) — cleanup, not logic changes

**Blocked / open questions:**
-

**Research (1hr):** Gradient boosting internals / LightGBM — `docs/research/2026-08-20-LightGBM.md`

**Tomorrow's first task:** None of today's three new Tier 3 features (SoS, recent damage, weight-class-change) have a measured val-set delta yet — per the Tuesday plan line ("measure each addition's delta"), run each through the LightGBM val evaluation (individually and combined) and log accuracy/log loss/Brier/ECE deltas to `docs/RESULTS.md` before treating them as part of the leading feature set. Then commit the uncommitted script deletions and confirm CI is green.

**Energy / notes:**

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-19 (Week 2, Day 8)

**Planned today:** Saturday's plan deliverable — leakage audit day (shuffle labels, drop each feature group, check train/val drift, log to `LEAKAGE_LOG.md`) — and start implementing LightGBM.

**Shipped:**
- Leakage audit fully checked, and clean
- Implemented untuned LightGBM on the full feature set + combined Elo signal (`features/build_lgbm_matrix.py`, `models/lightgbm_model.py`), logged to `docs/RESULTS.md`: 0.6165 accuracy / 0.6589 log loss / 0.2329 Brier full val, 0.6246 / 0.6535 / 0.2306 odds-covered subset — best accuracy and log loss of any model so far (`e47cd64`)
- Added `lightgbm` + supporting deps to `pyproject.toml`/`uv.lock` (`8eaaab4`)
- Built `models/cv.py`: expanding-window CV for hyperparameter search, growing by year with 1999–2010 folded in as a single first window (thin early history). Added `tests/test_cv.py` (`ac23c60`)
- Tuned LightGBM via Optuna (60 trials, expanding-window CV on train only, val untouched) and evaluated: 0.6224 / 0.6529 / 0.2304 full val, 0.6305 / 0.6483 / 0.2282 odds-covered — a real but modest improvement over untuned on accuracy/log loss. Flagged the tradeoff in `docs/RESULTS.md`: ECE moved the wrong direction (0.0214 → 0.0318, odds-covered) — still inside the ≤0.05 target but a genuine calibration cost from the model being more confident, which the planned isotonic/Platt calibration step should correct (`4252e37`)
- Added `optuna` dependency and one CLAUDE.md rule (ruff/mypy compliance extends to any up/downstream code touched) (`2dce71d`)
- (Uncommitted) Saved tuned hyperparameters to `models/artifacts/lgbm_best_params.json`
- (Uncommitted) Cosmetic cleanup in `scripts/load_first_ufc_stats.py`, `scripts/odds_api_small.py`, `scripts/wiki_api_test.py` (quote style, line wraps, trailing newlines) — no logic changes
- (Uncommitted) Backfilled Aug 16–18 entries into `docs/PROGRESS.md`

**Blocked / open questions:**
-

**Research (30 min):** — Data leakage taxonomy

**Tomorrow's first task:** Make CI green, then commit today's uncommitted work (`lgbm_best_params.json`, script formatting, PROGRESS.md backfill), then move to the calibration step flagged in today's tuned-model results — isotonic/Platt calibration to pull ECE back down before the tuned model is treated as the leading candidate.

**Energy / notes:**

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-18 (Week 2, Day 7)

**Planned today:** Elo/Glicko implementation + tuning of K-factor and weight-class priors (Friday's plan deliverable).

**Shipped:**
- Implemented the first version of the Elo feature (`features/elo.py`) with global, weight-class-blind, sequential ratings — pre-fight ratings recorded before any update touches them so a bout's own result can't leak into its own prediction (`b4b0d9e`)
- Added experience-based K-factor decay (higher K for low-fight-count fighters, lower for established ones) with tests in `tests/test_elo.py` (`71f87d3`)
- Tuned and landed the Elo baseline in `models/baselines.py`: starter K=80, veteran K=24, smooth decay scale=3 (`b1314ea`)
- Logged ADR-014 (experience-based K over constant K; Glicko-2 deferred to Week 3 Tuesday, gated on whether ECE-by-experience-bucket shows a real calibration gap) and the Elo baseline's validation results in `docs/RESULTS.md`: 0.5610 accuracy / 0.6781 log loss / 0.2426 Brier full val, 0.0059 ECE — weakest of the three baselines on accuracy/log loss as expected for a single Tier 3 signal in isolation, but lowest ECE of any baseline so far (`56c4eda`)
- (Uncommitted) Added `scikit-learn` dependency to `pyproject.toml`, tweaked `scripts/load_first_ufc_stats.py`, `scripts/odds_api_small.py`, `scripts/wiki_api_test.py`, and three new K-factor tuning scripts (`scripts/elo_k_factor_search.py`, `scripts/elo_experience_k_search.py`, `scripts/elo_spot_check.py`) plus a `data/tuning/` output directory — not yet committed
- (Uncommitted) One more research note added to `docs/research/2026-08-18-KFactor.md` on Glicko's loss-punishment/longevity tradeoff

**Blocked / open questions:**
-

**Research (1hr):** Elo rating system (K-factor) — `docs/research/2026-08-18-KFactor.md`

**Tomorrow's first task:** Commit today's uncommitted work (scikit-learn dep, tuning scripts, script tweaks) and confirm CI is green, then move to Saturday's plan deliverable — **leakage audit day**: shuffle labels and confirm accuracy collapses to ~50%, drop each feature group and re-measure, check for train/val distribution drift, and write findings to `LEAKAGE_LOG.md`.

**Energy / notes:**

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-17 (Week 2, Day 6)

**Planned today:** Check the Postgres DB and confirm the ingestion data is good (`docs/IngestWorkflow.md` step 6 onwards), then move to baselines: (1) always-favorite, (2) higher-Elo-wins, (3) logistic regression on differential features, recording accuracy/log loss/Brier on the validation split.

**Shipped:**
- Diagnosed and repaired the UFC 330 event duplication (Greco's ingest inserted a second `events` row instead of updating the existing Wikipedia-sourced one, since it matched on `source_url` which was still `NULL`) via a one-off merge script, then built `data/ingestion/reconciliation.py` — `claim_existing_events_for_greco`/`claim_existing_bouts_for_greco` write Greco's `source_url` onto the matching pre-existing row before the normal upsert runs, so it can't recur. Added `check_duplicate_bouts` alongside the existing `check_duplicate_events` quality check (`e0e74b8`)
- Researched ranking systems (Elo vs. Glicko vs. TrueSkill) ahead of building the Elo baseline — `docs/research/2026-08-17-RankingSystems.md` (`0146030`)
- Built the market baseline: `features/odds.py` (`get_closing_lines`, de-vigged), `models/baselines.py` (`market_baseline`), `models/metrics.py` (`evaluate`, `reliability_curve` for ECE), and `features/differential.py` (`to_differential` — self-minus-opp feature prep for the upcoming LR baseline). First validation-set result logged in `docs/RESULTS.md`: market baseline scores 0.6936 accuracy / 0.5897 log loss / 0.2019 Brier at 91.9% odds coverage (1,870/2,034 val rows) (`b862650`)
- (Uncommitted) Logged ADR-013 in `docs/DECISIONS.md`: the reconciliation direction rule (whichever source's row lands first keeps its `id`; the later source only contributes data), and the consequence that settlement logic (Week 4) must key on `predicted_winner_id`, never corner position
- (Uncommitted) Ran `ruff format` across the repo — cosmetic reflow only (blank lines, line wraps), no logic changes; touches ~55 files, not yet committed

**Blocked / open questions:**
-

**Research (30 min):** Ranking systems (Elo/Glicko/TrueSkill) — `docs/research/2026-08-17-RankingSystems.md`

**Tomorrow's first task:** Build the logistic regression baseline on `to_differential`'s output and log it to `docs/RESULTS.md` alongside the market baseline. Then make CI green.

**Energy / notes:**

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-16 (Week 2, Day 5)

**Planned today:** Finish Wednesday's deliverable — temporal split (train ≤2022, val 2023–24, test 2025+) on top of yesterday's symmetrization.

**Shipped:**
- Researched time-series cross-validation methods (simple time split vs. sliding/expanding window); logged findings in `docs/research/2026-08-16-CrossValidation.md`. Confirmed simple time split is the right starting point for this project (`4da611e`)
- Built `features/split.py`: `temporal_split()` cuts the symmetrized dataset into train/val/test by `event_date` with half-open boundaries at `VAL_START`/`TEST_START`, plus `validate_split()` to guard against a bout's paired self_/opp_ rows landing in different splits. Added `tests/test_split.py` (`b5c03c6`)
- Fixed CI lint failures (ruff import ordering/unused imports, a mypy missing-annotation catch in `split.py`) and gitignored `data/processed/` and `data/test_locked/` now that `split.py` materializes them locally (`2d3d7fb`)
- Added `odds_snapshots` to the Postgres → Parquet export in `features/snapshot.py` (`a8c1cad`)
- Ran the weekly data ingest per `docs/IngestWorkflow.md`: pulled Greco's updated fight-stats CSVs and ran the Wiki ingest for upcoming events, then resolved fighter conflicts (steps 3–5) — updated `data/ingestion/logs/unresolved_bout_fighters.csv` and cleared `unresolved_fighters.jsonl`, not yet committed

**Blocked / open questions:**
-

**Research (30 min):** Time-series cross-validation — `docs/research/2026-08-16-CrossValidation.md`

**Tomorrow's first task:** Check the Postgres DB and confirm the ingestion data is good — `docs/IngestWorkflow.md` step 6 onwards (row counts, bouts & bout stats check, quality check, then document row counts again). Then move to Thursday's plan deliverable — baselines: (1) always-favorite, (2) higher-Elo-wins, (3) logistic regression on differential features, recording accuracy/log loss/Brier for each on the validation split.

**Energy / notes:**

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-15 (Week 2, Day 4)

**Planned today:** With Tiers 1 & 2 complete and unit-tested, move to Wednesday's plan deliverable: symmetrization (dual rows / differential features, per ADR in `docs/DECISIONS.md`) so the model can't key off corner position, then the strict temporal split (train ≤2022, val 2023–24, test 2025+) with the test set moved to a locked directory.

**Shipped:**
- Built `features/labels.py`: `get_completed_decided_bouts`, the single source of truth for which bouts are eligible training rows (`status = 'completed'` and `winner_id IS NOT NULL`) — kept deliberately separate from the feature functions so the label can never leak into `store.py`
- Built `features/symmetrize.py`: reshapes `store.py`'s red_/blue_ rows into self_/opp_ rows (two rows per bout) to kill corner-position leakage per ADR-004, dropping `stance_matchup`'s two fields as a redundant mirror rather than double-feeding the model the same fact
- Established the leakage-audit reference baseline: raw red-corner win rate (pre-symmetrization) is 0.6319, logged in `docs/LEAKAGE_LOG.md` — if a trained model's implied red-corner advantage approaches this, symmetrization has failed
- Added `tests/test_symmetrize.py` to test the symmetrization logic for leaks (`6bb3fe7`)

**Blocked / open questions:**
-

**Research (1hr):** —

**Tomorrow's first task:** Make CI green, then build the top-level assembly script that ties `store.py` → `labels.py` → `symmetrize.py` together into the full training dataset, and apply the strict temporal split (train ≤2022, val 2023–2024, test 2025+) with the test set moved into a locked directory — the piece "orchestration" in today's commit message sets up but doesn't finish yet.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-14 (Week 2, Day 3)

**Planned today:** Resolve any outstanding CI issues, then keep building out the rest of the Tier 2 features in `tier2.py` (point-in-time career-rate features beyond the strike/takedown rates already built). Unit tests for Tier 1 + Tier 2 will come once the full feature set is built out, not per-tier.

**Shipped:**
- Researched data leakage taxonomy further, ahead of the pipeline audit — `docs/research/2026-08-14-DataLeakage2.md` (`e70e11a`)
- Wrote and passed unit tests for all 39 Tier 1 + Tier 2 features in `tests/test_features.py` (`9a2b47b`)
- Completed the feature store: finished out the remaining Tier 2 point-in-time career-rate features in `tier2.py` and fleshed out `store.py`, then fixed the resulting ruff/mypy errors — Tiers 1 & 2 are now fully implemented (`01b5909`)
- Added foreign key indexes to support feature-store query patterns (migration `..._add_query_indexes_for_feature_store_and_...`) (`1e7586c`)

**Blocked / open questions:**
-

**Research (1hr):** Data leakage taxonomy, part 2 — `docs/research/2026-08-14-DataLeakage2.md`

**Tomorrow's first task:** With Tiers 1 & 2 complete and unit-tested, move to Wednesday's plan deliverable: symmetrization (dual rows / differential features, per ADR in `docs/DECISIONS.md`) so the model can't key off corner position, then the strict temporal split (train ≤2022, val 2023–24, test 2025+) with the test set moved to a locked directory.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-13 (Week 2, Day 2)

**Planned today:** Compute the remaining Tier 1 features in `tier1.py` (`age_at_fight` was the only one built so far), then write hand-computed unit tests for `get_prior_bouts` and the Tier 1 features in `test_features.py`. After that, start Tier 2 point-in-time career-rate features in `tier2.py`.

**Shipped:**
- Built out the rest of the Tier 1 features in `tier1.py`: `height_at_fight`, `reach_at_fight`, `reach_to_height_ratio`, `stance_at_fight`, `stance_matchup`, plus bout-context lookups `weight_class_at_bout`, `is_title_fight_at_bout`, `scheduled_rounds_at_bout` (`1c8e53a`), then fixed the resulting ruff failures (`e01209e`)
- Started Tier 2 point-in-time career-rate features: `get_fight_duration_seconds`, `get_total_seconds_fought`, `_rate_per_time_window`, `strikes_landed_per_minute`, `strikes_absorbed_per_minute`, `takedowns_landed_per_15` in `tier2.py`; also fleshed out `store.py` (`_get_bout_context`, `build_feature_row`) into a Tier 1-ready feature store (`63052e9`)
- Researched Elo rating systems for feature engineering — `docs/research/2026-08-13-Feature2.md` (`19cbd74`)
- Resolved a fighter name conflict from upcoming-bouts ingestion (Eduardo Chaplin), clearing it from `unresolved_fighters.jsonl` (`bf43a2b`)

**Blocked / open questions:**
-

**Research (1hr):** Elo rating systems for sports prediction — `docs/research/2026-08-13-Feature2.md`

**Tomorrow's first task:** Resolve any outstanding CI issues, then keep building out the rest of the Tier 2 features in `tier2.py` (point-in-time career-rate features beyond the strike/takedown rates already built). Unit tests for Tier 1 + Tier 2 will come once the full feature set is built out, not per-tier.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-12 (Week 2, Day 1)

**Planned today:** Confirm the quality suite is catching real Greco/Wikipedia dupes correctly on a completed-event update. Also run unit test coverage for the quality checks themselves, and fix the CI corrections still outstanding.

**Shipped:**
- Fixed CI (red since 40e0bbf, 5 consecutive failing runs): ruff import-order/unused-import errors, a mypy typing gap in `fighter_resolution.py`'s alias-index build (`itertuples()` → `zip()`), and a pytest module-resolution gap (missing `pythonpath` config, plus a `tests/fixtures/` directory the wiki_parsers tests expected but that had never been committed) (`b675913`)
- Resolved a Wikipedia API name-conflict pull; one fighter (Timothy Cuamba) couldn't be resolved and was logged to `unresolved_fighters.jsonl` for manual review rather than guessed
- Investigated ~70 quality-check false positives on title fights scheduled for 3 rounds (mostly TUF/Road to UFC tournament finales, plus one legitimate historical exception) — individually confirmed each, then reused `rounds_confirmed` to suppress them so the check stays a clean signal going forward. Logged as ADR-012, including a caveat that `rounds_confirmed` isn't enforced the same way on the Greco loader as it is on the Wikipedia loader (`1f34c2c`)
- Researched feature engineering for time-series/sports — `docs/research/2026-08-12-Features.md` (`3ba77a7`)
- Started feature store design (Week 2 Monday deliverable): built `bout_history.py` (`get_prior_bouts`, the shared leak-safe "everything before this date" function), `bout_stats_history.py` (round-by-round stat history), `snapshot.py` (Postgres → Parquet point-in-time snapshotting), and the first Tier 1 feature (`age_at_fight`); stubbed `store.py`, `tier2.py`, and `test_features.py` as placeholders for tomorrow (`fe4e665`)

**Blocked / open questions:**
-

**Research (1hr):** Feature engineering for time-series/sports — `docs/research/2026-08-12-Features.md`

**Tomorrow's first task:** Compute the remaining Tier 1 features in `tier1.py` (`age_at_fight` is the only one built so far), then write hand-computed unit tests for `get_prior_bouts` and the Tier 1 features in `test_features.py` — closes out Monday's "unit-test each against a hand-computed example" bar before more untested functions get built on top. After that, start Tier 2 point-in-time career-rate features in `tier2.py`.

**Energy / notes:**
- The research today really made me realize how difficult it is to feature engineer to the appropriate extent that sports predictions need. Especially for UFC, a sport that's so unpredictable.
- I'll attempt to formulate as many different feature ideas as I can and get them documented 

**Metrics check (weekly only, Fridays):** —

---

## 2026-08-11 (Week 1, Day 8)

**Planned today:** Resolve the fighter name-collision bugs from yesterday's ingestion run — 4 unresolved fighters (Choi Doo-ho, Osman Diaz, Yoo Joo-sang, Wesley Schultz) — and check whether `alias_collisions.jsonl` double-logging each row is a real dupe-write bug in `fighter_resolution.py`.

**Shipped:**
- Researched idempotent ETL pipelines further — how the pipeline should handle new bouts, scheduled bouts, and pre-existing events consistently (`docs/research/2026-08-11-Idempotent2.md`) (`f86e0bd`)
- Fixed default scheduled rounds: non-main-card/title fights confirmed by a human at 5 rounds no longer get silently reset to the default (`40e0bbf`)
- Resolved the fighter name collisions flagged yesterday (`4c75f0c`)
- Logged key decisions for the upcoming-events ingest pipeline: tracking Wikipedia identities via a dedicated column, and storing upcoming events/bouts in the existing tables rather than separate staging tables (`docs/DECISIONS.md`) (`82c6bd8`)
- Updated `docs/PLAN_ADDENDUM.md` to reflect the Wikipedia pipeline work in progress (`ac43223`)
- Built a data quality suite (`data/ingestion/quality_checks.py`) to guard bout/event integrity — specifically checks Wikipedia ingestion health and that Greco/Wikipedia events don't get duplicated when completed events are later updated; added integration + unit test coverage (`tests/integration/test_upcoming_events_loader_integration.py`, `tests/test_wiki_api.py`, `tests/test_wiki_parsers.py`) (`82abe9f`)

**Blocked / open questions:**
-

**Research (1hr):** Idempotent ETL pipelines (part 2) — `docs/research/2026-08-11-Idempotent2.md`

**Tomorrow's first task:** Confirm the quality suite is catching real Greco/Wikipedia dupes correctly on a completed-event update. Also run unit test coverage for the quality checks themselves, and fix the CI corrections still outstanding.

**Energy / notes:**
-

**Metrics check (weekly only, Fridays):** —

---

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