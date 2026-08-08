# UFC Fight Predictor — Short-Term Execution Plan

**Window:** Aug 1 – Sep 11, 2026 (6 weeks) + Sep 12–18 buffer
**Capacity:** 20–30 hrs/week (~150 hrs total)
**Budget:** $50–100/mo
**Definition of success:** a deployed, publicly reachable app that shows the next 4 weeks of UFC cards with calibrated win probabilities, logs every prediction before the fight happens, and scores itself afterward — with an honest, benchmarked accuracy claim you can defend in an interview.

---

## 0. Read this before anything else

### 0.1 Your accuracy target is set too low, and that matters

Betting favorites win roughly 65–70% of UFC fights. <cite index="22-1">Over the last decade, UFC favorites have won around 65% of the time</cite>, and <cite index="17-1">longer-run samples put favorites near 68%</cite>. Some recent years have been higher still.

So "60% accuracy" is a model that **loses to a one-line Python script** that always picks the cheaper moneyline. If you ship 60% and call it a win, the first competent interviewer who knows sports data will find the hole in about 90 seconds.

**Reframe the goal:**

| Metric | Target | Why |
|---|---|---|
| Accuracy | ≥ 65% | Parity with the "always pick favorite" baseline |
| Log loss | ≤ 0.62 | Measures *calibration*, not just win/lose. This is the real metric. |
| Brier score | ≤ 0.215 | Sanity check on log loss |
| Calibration error (ECE) | ≤ 0.05 | When you say 70%, it should happen ~70% of the time |
| Accuracy on **market-priced-close fights** (odds between -150 and +150) | ≥ 55% | This is where a model can actually add value |
| Backtested ROI vs. closing line | Report it. Don't promise it. | |

You still hit the stated project goal ("at least 60% accuracy") — you just hit it *and* prove it means something. **The honesty is the resume asset.** A README that says "we achieve 66.1% accuracy vs. a 67.3% favorite baseline; our value is in calibration and in identifying mispriced close fights" is far more impressive than "70% accuracy!!" with a leaky pipeline behind it.

### 0.2 The single biggest technical risk: data leakage

Every hobby UFC predictor on GitHub that claims 75%+ accuracy has one of these three bugs. Assume you will make all three unless you actively defend against them.

1. **Career-aggregate stat leakage.** The fighter profile pages on ufcstats.com give you career SLpM, takedown accuracy, etc. — computed over their *entire* career, including fights that happen *after* the one you're predicting. Using those is time travel. You must recompute every fighter feature **as-of the day before the fight**, from that fighter's prior bouts only.
2. **Corner/ordering leakage.** ufcstats lists the winner in the red corner disproportionately often. A model given `(fighter_A, fighter_B)` in dataset order will learn "predict A" and look great. Fix: for every fight, emit **both** orderings with flipped labels, or use purely differential features (`reach_diff`, `slpm_diff`, ...) and randomize sides.
3. **Random train/test split.** Fights are time-ordered and fighters recur. A random split lets the model see a fighter's 2026 form while predicting their 2023 fight. Use a **strict temporal split**: train ≤ 2022, validate 2023–2024, test 2025 → present. Never touch the test set until Week 3.

**Rule:** any single change that jumps accuracy more than ~3 points is a leak until proven otherwise. Write it down in `LEAKAGE_LOG.md` and hunt it.

### 0.3 What gets cut from the short-term scope

| Cut | Why | When it comes back |
|---|---|---|
| Kubernetes | You will not need orchestration for one API container and ~100 users. Learning it now costs a week and buys nothing. | Long-term, or a separate weekend toy project |
| Video/footage analysis | Multi-month problem on its own. Data acquisition alone is a legal and technical wall. | Long-term goal #2 |
| Fine-tuning an LLM | Wrong tool. Tabular fight prediction is a gradient-boosting problem. | Never, probably |
| Deep learning / neural nets | ~8,000 usable UFC fights. GBDTs will beat a neural net on this dataset, and you'll debug 10x faster. | If the dataset grows or you add sequence data |
| Databricks | Enterprise-scale tooling for a dataset that fits in RAM twice over. | Job, not project |
| Real-money betting | Unvalidated model + real bankroll = expensive lesson. Paper-trade the whole short term. | After 3+ months of logged, out-of-sample results |

Keep from your "ideal concepts" list: **ML/modeling, data science, APIs, Docker, GitHub Actions, system design, agentic workflows.** That's 7 of 11 and it's plenty.

---

## 1. Architecture

Deliberately boring. Every piece is something you can debug alone at 11pm.

```
                    GitHub Actions (cron + event-driven)
                            |
          +-----------------+-------------------+
          |                 |                   |
   scrape_results     scrape_upcoming      refresh_odds
   (event-driven:     (every 3 days;       (6x daily)
   ~3hrs post-event,   every 12hrs
   retry ~15hrs post)  during fight week)
          |                 |                   |
          +--------> Postgres (Neon) <----------+
                            |
                    +-------+--------+
                    |                |
              FastAPI service   Model artifact
              (Docker, Fly.io)  (versioned, S3/R2 or repo LFS)
                    |
              Next.js frontend (Vercel)
```

### Stack + cost

| Layer | Choice | Cost/mo | Why this one |
|---|---|---|---|
| Language | Python 3.12 + `uv` | $0 | `uv` is fast and removes a whole class of env pain |
| Local analysis | DuckDB + Parquet | $0 | Query Parquet with SQL, no server. Perfect for feature dev. |
| Prod DB | Neon Postgres | $0–19 | Serverless Postgres, branching, generous free tier |
| Modeling | scikit-learn → LightGBM | $0 | LR baseline first, then GBDT. `scikit-learn` for calibration. |
| Experiment tracking | MLflow (local) or a `results.csv` | $0 | Don't over-tool. A disciplined CSV beats an unused dashboard. |
| Backend | FastAPI + Pydantic | $0 | Type-safe, auto OpenAPI docs, async |
| Container | Docker (multi-stage) | $0 | Learning goal + portable deploy |
| Hosting (API) | Fly.io or Railway | $5–20 | Real container hosting, not a serverless abstraction |
| Frontend | Next.js on Vercel | $0 | Free tier is more than enough |
| Scheduling | GitHub Actions cron | $0 | Learning goal, free for public repos |
| Errors | Sentry free tier | $0 | 5 min to wire up |
| LLM (stretch) | Anthropic API | $10–25 | Only in Week 6 |
| Odds | The Odds API | $0–30 | Free tier covers live/upcoming MMA; historical odds need a paid tier |

**Total: ~$20–70/mo.** Comfortably inside budget with room for the LLM stretch goal.

### Data sources

| Source | What | Notes |
|---|---|---|
| **ufcstats.com** | Every event, bout, round-by-round strike/TD/sub stats, fighter physicals | The canonical free source. Public, no auth. Scrape politely (1 req/sec, cache raw HTML). |
| **`Greco1899/scrape_ufc_stats` (GitHub)** | Pre-scraped CSVs of the above, refreshed daily | **Start here.** Use it as your Day-1 dataset so you're modeling by Week 2 instead of debugging BeautifulSoup. Then write your own scraper in parallel so you own the pipeline. |
| **The Odds API** | Live + upcoming MMA moneylines from multiple books | Free tier gives current odds; historical MMA odds back to June 2020 are on paid plans. Budget for one month of the paid tier to pull the historical backfill, then downgrade. |
| **ESPN public MMA API** | Rankings, event schedules, fighter bios | Undocumented but public and stable-ish. Good for the "next 4 weeks of cards" feature. |
| **Wikipedia UFC event pages** | Schedule backup, fight order | Fallback when the schedule scraper breaks (it will) |

**Cache every raw HTTP response to disk before parsing.** Re-scraping because you changed a regex is a waste of a whole evening.

### Scraping cadence — reasoning, not defaults

- **Results:** results only change the moment an event ends, and events happen roughly weekly. A blind daily timer is mostly no-op runs. Instead, trigger off the event calendar you already have: scrape ~3 hours after each event's scheduled end time, with a fallback retry ~12 hours after that (so ~15 hours post-event) in case the card ran long or the page lagged. This is both cheaper and more correct, since it's tied to the thing that actually changes the data.
- **Upcoming events:** fight cards change — injury replacements, added/removed bouts, occasional full reshuffles — sometimes with only days' notice. The risk of staleness is highest right before an event and low the rest of the time. So: check every 3 days by default, and tighten to every 12 hours during **fight week** (the 7 days leading into a scheduled event). This keeps the UI accurate when it matters most without running a needless scraper every single day of a quiet month.
- **Odds:** kept at 6x/day flat — odds move continuously (line movement is itself a signal you may want to capture), so this one legitimately benefits from a regular intraday cadence rather than an event-driven trigger.

---

## 2. Feature plan

Build these in three tiers. Ship Tier 1 + 2 in the short term; Tier 3 is where the actual edge lives.

**Tier 1 — Static / physical (easy, weak)**
`age_at_fight`, `height`, `reach`, `reach_to_height_ratio`, `stance`, `stance_matchup` (orthodox vs southpaw), `weight_class`, `is_title_fight`, `scheduled_rounds`

**Tier 2 — Point-in-time career rates (the workhorses)**
All computed **only from fights before this one**, with a `min_prior_fights` guard and explicit NaN handling for debutants:
`slpm`, `sapm`, `str_acc`, `str_def`, `td_avg_per15`, `td_acc`, `td_def`, `sub_avg_per15`, `control_time_pct`, `knockdown_rate`, `finish_rate`, `ko_loss_rate`, `sub_loss_rate`, `decision_rate`, `avg_fight_time`, `win_pct_last_5`, `days_since_last_fight` (layoff), `total_ufc_fights`, `title_fight_experience`, `cumulative_significant_strikes_absorbed` (damage proxy)

**Tier 3 — Contextual & relational (the differentiators)**
- **Elo / Glicko rating** per fighter, updated fight-by-fight, with a weight-class-adjusted variant. Cheap to implement, historically the strongest single feature in this domain.
- **Strength of schedule**: opponent Elo at time of fight, averaged over last N
- **Style matchup**: cluster fighters into archetypes (striker / wrestler / grappler / pressure / counter) via k-means on Tier 2 rates, then feed `archetype_A × archetype_B` interaction
- **Short-notice flag** (< 30 days between bout announcement and fight)
- **Weight-class change** (moving up/down since last fight)
- **Cage vs. home country**, altitude, card position (main/co-main/prelim)
- **Damage accumulation**: rolling significant strikes absorbed over trailing 24 months
- **Layoff × age interaction** (a 38-year-old off 18 months is a very different bet than a 26-year-old)

**Explicitly excluded from the feature set: the betting odds.** Include odds and the model just learns to parrot the market and your accuracy is meaningless. Keep odds strictly as (a) a baseline to beat and (b) the input to the ROI backtest.

---

## 3. Week-by-week plan

Each week has a **hard deliverable** and an **exit criterion**. If you miss an exit criterion, cut scope from the *next* week — do not extend the current one.

---

### Week 0 — Aug 1–2 (weekend, ~8 hrs) · Foundations

**Deliverable:** a repo you can work in, and an agentic workflow that runs.

- [ ] Repo: `ufc-predictor`, public, MIT. Monorepo: `/data`, `/features`, `/models`, `/api`, `/web`, `/infra`, `/notebooks`, `/docs`
- [ ] `uv init`, Python 3.12, pin deps. `ruff` + `mypy` + `pytest` configured
- [ ] Pre-commit hooks. GitHub Actions CI that runs lint + tests on every PR
- [ ] Neon Postgres project created, connection string in `.env`, `.env` gitignored
- [ ] `docs/PLAN.md` (this file), `docs/DECISIONS.md` (ADR log), `docs/PROGRESS.md` (daily log), `LEAKAGE_LOG.md`
- [ ] Agentic workflow scaffolded (see §4)
- [ ] Pull `Greco1899/scrape_ufc_stats` CSVs, load into DuckDB, run `df.describe()`, eyeball it

**Exit:** `git push` triggers green CI. You can run one command and get a DataFrame of UFC fights.

---

### Week 1 — Aug 3–9 (~25 hrs) · Data pipeline + schema

**Deliverable:** a reproducible, incremental ingest producing a clean fight-level table in Postgres.

| Day | Focus |
|---|---|
| Mon | Schema design. `events`, `fighters`, `bouts`, `bout_stats` (per-round), `odds_snapshots`, `predictions`, `prediction_results`. Write the migration (Alembic). |
| Tue | Your own ufcstats scraper: events index → event page → bout page → fighter page. Raw HTML cached to `/data/raw/`. Rate-limited, resumable. |
| Wed | Parsers + loaders. Fuzzy fighter-name resolution (`rapidfuzz`) — this is a real problem: "Jose Aldo" vs "José Aldo" vs "Junior Aldo". Build a `fighter_aliases` table. |
| Thu | Odds ingestion. Pay for one month of The Odds API historical tier, backfill MMA moneylines to 2020, join to bouts by (date, fighter pair). Expect ~10% match failures — log them, don't silently drop. |
| Fri | Upcoming-events scraper (ESPN + ufcstats + Wikipedia fallback). This feeds the "next 4 weeks" UI. |
| Sat | Data quality suite: row counts by year, null rates per column, duplicate bout detection, "did any fighter fight twice on the same night" check. Fail loudly. |

**Exit:** `make ingest` runs end to end, is idempotent, and Postgres holds ≥ 8,000 bouts with per-round stats plus ≥ 4,000 bouts with odds attached. Data-quality suite passes.

**Trap:** you will lose a day to name matching. Budget for it; don't be surprised by it.

---

### Week 2 — Aug 10–16 (~25 hrs) · Features + baselines

**Deliverable:** a leakage-free training matrix and three honest baselines.

| Day | Focus |
|---|---|
| Mon | Feature store design. One function per feature, signature `f(fighter_id, as_of_date) -> value`, all reading only prior bouts. Unit-test each against a hand-computed example. |
| Tue | Build Tier 1 + Tier 2 features. Materialize to Parquet. |
| Wed | **Symmetrization** (dual rows / differential features) + temporal split (train ≤2022, val 2023–24, test 2025+). Test set goes in a locked directory you don't read. |
| Thu | **Baselines.** (1) always-favorite, (2) higher-Elo-wins, (3) logistic regression on differential features. Record accuracy, log loss, Brier for each on validation. |
| Fri | Elo/Glicko implementation + tuning of K-factor and weight-class priors. |
| Sat | **Leakage audit day.** Shuffle labels — accuracy should collapse to ~50%. Drop each feature group and re-measure. Check for any feature whose distribution differs between train and val in a way it shouldn't. Write findings to `LEAKAGE_LOG.md`. |

**Exit:** validation numbers for all three baselines are in `docs/RESULTS.md`. Shuffled-label test yields ~50%. You can state your favorite-picking baseline number for your dataset exactly.

**This is the most important week in the plan.** If the feature pipeline is right, everything downstream is easy. If it's leaky, everything downstream is fiction.

---

### Week 3 — Aug 17–23 (~25 hrs) · Model

**Deliverable:** a frozen, calibrated, versioned model artifact that beats or matches baseline.

| Day | Focus |
|---|---|
| Mon | LightGBM. Sensible defaults first, then `optuna` with **time-series CV** (expanding window, never random KFold). |
| Tue | Tier 3 features: style clustering, SoS, short-notice, layoff interactions. Measure each addition's delta. |
| Wed | **Calibration.** Raw GBDT probabilities are overconfident. Apply isotonic or Platt on the validation set. Plot the reliability diagram — this plot goes in the README. |
| Thu | Ensemble: LR + LightGBM + Elo, blended. Usually worth 1–2 points of log loss for an hour of work. |
| Fri | **Test set unlock.** Run once. Whatever it says, that's your number. Also compute: accuracy on close fights, ROI backtest at closing odds with flat 1-unit stakes, and a Kelly-fraction simulation. |
| Sat | Freeze v1. Serialize with metadata (feature list, training cutoff, git SHA, metrics). `model_registry` table. Write `docs/MODEL_CARD.md`. |

**Exit:** `models/v1/` exists with a model card. Test-set metrics documented, including the comparison to the favorite baseline. Reliability diagram rendered.

**Discipline:** you get **one** look at the test set. If you tune against it, it becomes another validation set and you've lost your only honest estimate.

---

### Week 4 — Aug 24–30 (~25 hrs) · Backend

**Deliverable:** a Dockerized FastAPI service serving predictions for upcoming fights, with a full prediction audit trail.

| Day | Focus |
|---|---|
| Mon | FastAPI skeleton. `GET /events/upcoming?weeks=4`, `GET /fights/{id}/prediction`, `GET /predictions/history`, `GET /model/performance`, `GET /health` |
| Tue | Inference path: for an upcoming bout, compute features as-of today, run model, return calibrated probability + top feature contributions (SHAP or LightGBM gain). |
| Wed | **Prediction ledger.** Every prediction written to `predictions` with model version, feature snapshot, timestamp, and the market odds at time of prediction — *before* the fight. Immutable. This is your project goal #6 and it's also the thing that makes the whole project credible. |
| Thu | Settlement job: after each event, join results, compute rolling accuracy / log loss / paper ROI, write to `prediction_results`. |
| Fri | GitHub Actions jobs: event-driven results scrape (fires ~3hrs after each scheduled event end, retries ~12hrs later), upcoming-events refresh (every 3 days, tightened to every 12 hours during fight week), 6x-daily odds refresh, post-event settlement triggered off the same results job. Failure notifications to yourself. |
| Sat | Docker multi-stage build, deploy to Fly.io, wire Sentry, smoke test the live API. |

**Exit:** live URL returns real predictions for real upcoming fights. A cron/event-driven run has executed successfully unattended. Predictions from this week are already in the ledger.

---

### Week 5 — Aug 31 – Sep 6 (~25 hrs) · Frontend + ship

**Deliverable:** a public app a stranger can use.

| Day | Focus |
|---|---|
| Mon | Next.js scaffold, Tailwind, deploy an empty page to Vercel first (never leave deployment to the end). |
| Tue | Event list: next 4 weeks of cards, grouped by event, main-card/prelim split. |
| Wed | Fight card component: both fighters, records, key stat comparison, probability bar, confidence indicator, "why" panel with top 3 factors. |
| Thu | Track-record page: past predictions, hit/miss, rolling accuracy chart, model-vs-market comparison. |
| Fri | Polish: loading/empty/error states, mobile layout, dark mode, a real favicon, an "About the model" page that's honest about limitations and includes a not-betting-advice note. |
| Sat | Lighthouse pass, custom domain (~$12/yr), share with 5 people and watch them use it without helping. |

**Exit:** public URL. All six short-term goals demonstrably met. Someone who isn't you has used it and given feedback.

---

### Week 6 — Sep 7–11 (~25 hrs) · Stretch + presentation

**Deliverable:** the stretch goals, and the artifacts that convert this into job offers.

| Day | Focus |
|---|---|
| Mon | News/context ingestion: RSS from MMA outlets → store per-fighter articles with dates. Strictly as displayed context, not model input (too easy to leak). |
| Tue–Wed | **Chatbot.** RAG over: fighter stat history, bout history, your model's prediction + feature attributions, recent news. Tool-calling to your own API endpoints rather than a vector-search-only design — it'll be far more accurate. Rate-limit it and cap spend. |
| Thu | README as a portfolio piece: architecture diagram, the honest metrics table, reliability diagram, leakage-defense section, "what I'd do next." |
| Fri | Write-up / blog post. Record a 3-min demo video. Update resume and LinkedIn with concrete numbers. |

**Exit:** you can hand someone a link and a README and they understand the whole system in five minutes.

---

### Buffer — Sep 12–18

Reserved. It **will** get used. If it somehow doesn't: monitoring dashboard, load test, or start Tier-3 feature research for the long-term phase.

---

## 4. Your daily operating system

### Daily rhythm (~5 hrs on a full day)

| Block | Time | What |
|---|---|---|
| Standup | 15 min | Read yesterday's `PROGRESS.md` entry. Agent generates today's plan from this doc + open issues. You approve or edit it. |
| Research | 60 min | One topic from the curriculum below. **Notes go in `docs/research/YYYY-MM-DD-topic.md` and must end with "how this applies to my project."** Research without application is entertainment. |
| Deep work | 3 hrs | Single task, phone in another room. One PR per session. |
| Wrap | 30 min | Commit, push, update `PROGRESS.md`: what shipped, what's blocked, tomorrow's first task. Weekly: review metrics against exit criteria. |

**One PR per day, merged.** It forces scoping and it produces a green commit graph that recruiters actually look at.

### Research curriculum (1 hr/day, mapped to when you'll use it)

- **Week 0–1:** Postgres schema design & indexing · idempotent ETL patterns · rate-limited scraping · Alembic migrations
- **Week 2:** Feature engineering for time-series/sports · data leakage taxonomy · temporal cross-validation · Elo/Glicko math
- **Week 3:** Gradient boosting internals (how LightGBM splits) · probability calibration (Platt vs isotonic) · proper scoring rules · SHAP · Kelly criterion
- **Week 4:** FastAPI async & dependency injection · Docker multi-stage builds · GitHub Actions (matrices, secrets, cron, event-driven triggers) · model versioning & registries
- **Week 5:** Next.js App Router & server components · caching strategy · basic observability
- **Week 6:** RAG architecture · tool-calling agents · LLM evals · prompt injection basics

### Agentic workflow (your goal, done usefully)

Set up in `CLAUDE.md` at repo root, with `/commands` for repeatable prompts:

| Agent | Job | Runs |
|---|---|---|
| **Planner** | Reads `PLAN.md` + `PROGRESS.md` + open issues → proposes today's 3 tasks with time estimates | Each morning |
| **Manager** | Compares actual progress vs. this plan; flags slippage and names what to cut | Fridays |
| **Reviewer** | Reviews every PR for leakage patterns specifically (any feature touching future data), plus normal code review | On PR, via Actions |
| **Scribe** | Reads `git log` + diffs → drafts the `PROGRESS.md` entry for you to edit | End of day |

**The line you must not cross:** agents write scrapers, glue, tests, boilerplate, CI config, and CSS. **You** write the feature functions, the model code, the evaluation harness, and the leakage defenses — by hand, understanding every line. Those four things are the entire learning goal and the entire interview surface. If an agent writes your eval harness, you cannot defend your numbers, and the project is worth nothing to your career.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data leakage inflates results | **High** | Fatal to credibility | Week 2 audit day, shuffle test, `LEAKAGE_LOG.md`, reviewer agent |
| Model fails to beat favorite baseline | **Medium** | Moderate | This is a legitimate, publishable result. Pivot the pitch to calibration + close-fight performance. Do not fudge numbers. |
| ufcstats HTML changes mid-project | Medium | 1–2 days | Cached raw HTML, parser tests with fixture files, Greco CSVs as fallback |
| Name-matching between odds and stats | **High** | 1–2 days | Alias table, `rapidfuzz`, manual override CSV, log every failure |
| Scope creep (chatbot, video, K8s) | **High** | Miss deadline | §0.3 cut list is binding. New ideas go to `IDEAS.md`, not this week. |
| Frontend takes longer than budgeted | Medium | Miss deadline | Deploy an empty page Week 5 Monday. Ugly-but-working beats pretty-but-local. |
| Burnout at 25 hrs/wk on top of everything else | Medium | Project dies | One full day off per week, non-negotiable. Track energy in `PROGRESS.md`, not just tasks. |
| Sept UFC schedule is thin during the eval window | Low | Fewer live results | Backtest carries the evidence; live ledger is a bonus |
| Event runs long / results page lags, event-driven scrape fires too early | Low | Missed/incomplete results scrape | Fallback retry ~12hrs after the first attempt catches this; settlement job only runs once results are confirmed complete |

---

## 6. Exit checklist (mid-September)

Original goal → what "done" means:

1. **≥60% accuracy** → ✅ *and* reported alongside the favorite baseline, log loss, Brier, and ECE on a held-out temporal test set
2. **UI showing next 4 weeks + predictions** → ✅ public URL, mobile-usable
3. **Scalable database** → ✅ normalized Postgres, migrations, indexed, idempotent ingest
4. **Deployed for people to use** → ✅ custom domain, uptime monitoring, error tracking
5. **Scalable system for future dev** → ✅ CI/CD, Docker, typed API, model registry, ADR log
6. **Backend tracks previous predictions** → ✅ immutable pre-fight ledger + automated settlement + public track record

Stretch: chatbot ✅ · news context ✅ · public track-record page ✅

---

## 7. A note on the money goal

The betting market is the most efficient adversary you'll ever model against. <cite index="17-1">Closing lines price favorites almost exactly at their realized win rate</cite> — the market's implied probabilities have historically tracked actual outcomes within about a point. Beating that consistently, net of vig, is genuinely hard and is what quant sports firms do full-time.

That doesn't make the project pointless — it makes the *honest* framing the valuable one. Your realistic edges, in order of plausibility: early lines on prelim fights before sharp money arrives, props and method-of-victory markets that books price lazily, and your own calibration discipline.

For the short term: **paper trade only.** Log every prediction with the odds available at the time, settle it automatically, and let 3+ months of out-of-sample results accumulate before a dollar is at risk. If the model is real, it'll still be real in December. If it isn't, you'll have found out for free.

---

## 8. Day-one tasks (today)

1. `mkdir ufc-predictor && cd ufc-predictor && git init && uv init`
2. Push to GitHub, add the CI workflow, watch it go green
3. Create the Neon project, save the connection string
4. `git clone` the Greco1899 CSVs, load into DuckDB, answer three questions by hand: how many bouts total, what fraction end in decision, what's the red-corner win rate (this last one is your ordering-bias number — write it down)
5. Copy this file to `docs/PLAN.md`, create `docs/PROGRESS.md`, write today's entry
