# UFC Predictor — Project Instructions

## Project
UFC fight outcome predictor. See `docs/PLAN.md` for the full execution plan, `docs/DECISIONS.md` for architecture decisions, `docs/PROGRESS.md` for the daily log, `LEAKAGE_LOG.md` for leakage audit findings.

## Stack
- Python 3.12, dependency management via `uv` (never call `pip` directly)
- Postgres (Neon) as the prod DB; DuckDB + Parquet for local feature dev
- LightGBM for modeling, scikit-learn for baselines/calibration
- FastAPI backend, Next.js frontend

## The line you must not cross
You (the agent) may write: scrapers, glue code, tests, boilerplate, CI config, CSS.
The human writes, by hand: feature functions, model code, the evaluation harness, and leakage defenses. Never write or materially rewrite these four without being explicitly asked — they are the entire learning goal and the entire interview surface for this project.

## Non-negotiable rules
- Never use a feature computed from a fighter's future fights. Every career-rate feature must be `f(fighter_id, as_of_date)`, using only bouts prior to that date.
- Never include betting odds as a model feature. Odds are baseline/backtest input only — see `docs/DECISIONS.md` ADR-002.
- Never use a random train/test split. Always strictly temporal: train ≤2022, validate 2023–2024, test 2025+.
- Treat any accuracy jump of more than ~3 points from a single change as a leak until proven otherwise. Log the investigation in `LEAKAGE_LOG.md` before accepting the result.
- The test set is locked until Week 3 of the plan. Flag it immediately if any change touches it before then.
- For any code written, ensure it follows ruff & mypy guidelines. This goes for any up/downstream code that's affected by what you write

## Conventions
- One PR per day, merged, with green CI (`ruff`, `mypy`, `pytest`)
- Reference the relevant PLAN.md week/day in commit messages
- Update `docs/PROGRESS.md` at the end of each work session, or invoke the `scribe` skill to draft it