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

## Log

## [ADR-001] Exclude betting odds from model features

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

## [ADR-002] Use gradient-boosted trees (LightGBM) instead of a neural network

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