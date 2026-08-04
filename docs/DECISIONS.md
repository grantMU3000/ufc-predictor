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