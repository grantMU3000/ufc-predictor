# Leakage Log

This file is the audit trail for data leakage in the UFC predictor pipeline — every time a metric looks suspiciously good, or a leak is found and fixed, it gets an entry here.

**Why this file exists**: every hobby UFC predictor that claims 75%+ accuracy has one of three bugs — career-aggregate stat leakage, corner/ordering leakage, or a random train/test split. This project's credibility rests on being able to show, in writing, that these were actively hunted rather than assumed away.

**Rule of thumb:** any single change (new feature, split method, join) that moves accuracy more than ~3 points is a leak until proven otherwise. Log it *before* you've fully explained it — half-investigated entries are fine and expected. Update the entry as you learn more; don't delete and rewrite.

**Log every suspicious result, not just confirmed leaks.** A ruled-out suspicion is worth keeping — it stops you (or a reviewer) from re-investigating the same dead end later.

---

## How to file an entry

Copy the template below, fill in what you know now, and prepend it to the "Entries" section (newest first). Leave fields blank rather than guessing — `TBD` is a valid value while you investigate.

```markdown
### [YYYY-MM-DD] <short title>

- **Status:** 🔴 Open / 🟡 Investigating / 🟢 Resolved — leak confirmed & fixed / ⚪ Resolved — not a leak
- **Where noticed:** <script / notebook / PR / metric dashboard>
- **Symptom:** <what you saw — e.g. "val accuracy jumped from 66% to 79% after adding `td_avg_per15`">
- **Suspected category:** career-aggregate stat leakage / corner-ordering leakage / temporal split leakage / odds-derived leakage / other (name it)
- **Hypothesis:** <your best guess at the mechanism before you've confirmed it>
- **Investigation:**
  - <step taken, what you checked, what you found — add lines as you go>
- **Root cause:** <once known — the exact line/join/feature responsible>
- **Fix:** <what changed — commit SHA or PR link>
- **Verification:** <how you confirmed the fix worked — re-run metric, shuffle-label test, train/val distribution check>
- **Metric impact:** before → after (e.g. val accuracy 79% → 67%, log loss 0.51 → 0.61)
- **Lesson / guardrail added:** <test, assertion, or review-checklist item added so this can't silently recur>
```

---

## Quick-reference: known leakage categories for this project

Use these as a checklist when investigating — don't rediscover them from scratch each time.

1. **Career-aggregate stat leakage** — any fighter stat computed over their *whole* career (not as-of the day before the fight) has seen the future. Every Tier 2 feature must be `f(fighter_id, as_of_date)` using only prior bouts.
2. **Corner/ordering leakage** — ufcstats lists winners in the red corner disproportionately. Check that fights are symmetrized (both orderings, flipped labels) or that only differential features are used with randomized sides.
3. **Temporal split leakage** — a fighter's later-career form leaking into an earlier prediction via a random (non-time-ordered) split. Split must be strict: train ≤2022, val 2023–24, test 2025+, and a fighter's post-cutoff fights must never inform pre-cutoff features.
4. **Odds-derived leakage** — betting odds (or anything downstream of them, e.g. an "implied probability" feature) sneaking into the training features. Odds are baseline/backtest-only, never model input.
5. **Train/val/test cross-contamination** — the same fight, or derived rows from it (e.g. both symmetrized orderings), ending up split across train and val/test.
6. **Join/dedup artifacts** — a fuzzy name-match or dedup step that accidentally merges a fighter's future record into a past row.

Sanity checks to run whenever a metric looks too good:
- **Shuffle-label test**: randomly permute the target and retrain — accuracy should collapse to ~50%. If it doesn't, something is leaking through the features regardless of the label.
- **Feature ablation**: drop each feature/feature-group and re-measure; a single feature responsible for a huge jump is a suspect.
- **Train/val distribution check**: does a feature's distribution differ between train and val in a way it shouldn't (e.g. a stat that's suspiciously more separable post-cutoff)?

---

## Entries

_(none yet — first suspicious result goes here)_
