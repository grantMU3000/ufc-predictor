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

### [2026-08-16] Shuffle-label test — Week 2 Saturday audit

- **Status:** 🟢 Resolved — not a leak
- **Where noticed:** `notebooks/leakage_audit.py`, `shuffle_label_test()`
- **Symptom:** N/A — proactive check, not triggered by a suspicious result
- **Suspected category:** general (tests all categories at once)
- **Investigation:**
  - Shuffled `y_train` only (seed=42), refit the LR pipeline, scored against real, untouched `y_val`
  - Result: accuracy=0.5211, log_loss=0.6921 (theoretical coin-flip is ln(2)=0.6931), brier=0.2495, ece=0.0059 — collapses to ~coin-flip as expected
- **Root cause:** N/A
- **Fix:** N/A
- **Verification:** numbers land within expected sampling noise of a true coin-flip
- **Metric impact:** N/A (diagnostic only)
- **Lesson / guardrail added:** `shuffle_label_test()` kept in repo — worth rerunning any time a future model's accuracy jumps suspiciously

---

### [2026-08-16] Corner-ordering leakage check — naive baseline vs. real LR model

- **Status:** 🟢 Resolved — not a leak (era drift explains the naive-baseline number; real model shows no corner leakage)
- **Where noticed:** `notebooks/leakage_audit.py`, `naive_corner_only_baseline()` + `corner_symmetry_check()`
- **Symptom:** `naive_corner_only_baseline()` came in at 0.5565 accuracy, well below the ~0.6319 whole-dataset red-corner win rate already on record
- **Suspected category:** corner-ordering leakage / possible bug in `source_corner` derivation
- **Investigation:**
  - `print_corner_win_rates_by_era()`: train red-corner win rate = 0.6529 (n=6604), val = 0.5565 (n=1017) — confirms real era drift, not a bug. The original 0.6319 figure is a whole-dataset average and isn't directly comparable to a single-era subset.
  - `corner_symmetry_check()` on the actual LR model (`diff_` features only — `source_corner` is never an input): `lr_corner_red` acc=0.599803, `lr_corner_blue` acc=0.599803 — identical to 6 decimal places; log_loss/brier/ece near-identical across both corners
  - `source_corner` confirmed exactly 50/50 in both train and val
- **Root cause:** N/A — naive baseline's lower accuracy is genuine train/val era drift, not a bug
- **Fix:** N/A
- **Verification:** corner symmetry check shows the real model performs identically regardless of source corner
- **Lesson / guardrail added:** the 0.6319 baseline should be read as whole-dataset, not assumed to hold in any single era — train/val-specific rates now on record (0.6529 / 0.5565) for future comparisons

---

### [2026-08-16] Feature ablation — physical stats contribute more than Tier 1's "weak" framing assumed

- **Status:** ⚪ Resolved — not a leak (real signal, contradicts `docs/PLAN.md` §2's original framing, not a bug)
- **Where noticed:** `notebooks/leakage_audit.py`, `feature_ablation_test()`
- **Symptom:** `minus_physical` showed a log_loss delta of +0.0149 — ~3.5x the next-largest group (`minus_striking`, +0.0043) — and a ~1.8pt accuracy drop (0.5998 → 0.5821)
- **Suspected category:** possible leak in physical features (age/height/reach)
- **Investigation:**
  - Confirmed physical features have no history-lookup mechanism that could leak: age is a plain date subtraction, height/reach/reach-ratio are static bio fields — unlike Tier 2 rate features, there's no "prior fights" window that could accidentally pull in a future fight
  - No other group showed a comparably outsized delta; two groups (`experience_recency`, `knockdowns`) showed small negative deltas, judged as noise at n=2034
- **Root cause:** N/A — not a leak, a real effect. `docs/PLAN.md` §2 labeled physical stats "Tier 1 — easy, weak"; this dataset shows them as the single largest per-group contributor
- **Fix:** N/A (documentation-only finding)
- **Verification:** mechanism-level reasoning (no as-of computation exists for this group to leak through) + no anomalous behavior in any other group
- **Lesson / guardrail added:** worth reflecting the corrected framing in the eventual README / model card — physical stats aren't "weak" for this dataset

---

### [2026-08-16] Train/val distribution drift — 7/31 features flagged, all explained by era effects

- **Status:** 🟢 Resolved — not a leak
- **Where noticed:** `notebooks/leakage_audit.py`, `train_val_distribution_check()`
- **Symptom:** KS test (alpha=0.01) flagged 7/31 `diff_` features: `total_ufc_fights`, `times_knocked_down`, `sub_attempts_per_15`, `sapm`, `title_fight_experience`, `striking_defense`, `striking_accuracy`
- **Suspected category:** temporal split leakage / parsing inconsistency between eras
- **Investigation:**
  - All 31 features confirmed mean=0.0 in both splits — expected, symmetrization forces this structurally
  - 3 flagged features showed **widened** spread in val (`total_ufc_fights`, `times_knocked_down`, `sapm`) — consistent with a wider range of career lengths existing by 2023–24 than in the sport's earlier years
  - 3 flagged features showed **narrowed** spread in val (`sub_attempts_per_15`, `striking_defense`, `striking_accuracy`) — consistent with the sport professionalizing and skill gaps narrowing over time
  - `title_fight_experience` barely crossed the threshold (p=0.0027, highest of the 7) with near-identical std (1.696 vs 1.687) — judged as noise at large n, not a real shift
  - All `ks_statistic` values were small (0.04–0.06) — small effect size despite low p-values, consistent with expected drift rather than a bug
- **Root cause:** N/A — real, explicable era effects
- **Fix:** N/A
- **Verification:** every flagged feature ties to a plausible real-world explanation; nothing showed an unexplainable jump
- **Lesson / guardrail added:** `train_val_distribution_check()` kept in repo — worth rerunning once test unlocks in Week 3

---

### [2026-08-16] Split integrity check — no contamination found

- **Status:** 🟢 Resolved — not a leak
- **Where noticed:** `notebooks/leakage_audit.py`, `split_integrity_check()`
- **Investigation:**
  - No `bout_id` appears in more than one of train/val
  - Every `bout_id` has exactly 2 rows (both `source_corner` values) within its own split — no bout's symmetrized pair is split across train/val
  - train: 6,604 bouts × 2 = 13,208 rows (matches); val: 1,017 bouts × 2 = 2,034 rows (matches)
- **Root cause:** N/A
- **Fix:** N/A
- **Verification:** all three sub-checks passed cleanly
- **Lesson / guardrail added:** rerun including `test` once Week 3 Friday's test-set unlock happens, to verify test's boundary against train/val as well

Raw red-corner win rate (all decided bouts, pre-symmetrization): 0.6319 — recorded 2026-08-15. Reference baseline for Saturday's leakage audit (ADR-004): if a trained model's implied red-corner advantage approaches this number, symmetrization has failed.