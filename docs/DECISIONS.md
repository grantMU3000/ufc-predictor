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
## [ADR-019] Ensemble (LR + LightGBM + Elo): Gate A missed by 0.000015 — v1 ships as tuned LightGBM alone

**Date:** 2026-08-22
**Status:** Accepted

### Context

`docs/PLAN.md` §3's Week 3 Thursday entry calls for blending LR +
LightGBM + Elo. Motivated by ADR-016 (training volume saturated) and
ADR-017 (raw model already reasonably calibrated) leaving little else
to try before Friday's test unlock. A blend only helps where its
components make different mistakes, so a correlation/disagreement
diagnostic ran first (`models/ensemble.py`) before any blend was
built — it showed real room (LightGBM/LR correlation 0.845; either vs.
Elo 0.49-0.58; 42.3% disagreement rate). One diagnostic metric, raw
residual correlation, turned out structurally uninformative
(dominated by the shared `y_true` term regardless of model
similarity) and was misread as a warning sign at first. Fix identified
(split by outcome class), deferred to `IDEAS.md` — never fed into a
gate, so it doesn't affect this decision.

### Method

Four pre-registered gates, same discipline as ADR-014/015/016/017:

- **Gate A:** holdout log loss must beat the best single model's own
  holdout log loss by ≥ 0.002.
- **Gate B:** a stacker must beat the best fixed-weight blend by
  ≥ 0.002 to be preferred; ties go to the simpler blend.
- **Gate C (ADR-004 symmetry):** `max_pair_deviation` ≤ 1.5x the best
  single model's own deviation, same holdout.
- **Gate D (calibration guard):** holdout ECE must not worsen by more
  than 0.005 vs. the best single model's holdout ECE.

Accuracy was explicitly not a gate (`docs/PLAN.md` §0.1, reconfirmed
by ADR-016) — reported for parity only.

Same fit/holdout split as ADR-017 (fit 2011-2020, n=8,620; holdout
2021-2022, n=2,006). LightGBM and LR refit per fold (LR's
imputer/scaler included — fitted transforms, same leak risk). Elo not
refit — it's a sequential rating walk, already point-in-time-safe by
construction, no training set to have peeked at.

Seven candidates: three single models, equal-weight probability
average, equal-weight logit average, fitted-weight logit blend (all
three, non-negative weights), fitted-weight logit blend (LightGBM +
Elo only — testing whether LR was redundant), and two stackers
(with/without intercept — the intercept version included to give
Gate C something real to catch, since a fitted intercept breaks
ADR-004's `P(self)+P(opp)=1` invariant by construction).

### Results

Holdout (2021-2022, n=2,006):

| method | n | log loss | accuracy | ece | max_pair_dev | d_log_loss |
|---|---|---|---|---|---|---|
| single_p_lgbm | 1 | 0.663434 | 0.5972 | 0.0133 | 0.0702 | 0.000000 |
| single_p_lr | 1 | 0.664535 | 0.6162 | 0.0240 | 0.00001 | +0.001101 |
| single_p_elo | 1 | 0.679832 | 0.5623 | 0.0126 | ~0 | +0.016398 |
| equal_prob_mean | 3 | 0.662230 | 0.6107 | 0.0336 | 0.0234 | -0.001204 |
| equal_logit_mean | 3 | 0.662335 | 0.6097 | 0.0328 | 0.0233 | -0.001099 |
| **weighted_logit_3** | 3 | **0.661449** | 0.6067 | 0.0162 | 0.0526 | **-0.001985** |
| weighted_logit_lgbm_elo | 2 | 0.663173 | 0.5977 | 0.0042 | 0.0667 | -0.000261 |
| stacker_no_intercept | 3 | 0.661459 | 0.6072 | 0.0169 | 0.0547 | -0.001975 |
| stacker_intercept | 3 | 0.661457 | 0.6067 | 0.0163 | 0.0546 | -0.001977 |

Fitted weights, `weighted_logit_3`: lgbm=0.75, lr=0.23, elo=0.02.

**Gate D:** both equal-weight blends disqualified — ECE (0.0336,
0.0328) exceeded the limit (0.0133 + 0.005 = 0.0183).

**Gate C:** no failures, including `stacker_intercept`
(0.0546 vs. limit 0.1053) — contrary to expectation. Likely because
the symmetrized dual-row design forces a globally balanced target,
leaving the fitted intercept little room to drift from ~0.

**Gate A:** best surviving candidate, `weighted_logit_3`, reached
**-0.001985** — short of -0.002 by **0.000015**. Nothing cleared Gate
A; Gate B never reached.

**Contrary to the diagnostic's prediction:** every blend that improved
on `single_p_lgbm` weighted LR meaningfully (0.21-0.23); the
LightGBM+Elo-only blend (excluding LR) barely moved (-0.000261). Low
correlation with a *weaker* model isn't automatically useful diversity
— Elo's independence from LightGBM was real but not informative
enough to matter; LR's higher correlation still carried more usable
signal.

**Observation, not gated:** `single_p_lr` beats `single_p_lgbm` by 1.9
accuracy points while losing on log loss — the first time in the
project accuracy and log loss have ranked two models differently.
Brier agrees with accuracy. Not actionable (ADR-016 already settled
log loss as the headline metric), but worth a `docs/MODEL_CARD.md`
note.

### Decision

**No candidate clears Gate A. v1 ships as the tuned LightGBM alone**,
unchanged from ADR-017. Third pre-registered negative result of Week
3 (after ADR-015, ADR-016), fourth counting ADR-017.

### Why

The gate was fixed before any number was seen specifically so a
result this close couldn't be rationalized into a pass — every other
negative result this week is only credible because thresholds held
when inconvenient.

Worth distinguishing from ADR-016's negative result: that one was
scattered and unsigned (-0.0006 to +0.0009, no consistent direction).
Today, four independently-built blends all landed between -0.0011 and
-0.0020, all correctly signed. Real, small, consistent — "below the
bar," not "not there." Honest framing for the README: the ensemble
recovers roughly the low end of the 1-2 points `docs/PLAN.md` §3
predicted, just short of the bar set for the added inference
complexity.

### Consequences

**Easier:** Week 4's inference path stays single-artifact — one
`predict_proba` call per upcoming bout, no blend weights to load or
keep in sync with a retrained model.

**Standing tool:** `models/ensemble.py`'s gate/OOF/candidate structure
reuses the same pattern as `models/feature_deltas.py` (ADR-016) and
`models/calibration.py` (ADR-017) — ready if revisited at test unlock
with a larger holdout.

**Logged to `IDEAS.md`:** revisit the ensemble with a larger
population; fix the residual-correlation diagnostic; the LR-vs-LGBM
accuracy/Brier disagreement; LR-carries-more-signal-than-Elo as a
caution for any future Elo-adjacent feature (e.g. Glicko-2 RD, still
gated closed per ADR-015).

**Not foreclosed:** standalone Elo/LR numbers already in
`docs/RESULTS.md` are unaffected — this ADR concerns only whether
combining them with LightGBM clears a production bar.

**Realistic framing heading into Friday:** Tuesday's saturation
finding, Wednesday's calibration rejection, and today's near-miss
ensemble all land in the same place — small, real, sub-threshold
effects. The tuned LightGBM alone is likely close to this feature
set's ceiling. Friday's test unlock should be read with that
expectation set, not as a moment to go looking for a number these
three sessions didn't find.

---

## [ADR-018] Ensemble (LR + LightGBM + Elo): pre-registered gates

**Date:** 2026-08-22
**Status:** Proposed — gates registered before any blend is scored; results to follow

### Context

`docs/PLAN.md` §3's Week 3 Thursday entry calls for blending LR +
LightGBM + Elo, noting it's "usually worth 1–2 points of log loss for
an hour of work." Two findings from earlier this week point at this as
the most promising remaining lever before Friday's one-shot test
unlock: Tuesday's fold-trend check found training volume saturated
(4.7x more data, ~0 log-loss benefit — ADR-016), and Wednesday's
calibration work found the raw model already reasonably calibrated
(ADR-017) — neither more data nor recalibration had room left to give.
What hasn't been tried is combining different model *shapes*: LR reads
the feature set linearly, LightGBM reads threshold interactions, Elo
reads only the sequential win/loss record with no per-fight stats at
all. An ensemble only helps if these three make genuinely different
mistakes — a residual-correlation diagnostic runs before any blend is
built, specifically to check that assumption rather than take it on
faith.

### Options considered

1. **Equal-weight average, probability space** — simplest possible
   blend, zero fitted parameters.
2. **Equal-weight / weighted average, logit space** — averages evidence
   additively rather than pulling confident correct calls toward 0.5,
   which is what probability-space averaging does.
3. **Weighted logit blend, weights fit on OOF folds** — same
   mechanism as (2), non-negative weights summing to 1, fit rather
   than assumed equal.
4. **Logistic-regression stacker** on the three component logits — one
   intercept, three coefficients, the highest-capacity and
   highest-overfit-risk option of the four.

### Decision — pre-registered gates (set before results are seen)

- **Gate A — improvement:** ensemble holdout log loss must beat the
  **best single model's own holdout log loss** by ≥ **0.002**.
  (Sized to the holdout's own baseline, not to a number from a
  different split — the process lesson ADR-017's Gate A miscalibration
  left behind.)
- **Gate B — simplicity tiebreak:** the stacker (option 4) must beat
  the best fixed-weight blend by ≥ **0.002** to be preferred; ties go
  to the simpler blend.
- **Gate C — symmetry (ADR-004):** blend's `max_pair_deviation` on the
  holdout must be ≤ **1.5×** the best single model's own deviation on
  the same holdout. Averaging preserves symmetry if every input does;
  a fitted stacker with a non-zero intercept does not, by construction
  — a stacker that fails this gate gets refit without an intercept,
  not waved through.
- **Gate D — calibration guard:** holdout ECE must not worsen by more
  than **0.005** versus the best single model's holdout ECE.

**Accuracy is explicitly not a gate.** Per `docs/PLAN.md` §0.1's
metric hierarchy (reconfirmed by ADR-016), accuracy is reported
alongside the gated metrics as a parity check only — a candidate is
never selected or rejected on accuracy alone.

If no candidate clears all four gates, **v1 ships as the tuned
LightGBM alone**, unchanged from Wednesday, and this becomes this
week's fourth pre-registered negative result (after ADR-014's initial
Glicko gate, ADR-016's Tier 3 cut, and ADR-017's calibration
rejection).

### Why

Same discipline as ADR-016/017: thresholds fixed before any number is
seen, so a marginal result can't be rationalized into a win after the
fact. Gate C exists because nothing about blending guarantees the
project's ADR-004 symmetry invariant — it has to be checked, not
assumed. Gate B exists because the stacker is the one candidate with
real capacity to overfit three inputs on a modest OOF set; it has to
earn its extra complexity the same way isotonic was asked to in
ADR-017, not get preferred by default for being fancier.

### Results

*TBD — filled in after Step 4 (blend scoring) and Step 6 (gate
application).*

### Consequences

*TBD.* Known in advance regardless of outcome: a winning blend adds
real operational cost to Week 4's inference path (three artifacts
loaded, three predict calls per upcoming bout instead of one), and
would need its own, separately re-registered calibration gates before
any recalibration attempt — today's ADR-017 thresholds were fit to
LightGBM-alone's error population and don't transfer as-is.

---
## [ADR-017] Calibration: both methods rejected — v1 ships uncalibrated

**Date:** 2026-08-21
**Status:** Accepted

### Context

Monday's tuning improved log loss but regressed ECE (0.0214 → 0.0318,
odds-covered). `docs/PLAN.md` §3 calls for isotonic/Platt calibration
to fix this. Deviation from the plan's literal wording: the calibrator
is fit on **out-of-fold predictions from train** (12 expanding-window
folds), not on val directly — fitting on val and grading on val would
be circular, same trap `tune_lightgbm.py` avoids for hyperparameters.
Val is read once, after the method is chosen.

### Method — three pre-registered gates

- **Gate A:** must cut holdout ECE by ≥0.005, log loss may not rise >0.002.
- **Gate B:** isotonic must beat Platt's log loss by ≥0.002 to be preferred.
- **Gate C (added mid-session):** symmetrized bout pairs must sum to
  ~1.0 (ADR-004). First version used an absolute bar (1e-6), which
  turned out unfair — the raw model's own pair-sum deviation is
  0.070, not near-zero (LightGBM has no structural symmetry
  guarantee). Revised to a **ratio vs. the raw baseline** (≤1.5×)
  before re-testing.

### Results (train-internal holdout, OOF folds 2011–20 fit / 2021–22 held out)

| method | d_log_loss | d_ece | max_pair_dev (vs. baseline) |
|---|---|---|---|
| platt | +0.000039 | −0.0022 | 0.97× — passes Gate C |
| isotonic | +0.001898 | −0.0076 | **1.86× — fails Gate C** |

**Isotonic disqualified on Gate C** — its step function amplifies the
raw model's existing corner-pair asymmetry into a real ADR-004
violation, confirmed again on val (pair-sum deviation 0.125) and
bin-count sensitivity (beats raw ECE at 5 bins, loses at 10/15/20 — a
step-function artifact, not a robust improvement).

**Platt passes Gate C, fails Gate A** — real, correctly-signed effect
(consistent with the raw model's reliability curve showing genuine
*underconfidence*, not overconfidence), but too small to clear the bar.

**No method clears both gates → `chosen = None`.**

### Decision

**Ship v1 uncalibrated.** Week 3 Saturday's frozen artifact is the
tuned LightGBM alone.

### Why

Two distinct, non-overlapping failures — not one blanket "didn't
work." Isotonic fails structurally (incompatible with the symmetrized
dual-row design). Platt fails on magnitude, not mechanism — its
−0.0022 sits inside the noise range this project treats as
insignificant elsewhere (ADR-014/016's floors). Underlying both: raw
full-val ECE (0.0235) was already comfortably under the ≤0.05 target,
so there wasn't much of a real problem to fix.

**Caveat on the gate itself, not grounds to reopen it:**
`MIN_ECE_IMPROVEMENT=0.005` was sized to val's 0.0318 regression, but
the holdout's own baseline ECE is 0.0133 — under half that. Platt was
held to a bar built for a different population. Noted for next time,
not corrected retroactively.

Bucketed ECE re-check (`calibration_buckets.py`) against the shipped
raw model reproduces ADR-015 exactly — no drift, no new evidence on
the Glicko-2 gate.

### Consequences

**Easier:** Week 4 inference stays single-artifact — no calibrator to
load or sequence.

**Standing tool:** `models/calibration.py`'s gate/OOF structure is
reusable for Thursday's ensemble, but needs its own OOF generator and
its own re-registered thresholds — don't reuse today's numbers.

**Retained for later:** the raw model's underconfidence direction is
real and corroborated (reliability diagram + Platt's correctly-signed
holdout number) — logged to `IDEAS.md` for a revisit at test unlock,
starting from Platt or beta calibration, not isotonic.

**Note for the record:** all project ECE is reported at n_bins=10;
today's sensitivity check showed it varies ~2× by bin count alone.

**Forecloses, for now:** isotonic for this specific symmetrized
pipeline — not permanently, only until the step-function/symmetry
interaction is addressed directly.

---
## [ADR-016] Week 3 Tuesday Tier 3 features: all four groups cut — no measured contribution on either metric

**Date:** 2026-08-20
**Status:** Accepted

### Context

`docs/PLAN.md` §3's Week 3 Tuesday entry calls for Tier 3 features
("style clustering, SoS, short-notice, layoff interactions") with the
explicit instruction to "measure each addition's delta." Four groups
were built and wired into the training matrix, taking it from 32 to
38 `diff_` features:

| group | columns | rationale |
|---|---|---|
| `sos` | `diff_sos_last_3`, `diff_sos_last_5` | Elo says a fighter is 1650; it doesn't say whether they got there in deep water or against padding. `docs/PLAN.md` §2's headline Tier 3 item. |
| `damage` | `diff_recent_damage_24mo` | Trailing-24-month significant strikes absorbed. Distinct from the existing career-cumulative Tier 2 stat, which can't separate "took damage early, untouchable since" from "just survived three wars." |
| `interactions` | `diff_layoff_x_age`, `diff_age_x_experience` | Trees don't automatically find multiplicative relationships. A 38-year-old off 18 months ≠ a 26-year-old off 18 months. |
| `weight_change` | `diff_weight_class_change` | -1/0/+1 division move since last bout. |

Two planned items were not built: **short-notice** (no bout
announcement date exists in the schema or in either data source —
see `IDEAS.md`) and **style clustering** (designated stretch, cut for
time after the ADR-015 permutation work).

### Method

`models/feature_deltas.py`. Every number comes from
`models/cv.py`'s expanding-window folds carved out of **train only** —
val was not read at any point during feature selection. Adding a
feature, scoring val, and keeping it if val improved would be feature
selection against val, which destroys val's honesty exactly as surely
as hyperparameter tuning against it would (the trap
`models/tune_lightgbm.py` was built to avoid, one level up).

Hyperparameters held fixed at Monday's Optuna winner across every
configuration. Those params were found on the 32-feature set, so they
mildly favor the baseline — the conservative direction, making new
features work harder to prove themselves.

**Pre-registered thresholds, set before results were seen:**
- Log loss: |Δ| ≥ **0.002** to count. Below that is noise at this
  fold structure — the same stopping rule ADR-014 used for Elo
  K-factor tuning.
- Accuracy: |Δ| ≥ **0.005**. Accuracy discards confidence
  information and is noisier; at val's n≈2,034 and p≈0.62, one
  standard error is ~1.1 points, so half a point is the floor for a
  correlated-fold CV average.

Both a cumulative walk (add groups one at a time) and a leave-one-out
ablation (drop each group from the full 38-feature model) were run.
Both were needed: a group can look useless when added because another
group already covers the same ground, yet still be the one carrying
signal when both are present.

### Results

**Baseline reproduced Monday's tuned CV log loss to six decimals
(0.660179).** That is a free end-to-end regression test on the Elo
attach, symmetrization, CV splitter, `to_differential` build, and
tuned-params loading — nothing drifted while six features were added.

Cumulative:

| config | n_feat | CV log loss | CV accuracy | Δ log loss vs. baseline | Δ accuracy vs. baseline |
|---|---|---|---|---|---|
| baseline (Monday) | 32 | 0.660179 | 0.606076 | 0.000000 | 0.000000 |
| + sos | 34 | 0.660236 | 0.604414 | +0.000057 | -0.001662 |
| + damage | 35 | 0.659999 | 0.605032 | -0.000180 | -0.001044 |
| + interactions | 37 | 0.659913 | 0.607627 | -0.000266 | +0.001551 |
| + weight_change | 38 | 0.660239 | 0.602612 | +0.000060 | -0.003464 |

Leave-one-out from the full 38-feature model (positive Δ = removing
it made things worse = the group was contributing):

| dropped | CV log loss | CV accuracy | Δ log loss | Δ accuracy |
|---|---|---|---|---|
| (none — full) | 0.660239 | 0.602612 | 0.000000 | 0.000000 |
| sos | 0.661142 | 0.605147 | +0.000902 | +0.002535 |
| damage | 0.660101 | 0.607818 | -0.000138 | +0.005206 |
| interactions | 0.660130 | 0.606500 | -0.000109 | +0.003888 |
| weight_change | 0.659913 | 0.607627 | -0.000327 | +0.005015 |
| `diff_sos_last_3` only | 0.659646 | 0.605996 | -0.000594 | +0.003385 |
| `diff_sos_last_5` only | 0.660468 | 0.606017 | +0.000229 | +0.003405 |

**Every log-loss effect in the entire run falls between -0.0006 and
+0.0009 — the largest is under half the 0.002 threshold.** Six
features, three new query paths, and the full 38-feature model landed
0.00006 *worse* than the 32-feature baseline.

### Decision

**All four groups cut from the model.** The three
`build_train_val_with_elo` flags (`include_sos`, `include_damage`,
`include_weight`) default to `False`.

**Code retained, not deleted.** `features/tier3.py` and
`tests/test_tier3.py` stay in the repo, tested and correct. What's
absent is the *evidence for inclusion*, not the correctness of the
implementation — and the flags make re-testing a one-argument change
if the training population later changes (see Consequences).

**No val read was performed for today's work.** With every feature
cut, the val configuration is Monday's exact 32-feature model, and
val's numbers are already in `docs/RESULTS.md`. Spending one of val's
looks to confirm that an unchanged model produces an unchanged number
would buy zero information.

### Why

The pre-registered thresholds did their job. Without them, several of
these numbers are small enough to rationalize in either direction —
`+ interactions` at -0.000266 could be written up as "a modest
improvement" by someone motivated to keep it.

**On the accuracy question specifically:** accuracy was added to this
harness mid-session, after log loss came back flat, on the reasoning
that a feature could push near-coinflip fights across the 0.50 line
without moving log loss much — a real mechanism worth testing rather
than assuming. The measured answer was not just "flat" but mildly
unfavorable: **all four leave-one-out removals improved accuracy**
(+0.0025 to +0.0052), and the cumulative walk lost 0.35 points from
baseline to full. Those magnitudes still sit at or inside the 0.005
accuracy noise floor, so the honest read is "no evidence of benefit,
with unfavorable drift," not "these features actively hurt." Four out
of four pointing the same direction is more suggestive than any single
number, but not a claim worth staking.

Because both metrics agree, no revision to `docs/PLAN.md` §0.1's
metric hierarchy (log loss as the headline, accuracy as parity check)
was needed — the question of whether to promote accuracy over log
loss was closed by data rather than by argument.

**Most likely mechanism: redundancy.** SoS correlates with Elo by
construction — beating strong opponents raises your own rating, so
`diff_elo_pre` already carries much of what SoS was meant to add.
`recent_damage_24mo` overlaps the existing career-cumulative absorbed
stat. The interactions are relationships trees can approximate through
repeated splits on `diff_age` and `diff_days_since_last_fight`, which
are already features #1 and #10 by gain. `weight_class_change` is 75%
zeros with only 7.4% non-zero rows.

### Secondary finding: `diff_sos_last_3` is the weaker of the two windows

Dropping *only* `diff_sos_last_3` produced the best log loss of any
configuration tested (0.659646, -0.000594 vs. full), while dropping
only `diff_sos_last_5` made things worse (+0.000229). Both are inside
the noise threshold and neither is actionable alone, but the direction
is mechanically sensible: two highly correlated columns measuring the
same thing, with the noisier 3-fight window diluting the 5-fight one.

**If SoS is ever revisited, the starting configuration should be
`n=5` alone, not both windows.** Recorded here so that isn't
re-derived from scratch later.

### Consequences

**Easier:** Week 4's FastAPI inference path stays at 32 features with
no additional live Postgres query paths. `recent_damage_absorbed` and
`weight_class_change` are per-fighter DB round-trips that would have
had to run for every upcoming bout at inference time — real
operational cost for measurably zero predictive benefit.

**Retained for cheap retest:** all three flags flip to `True` with one
argument. The most likely retest is the modern-only training window in
`IDEAS.md` — these features may behave differently on a 2010+
population where the sport is more professionalized and the era-drift
effects from Saturday's KS-test findings are absent.

**Standing tool:** `models/feature_deltas.py` is reusable for any
future feature-set question. Its `FEATURE_GROUPS` dict is the only
thing that needs editing to test a new group, and the pre-registration
discipline it encodes — thresholds written before results are seen,
both cumulative and leave-one-out run together — applies beyond
today's four groups.

**Not foreclosed:** style clustering was never built (cut for time),
so this ADR says nothing about it. Short-notice remains blocked on
data availability, not on evidence.

**Realistic framing for the rest of Week 3:** two more chances remain
to close the gap to the market baseline (0.5897 log loss) before
Friday's one-shot test unlock — Wednesday's calibration and Thursday's
ensemble. Today's result, combined with the fold-trend saturation
finding in `docs/RESULTS.md`, suggests the ensemble (combining
genuinely different model *shapes*) is the more promising of the two,
since adding more features of the same *kind* moved nothing.

---
## [ADR-015] Glicko-2 RD: evidence gate closed — bucketed ECE differences are sampling noise

**Date:** 2026-08-20
**Status:** Accepted — resolves the deferral in ADR-014

### Context

ADR-014 gated Glicko-2's rating deviation (RD) on a measured
calibration gap: bucket the tuned LightGBM's val ECE by
`total_ufc_fights` and `days_since_last_fight`, build RD only if
low-fight-count or long-layoff fighters are measurably worse
calibrated than the model overall. `models/calibration_buckets.py`
implements that check with a pre-registered rule (bucket ECE ≥ 2x
full-val ECE of 0.0235, n ≥ 150, in an ADR-named population).

### What the rule said

Gate OPEN. Three triggers: debut (0.0682, 2.90x baseline), 365-730d
layoff (0.0677, 2.88x baseline), and `no prior fight` — the last
being the same 228 rows as the debut bucket viewed through the other
bucketing dimension (a debutant has both `total_ufc_fights = 0` and
`days_since_last_fight = NaN`), so two distinct populations
triggered, not three.

### Why that verdict was discarded

The pre-registered threshold compared small-bucket ECE against a
full-val ECE computed on 2,034 rows. ECE is biased upward at small
n — with fewer rows per confidence bin, each bin's actual-rate
estimate is noisier, inflating the predicted-vs-actual gap even for a
perfectly calibrated model. Comparing a 173-row bucket's ECE directly
against a 2,034-row baseline compares unlike quantities.

A permutation test (`check_buckets_against_null` in
`models/calibration_buckets.py`, 2,000 size-matched random draws from
the full val set, seed=42) established each triggered bucket's null
distribution — what ECE looks like for a random group of that exact
size, with no real effect present:

| bucket | n | observed ECE | null mean ECE | percentile | p-value |
|---|---|---|---|---|---|
| 0 (debut) | 228 | 0.0682 | 0.0590 | 70.7 | 0.29 |
| 365-730d | 173 | 0.0677 | 0.0673 | 53.4 | 0.47 |
| 730d+ | 42 | 0.1872 | 0.1349 | 87.3 | 0.13 |
| 21+ | 116 | 0.0610 | 0.0825 | 21.4 | 0.79 |

No bucket reaches p < 0.05. **365-730d — the strongest apparent
finding, the one with n ≥ 150, and the only bucket RD could
mechanistically address (RD varies continuously with time since last
fight, unlike a debutant's constant initial RD) — landed at the 53rd
percentile of its own null.** It is indistinguishable from a random
draw of that size.

### Decision

Gate **CLOSED**. Glicko-2 RD is not built. Deferred to `IDEAS.md`
with these numbers attached, for revisit once the test set unlocks
and larger samples (especially 730d+, currently n=42) are available.

### Why

The pre-registered rule's own comparison was flawed: it measured
whether a bucket was *small*, not whether it was *miscalibrated*,
because ECE's small-sample bias was never accounted for in the
threshold. The permutation check corrects for this by testing each
bucket against its own size-matched null rather than against a
whole-dataset aggregate, and under that correction, every triggered
bucket collapses to statistical noise.

This is consistent with the debunking pattern already established
elsewhere in the project (the naive-corner-baseline investigation in
`LEAKAGE_LOG.md`): a number looked meaningful, was investigated
before being acted on, and turned out to be explainable by a known,
non-leak mechanism. The same discipline — verify before building —
applies to feature-justifying evidence, not just leakage suspicions.

### Secondary finding, also not acted on

The 21+ fight bucket (n=116, below the n ≥ 150 interpretability
threshold) showed `mean_pred` 0.4543 vs. `actual_rate` 0.4138,
suggesting the model may under-discount high-mileage veterans. At the
21st percentile of its own null (p=0.79), this is not evidence of a
real effect and should not be over-read. Logged to `IDEAS.md` as an
age-nonlinearity / `age x total_ufc_fights` mileage-interaction
question to revisit at test unlock — explicitly *not* as an
RD/Glicko question, since RD is characteristically **low** for
actively-fighting veterans and would reinforce trust in their rating
rather than discount it, the opposite of the effect this bucket would
need explained.

### Consequences

**Easier:** Week 3 stays on schedule — no half-day Glicko-2 build
(rating periods, iterative volatility solve, hand-verified tests)
displacing today's remaining Tier 3 work (strength of schedule,
contextual features, CV-based delta measurement).

**Adds a standing tool:** `check_buckets_against_null` in
`models/calibration_buckets.py` is reusable for any future subgroup-
metric comparison in this project. The guardrail it encodes — compare
a bucket against a size-matched permutation null, never against a
whole-dataset aggregate — applies beyond this one check.

**Forecloses, for now:** RD as a standalone feature. Not permanently —
if a future recheck (larger val/test population, or a differently-
constructed bucket) shows a real, permutation-surviving gap, this ADR
should be superseded, not edited.

**Requires care going forward:** any calibration-gap claim in
`docs/RESULTS.md`, `docs/MODEL_CARD.md`, or a future README should be
read against this ADR — the raw bucketed-ECE table alone
(`models/calibration_buckets.py`'s primary output) is not sufficient
evidence of a real subgroup effect without the permutation check
behind it.

---
## [ADR-014] Elo rating system: experience-based K-factor over constant K, Glicko-2 deferred to Week 3

**Date:** 2026-08-14
**Status:** Accepted

### Context

Per `docs/PLAN.md` §3, Week 2 Friday's deliverable is Elo/Glicko
implementation plus K-factor and weight-class-prior tuning.
`features/elo.py`'s `compute_elo_ratings` was built as a global,
weight-class-blind, sequential rating system — every fighter starts
at 1500, ratings update fight-by-fight in strict chronological
order, and each bout's pre-fight ratings are recorded before any
update touches them (the leakage-safety mechanism: a fight's own
result can never influence its own prediction).

Three real design questions had to be settled before this could be
evaluated as a baseline: Elo vs. Glicko-2, weight-class handling,
and constant vs. experience-based K. None were assumed going in.

### Options considered

**Elo vs. Glicko-2.** Glicko-2 models rating uncertainty (a rating
deviation, RD, alongside skill) and volatility — theoretically
well-suited to a sport with irregular activity and layoffs, but
structurally more complex to implement (rating periods, an iterative
volatility solve, not closed-form), and there was no measured
evidence yet that plain Elo's calibration was actually failing in a
way Glicko would fix. Plain Elo was simpler and directly testable
through the same `models/metrics.evaluate()` harness every other
baseline uses.

**Weight class.** Global rating per fighter vs. a separate rating
per weight class. The latter is more correct in principle, but adds
real complexity for fighters who move classes, with no evidence yet
that the global version is actually mispriced by weight-class
mixing.

**K-factor.** Constant K for every fighter/fight (simplest, one
number to tune) vs. experience-based K (higher for low-fight-count
fighters, who carry more uncertainty; lower for established
fighters, where a single result is mostly noise). Two decay shapes
considered for the latter: a step function (flat K_new for the first
N fights, then flat K_veteran — simple, precedented in chess
federations) vs. smooth decay (K decreases continuously with fight
count, no hard cutoff).

Started with constant K deliberately — "start constant, revisit if
tuning says it's not enough" — the same instinct as ADR-001's
LightGBM-over-neural-net reasoning: don't reach for the more complex
tool until the simpler one is shown to fall short.

### Decision

- **Plain Elo**, not Glicko-2, for now. Glicko-2 is deferred to
  Week 3, not dropped — see "Deferred: Glicko-2 in Week 3" below.
- **Global rating**, weight class ignored. Flagged as a long-term
  revisit (`docs/PLAN.md` §2 already lists a weight-class-adjusted
  variant as a real Tier 3 idea), not a permanent decision.
- **Experience-based K, smooth decay:** K = k_veteran + (k_new - k_veteran) * e^(-fight_count / decay_scale)
Final tuned parameters: **k_new=80, k_veteran=24, decay_scale=3.**

### Why

**The constant-K sweep is what forced the move to experience-based
K.** A grid over K ∈ {8, 16, 24, 32, 40, 48, 64, 80, 96, 128} showed
log loss and Brier improving nearly monotonically through K=80–96
before reversing by K=128, while ECE bottomed out cleanly at K=32
and then degraded continuously — 11.7x worse by K=64, and by K=128
it *breached* the plan's own ≤0.05 ECE target (§0.1) on the
odds-covered subset (0.0553). Log loss wanting a high K while ECE
wanted a low K, with no single value serving both, was itself
evidence that one global K is the wrong shape for the population.

This was corroborated independently, before any tuning happened, by
hand-verifying two real fighters under constant K: Islam Makhachev
(16 fights, ~13.5 rating points gained per fight on average) vs.
Neil Magny (33 fights, ~5.9 points/fight average) — Magny's rating
closed to within ~50 points of Makhachev's peak almost entirely
through fight *volume* compounding a smaller average return, not
comparable win quality. A textbook case for K that decays with
experience rather than treating a prolific journeyman's 26th fight
the same as a rising contender's 3rd.

**Smooth decay over a step function:** a more realistic "cooling"
shape with no cliff at an arbitrary fight-count threshold, at the
cost of one more parameter (`decay_scale`) to tune — accepted since
the tuning infrastructure already existed and reusing it was cheap.

**Final parameters, two grid rounds, 60 combinations total.** Round
1 found log-loss and ECE optima disagreeing sharply (best log loss
at k_new=96, best ECE at k_new=64), with k_new=96 costing 7–10x the
calibration for the last ~0.001–0.0013 of log-loss gain past
k_new=80 — the same cliff shape as the constant-K K=128 case, one
level down. Round 2 narrowed to k_new∈{64,72,80}, k_veteran∈{16,24,32},
decay_scale∈{1,2,3,5} — decay_scale=1 was worse on both metrics than
2/3/5 (didn't place in either top-10, closing the "extend downward"
question), and `(80, 24, 3)` emerged **Pareto-optimal**: nothing
tested beat it on both log loss and ECE at once. Its log loss
(0.6788, odds-covered) beats flat K=32's (0.6864) by 0.0076 — a real
gain — at a small calibration cost (ECE 0.0041 vs. flat K=32's
0.0023, both comfortably inside target).

Tuning stopped after two rounds: remaining differences between top
candidates were ~0.001–0.002, close to noise, and further narrowing
risked tuning to this specific val window rather than finding real
signal — the same "any 3+ point jump is a leak until proven
otherwise" discipline from §0.2, applied to hyperparameters instead
of features.

### Consequences

**Easier:** `compute_elo_ratings` accepts either a constant float or
a callable `k_factor` — the constant path is untouched (every
existing test and the Step 5 constant-K script still work
unmodified), the experience-based path is additive. `elo_baseline()`
reuses `k_factor_by_experience`'s defaults, so a future re-tune only
means editing default arguments, not call sites.

**Standalone Elo is the weakest of the three Week 2 baselines on
accuracy/log loss (56.1% / 0.6788, odds-covered) — expected, not a
regression.** One accumulated rating vs. LR's ~30 stat-differential
features, or the market, was never going to win alone. Elo's real
test is as one input among several in Week 3's LightGBM model. Its
best ECE of the day (0.0041) should be read with the same skepticism
already applied to LR's ECE in `docs/RESULTS.md` — a low-signal
model hedging toward 50/50 has less room to be badly wrong, which
isn't the same as being sharp.

**Requires care going forward:** different fighters in the same bout
can now carry different K's, so the system is no longer perfectly
zero-sum in aggregate (a debutant losing to a veteran can lose more
points than the veteran gains) — an accepted, known property of
experience-adjusted Elo, worth remembering if a future check assumes
strict zero-sum behavior the way the original constant-K version had.

**Forecloses, for now:** weight-class-adjusted Elo (long-term per
§2) and a step-function K-factor (smooth decay was chosen; only
revisit if the smooth version shows a specific failure a hard cutoff
would fix).

---

### Deferred: Glicko-2 in Week 3

Not a "maybe someday" note — a specific, evidence-gated plan, written
down now so it survives past today.

**What Glicko-2 actually adds that Elo can't:** not a better skill
estimate — its rating deviation (RD) is the piece with no Elo
equivalent. RD tracks *how much is actually known* about a fighter's
true skill right now, growing during inactivity and shrinking with
consistent activity. The promising use isn't replacing Elo's rating
with Glicko's; it's adding **RD as its own standalone feature**
(`self_rd`, `opp_rd`, or their difference) alongside the existing Elo
rating — a genuinely non-redundant signal, not two versions of the
same number.

**The trigger, not immediate action:** once Week 3's Tier 3 features
are built and LightGBM is trained, bucket ECE by `total_ufc_fights`
and by `days_since_last_fight` (both already-built Tier 2 features).
If the model is measurably worse-calibrated for low-fight-count
fighters or fighters returning from a long layoff — exactly the
populations Elo's rating can't distinguish from a well-established,
active fighter sitting at the same number — that's the concrete,
data-backed signal to build Glicko-2's RD as a Week 3 ablation. If
that bucketed check doesn't show a real gap, Glicko-2 stays a
documented idea in `IDEAS.md`, not a rebuild.

**Why this is the right order:** it's the same discipline this
entire ADR is built on — plain Elo before Glicko, constant K before
experience-based, and RD-as-a-feature only once there's measured
evidence of a gap, not because it sounds more sophisticated going
in. Today's K-factor journey is the case study: every piece of added
complexity (experience-based K, the decay shape, the final
parameters) was earned by a measured finding, not intuition — Glicko
should clear the same bar before it gets built.

---
[ADR-013] Greco↔Wikipedia event/bout reconciliation — claim existing rows by writing Greco's source_url onto them

Date: 2026-08-17
Status: Accepted
Supersedes: The "not yet built" follow-up flagged in ADR-011's Consequences section.

Context

ADR-011 anticipated that Greco ingesting a real result for an event that already has a Wikipedia-sourced row (source_url IS NULL, wikipedia_pageid set) would need a reconciliation step, and left it unbuilt. That gap surfaced for real on UFC 330: Greco's ingest ran, found no events row with a matching source_url (because the existing row's source_url was still NULL), and inserted a second events row instead of updating the first. This cascaded — the new event got its own set of bouts rows, so the same card existed twice, with bout_stats and results attached only to the new, duplicate side. odds_snapshots collected pre-fight remained attached to the original (now orphaned-in-spirit) bout rows.

This was diagnosed and manually repaired via a one-off script (scripts/one_off/2026-08-17_merge_duplicate_event_ufc330.sql): the surviving row kept the Greco-sourced event's id (since it already owned bout_stats), with wikipedia_pageid/venue/location grafted onto it and all child rows (odds_snapshots, predictions) repointed before the duplicate was deleted.

The one-off fix and the systemic fix pull in different directions, which is the part worth writing down clearly:

The one-off fix kept the Greco row's id, because by the time it was discovered, the Greco row already owned bout_stats and repointing those would have been more invasive than repointing the handful of odds_snapshots rows the other way.
The systemic fix (this ADR) keeps the Wikipedia row's id going forward. Once this is running, Greco will never again get a multi-day head start on a completed event before reconciliation happens — so the Wikipedia row will always still be the one predictions/odds already point at, and identity should flow toward it rather than away from it.
Options considered
Do nothing; treat this as a one-time fluke. Rejected — it isn't a fluke, it's the direct, guaranteed consequence of two pipelines writing into the same tables keyed on two different columns, exactly as ADR-011 predicted. It will recur on every future card that goes: Wikipedia-sourced upcoming event → event happens → Greco ingests the result, unless something bridges the two keys before Greco's upsert runs.
Change loaders.py's upsert to match on (event_date, fuzzy name) instead of source_url when source_url match fails. Would work, but conflates two different jobs in one query — "find the row to update" is currently a simple, fast, unambiguous equality lookup; adding fuzzy fallback logic directly into the upsert path makes every future Greco load pay a fuzzy-matching cost and makes the upsert function harder to reason about and test in isolation.
A separate reconciliation pre-pass (data/ingestion/reconciliation.py) that runs immediately before each upsert, finds any unreconciled Wikipedia-sourced row for the same real card, and writes Greco's source_url onto it — so the existing, unmodified upsert logic in loaders.py then finds a matching source_url and updates in place instead of inserting.
Decision

Option 3. Two functions, claim_existing_events_for_greco and claim_existing_bouts_for_greco, called from load_events and load_bouts respectively, immediately before each function's existing upsert call.

Events are matched on event date (±1 day, to absorb UTC/local-date boundary cases for cards outside the US) plus a name-compatibility check: for numbered cards ("UFC 330"), the number must match exactly — a plain fuzzy score was rejected here specifically because it's unreliable when one name is a strict prefix of the other, which is exactly the UFC 330 case that broke. For Fight Nights (no number on either side), falls back to token_set_ratio fuzzy matching at a low threshold, since the date-plus-numbered-event check already does most of the filtering and this is only a last-resort tripwire.
Bouts are matched on (event_id, unordered fighter pair) rather than (event_id, fighter_red_id, fighter_blue_id) in order, since Wikipedia and Greco don't always agree on which fighter is in which corner.
Both are additive and non-destructive: they only ever touch rows where source_url IS NULL, so a second run finds nothing left to claim (idempotent by construction), and ambiguous matches (one Wikipedia row plausibly matching more than one incoming Greco event) are left alone and logged rather than guessed — check_duplicate_bouts (new, alongside the existing check_duplicate_events) will then keep flagging the case until it's resolved by hand, which is the intended outcome: a visible unresolved duplicate over a silent wrong merge.
Why

This keeps the direction of the fix aligned with why wikipedia_pageid exists as a separate column in the first place (ADR-011): the two pipelines shouldn't have to agree on identity ahead of time, and neither should have to change what it considers "the" key. Reconciliation instead runs as a bridge step that translates one pipeline's identity onto the other's key, right before the existing upsert logic needs it — the upsert itself never has to know reconciliation happened.

The identity-direction rule this settles for the future: whichever source's row got there first keeps its id; the later source contributes data, not identity. In steady state that will always mean the Wikipedia-sourced row survives, because Wikipedia ingestion runs continuously against the upcoming schedule while Greco only picks up a card after it's over — so Wikipedia will structurally always have the head start. The one-off UFC 330 fix went the other way only because reconciliation didn't exist yet when that gap opened, letting Greco's row accumulate bout_stats before anyone noticed there were two rows to merge.

Consequences

Easier: Future Greco ingests of a previously Wikipedia-sourced card update the existing row in place — no duplicate events/bouts rows, no orphaned odds_snapshots/predictions, no manual SQL merge required. check_duplicate_bouts gives a standing, automated signal if this ever fails silently (e.g. a stub-fighter row on one side never got reconciled to the real fighter, so the unordered-pair match misses).

Requires care going forward: if a fighter's corner flips between the Wikipedia-sourced pre-fight listing and Greco's post-fight result (rare, but possible when a late-notice replacement swaps who's "red"), claim_existing_bouts_for_greco logs a warning rather than silently accepting it, because it invalidates predicted_prob_red on any prediction already logged against that bout pre-fight. Settlement logic (Week 4) must key on predicted_winner_id, not corner position, for exactly this reason — that's now a hard requirement, not just a convenience.

Forecloses: Nothing schema-level. This is pure ingestion-time logic; no migration was needed, since it only changes what gets written into source_url before the existing upsert runs, not the shape of any table.

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