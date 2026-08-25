# Model Card — UFC Fight Outcome Predictor v1

**Version:** v1
**Status:** Frozen, shipped
**Frozen:** 2026-08-25
**Artifact:** `models/v1/model.txt`
**SHA-256:** `cf65091803b4ca8ed6f13fe39d6f76d3a30e5b5a25e2bc535fad9c10f37826b9`

---

## 1. Summary

A LightGBM binary classifier predicting the winner of a UFC bout from
pre-fight differential features (physical attributes, career stats,
Elo rating). Trained on fights through 2024, evaluated once against a
locked 2025+ test set.

**Headline result: the model is a validated, leakage-free pipeline
that falls short of a highly efficient betting market — on every
metric, on every slice, without exception.** This is the project's
actual finding, not a caveat to explain away. See §5.

---

## 2. Intended use

**Intended:** portfolio/research demonstration of an end-to-end ML
pipeline — data engineering, leakage-safe feature construction,
disciplined evaluation methodology, honest reporting of a negative
result.

**Explicitly NOT intended:**
- **Real-money betting of any kind.** See §6 — the Kelly simulation
  on this exact artifact shows a ~98% bankroll drawdown.
- Production decision-making of any consequence.
- A claim of predictive edge over sportsbook markets. It does not
  have one.

---

## 3. Model details

| | |
|---|---|
| Algorithm | LightGBM (gradient-boosted trees), binary objective |
| Training data | Symmetrized train+val, `event_date < 2025-01-01` (through 2024-12-31) |
| Rows / bouts | 15,242 rows / 7,621 bouts |
| Features | 32 differential (`diff_`) features, listed in full below |
| Calibration | **None.** Both Platt and isotonic calibration were tested and rejected (ADR-017) — see §5.2 |
| Hyperparameters | Tuned via Optuna, 60 trials, frozen (`models/artifacts/lgbm_best_params.json`) |
| Serialization | Native LightGBM text format (`Booster.save_model`), not pickle — see ADR-021 Decision 1 for why |

**Hyperparameters:**
```json
{
  "learning_rate": 0.011451042355352742,
  "n_estimators": 498,
  "num_leaves": 18,
  "max_depth": 5,
  "min_child_samples": 79,
  "colsample_bytree": 0.7295400927447112,
  "subsample": 0.6228874802663125,
  "subsample_freq": 2,
  "reg_alpha": 5.259505835392509e-07,
  "reg_lambda": 0.6626901884005618
}
```

**Feature list, in order** (order matters — inference must build
matrices with this exact column order):

```
diff_age, diff_height_cm, diff_reach_cm, diff_reach_to_height_ratio,
diff_slpm, diff_sapm, diff_td_avg_per_15, diff_sub_attempts_per_15,
diff_striking_accuracy, diff_takedown_accuracy,
diff_significant_strike_rate, diff_striking_defense,
diff_takedown_defense, diff_career_win_pct, diff_decision_win_pct,
diff_decision_loss_pct, diff_submission_win_count,
diff_submission_success_rate, diff_finish_rate, diff_ko_loss_rate,
diff_sub_loss_rate, diff_total_ufc_fights, diff_days_since_last_fight,
diff_avg_fight_time_seconds, diff_title_fight_experience,
diff_times_knocked_down, diff_knockdown_rate, diff_control_time_pct,
diff_time_controlled_pct, diff_striking_output_decay,
diff_takedown_output_decay, diff_elo_pre
```

**All four Tier 3 feature groups (style/SoS/damage/interaction/weight-
class-change) were tested and cut** — none cleared the pre-registered
0.002 log-loss-delta threshold (ADR-016). They do not appear above.

---

## 4. Provenance & reproducibility

| | |
|---|---|
| Freeze commit | `9bd9b28ef338cdda93e2939d0ace9b7ea773b028-DIRTY` |
| Test-eval commit | `55572472e38b9f1ae974c8ec37796769c7f3de68-DIRTY` |
| Library versions | lightgbm 4.7.0, numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0 |
| Determinism | **Bit-identical.** Refitting the model twice on identical data/params produces byte-for-byte identical LightGBM text dumps. Confirmed empirically (`models/freeze.py`), not assumed from a pinned seed alone. |

**Known provenance gap:** the test-day evaluation run
(`5557247-DIRTY`) had uncommitted changes in its working tree. The
immediately-following commit (`9c08a8c`) touched only documentation
and results files — no code — so it cannot be assumed to contain
whatever change was in flight. The specific diff active at test-day is
not recoverable from git history. This does not affect the validity of
the numbers below: the code as committed and reviewed for this freeze
already implements the documented train+val boundary correctly. It is
recorded as an open gap in exact historical reproducibility, not a
known error. See ADR-021 Decision 5.

---

## 5. Evaluation results

One-time test-set evaluation (2025-01-11 to 2026-08-08, 835 bouts),
protocol pre-registered in ADR-020. **This is the only test-set read
this model, or any variant of it, will ever receive.**

### 5.1 Headline: odds-covered slice (n=756 bouts) — model vs. market

| | model (v1) | market (de-vigged) |
|---|---|---|
| Accuracy | 0.6574 | 0.7011 |
| Log loss | 0.6333 | 0.5757 |
| Brier | 0.2211 | 0.1961 |
| ECE | 0.0511 | 0.0236 |

**The market beats the model on every metric.** 95% bootstrap CI on
model log loss: [0.6165, 0.6502] — bout-level resampling, doesn't
overlap the market's number.

### 5.2 Full test / close-fight slices

| slice | who | n bouts | accuracy | log loss | ECE |
|---|---|---|---|---|---|
| full (incl. odds-uncovered) | model | 835 | 0.6623 | 0.6318 | 0.0552 |
| close (market-defined 0.40–0.60) | model | 201 | 0.5622 | 0.6867 | 0.0430 |
| close | market | 201 | 0.5672 | 0.6819 | 0.0156 |

Close-fight n=402 rows (201 bouts) clears the 150-bout floor set in
ADR-020 for a real, not purely directional, read. Market still wins,
by a narrower margin than elsewhere — the model's best relative
showing, still a loss.

### 5.3 Pre-registered success criteria (ADR-020 Decision 5)

| Criterion | Target | Actual | Met? |
|---|---|---|---|
| Primary: test log loss within val ±0.01 | 0.6483 ± 0.01 | 0.6333 (drift −0.0150) | **No** — better than val, outside tolerance, flagged for a v2 composition check, not acted on |
| Secondary: ECE ≤ 0.05 | ≤ 0.05 | 0.0511 | **No** — narrowly missed |
| (Not a criterion) Beat market | — | −0.0576 log loss gap | No |

### 5.4 Calibration

Both Platt and isotonic scaling were tested against pre-registered
gates (ADR-017) and **both rejected** — for different reasons.
Isotonic amplified an existing corner-symmetry artifact past its
allowed threshold; Platt showed the correct sign but too small an
effect to clear the bar. **v1 ships uncalibrated.**

The raw model's reliability curve shows genuine **underconfidence** at
low predicted probabilities — sitting below the diagonal — the
opposite of the overconfidence a log-loss-optimized GBDT is typically
expected to show. See `docs/results/test_reliability_20260823_204901.csv`
for the underlying curve data; link the rendered plot here once
confirmed in the README (Week 3 exit criterion).

### 5.5 Feature importance — corrected framing

Physical stats (`diff_age`, `diff_reach_to_height_ratio`,
`diff_reach_cm`) substantially outperformed this project's original
planning-doc label of "Tier 1 — weak." This held consistently across
a logistic-regression ablation, the default LightGBM baseline, and
the tuned model's own gain importance. The original framing is
retired; physical stats are a real, consistent signal in this
feature set.

### 5.6 Ensemble (not shipped)

An LR + LightGBM + Elo blend was tested (ADR-019) and **missed its
pre-registered gate by 0.000015** — a real, consistently-signed
effect, too small to justify shipping. v1 remains LightGBM alone.

---

## 6. Betting / Kelly simulation — why real-money betting is gated

Quarter-Kelly sizing (5%-of-bankroll cap, chronological compounding),
run on this exact artifact's test predictions:

| edge threshold | ROI | 95% CI | Kelly outcome ($100 start) |
|---|---|---|---|
| 0.00 | −13.86% | [−23.8%, −3.8%] | $1.67 |
| 0.02 | −12.62% | [−22.7%, −1.9%] | $1.69 |
| 0.05 | −14.19% | [−25.4%, −2.0%] | $1.73 |

All three 95% CIs sit **entirely below zero** — a decisive negative
result, not a coin-flip-with-noise outcome. A ~97–98% bankroll
drawdown at every threshold tested.

**This is the concrete evidence behind the project's standing rule:
real-money betting stays off the table until 3+ months of logged,
out-of-sample prediction results exist.** A small, real calibration
gap (§5.4) compounds catastrophically once bet sizing assumes the
model's stated probabilities are exact — this simulation is the
receipt for why that assumption is dangerous here.

---

## 7. Limitations

- **Uncalibrated**, with a measurable ECE gap on test (0.0511 vs. a
  0.05 target).
- **Underconfident**, not overconfident, at low predicted
  probabilities — an unusual direction for a GBDT and worth
  investigating in any v2 (calibration retry logged to `IDEAS.md`,
  starting from Platt/beta, not isotonic).
- **Odds coverage is capped**, not a modeling limitation but a data
  one — The Odds API's historical MMA coverage begins June 2020,
  hard-limiting the addressable odds-covered population regardless of
  budget or effort.
- **Training volume appears saturated** (ADR-016) — additional
  history moves metrics only marginally; the diagnostic train-only
  artifact (≤2022) tracked the shipped artifact within ~0.005 log
  loss despite two fewer years of data.
- **No calibrator, no ensemble** — both explored, both rejected on
  pre-registered gates, not on a hunch.
- **Test-day code provenance has an unresolved gap** (§4) — bounded
  and disclosed, not silent.

---

## 8. Framing

The pitch for this project is **market efficiency, not market
beating**: a disciplined, leakage-audited pipeline (pre-registered
gates at every decision point, a test set read exactly once) that
lands short of a highly efficient sportsbook market. That is a
legitimate, pre-anticipated outcome under this project's own risk
register — not a result requiring damage control.
