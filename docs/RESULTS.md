Initial baseline check: 1,037 distinct val bouts, and 954 with odds

## Baseline Results — Validation Set (2023–2024)

| Baseline | n | Accuracy | Log Loss | Brier | ECE |
|---|---|---|---|---|---|
| Market (de-vigged closing line) | 1,870 | 0.6936 | 0.5897 | 0.2019 | 0.0360 |
| Logistic Regression (full val) | 2,034 | 0.5998 | 0.6653 | 0.2365 | 0.0212 |
| Logistic Regression (odds-covered subset) | 1,870 | 0.6043 | 0.6634 | 0.2356 | 0.0209 |
| Elo (experience-based K, full val) | 2,034 | 0.5610 | 0.6781 | 0.2426 | 0.0059 |
| Elo (experience-based K, odds-covered subset) | 1,870 | 0.5615 | 0.6788 | 0.2430 | 0.0041 |
| LightGBM Untuned (full val) | 2,034 | 0.6165 | 0.6589 | 0.2329 | 0.0191 |
| LightGBM Untuned (odds-covered subset) | 1,870 | 0.6246 | 0.6535 | 0.2306 | 0.0214 |
| LightGBM Tuned (full val) | 2,034 | 0.6224 | 0.6529 | 0.2304 | 0.0235 |
| LightGBM Tuned (odds-covered subset) | 1,870 | 0.6305 | 0.6483 | 0.2282 | 0.0318 |

Coverage: 1,870 / 2,034 val rows (91.9%) had at least one sportsbook
odds snapshot. Rows without coverage are excluded, not imputed.

**On LR's ECE vs. the market's ECE:** LR's ECE (0.0212) looks lower
than the market's (0.0360), but this isn't LR being better
calibrated than Vegas. LR has weaker signal and hedges most
predictions closer to 50%, so there's less room to be badly wrong
about a confident claim it rarely makes. A model that mostly says
"toss-up" will look "well calibrated" almost by default — it isn't
the same thing as being a sharp, useful forecaster. The market's
higher ECE alongside meaningfully better accuracy/log loss/Brier is
the real signal: it's making confident calls, and backing them up.

**On Elo's ECE vs. LR's and the market's:** Elo's ECE (0.0041,
odds-covered) is the lowest of any baseline built so far — lower
even than LR's. The same caveat applies, more strongly: Elo alone
carries less signal than LR (one accumulated rating vs. ~30
stat-differential features), so its probabilities likely hedge
closer to 50/50 more often, and a model that rarely makes a
confident call has less room to be badly wrong about one. Elo's
accuracy (56.1%) and log loss (0.6788) are the weakest of the three
baselines here — expected for a single Tier 3 signal evaluated in
isolation, not yet combined with Tier 1/2 stats or the other Tier 3
differentiators (SoS, style matchup, weight class). Elo's real test
is as an additional LightGBM input in Week 3, not as a standalone
predictor.

Ratings computed with experience-based K (smooth decay):
k_new=80, k_veteran=24, decay_scale=3 — full tuning process and
reasoning in ADR-014.

**On `diff_submission_success_rate`'s missingness:** this feature is
NaN for ~65% of rows in train — most fighters simply don't have
enough submission attempts in their prior-fight history to produce a
rate. That's expected given the sport (most UFC fighters aren't
attempting submissions every fight), not a bug. Practically, this
means `SimpleImputer(add_indicator=True)` is mostly working off the
*indicator* column here rather than the underlying rate, since real
values only exist for about a third of rows. This isn't a reason to
drop the feature — just a reason not to read a small or noisy
coefficient on it as "submission ability doesn't matter." It mostly
means "this stat mostly isn't there yet."

Tuned via Optuna, 60 trials, expanding-window CV (2011-2022 folds,
`models/cv.py`) on train only — val never touched during search.
Best CV log loss: 0.6602 (not directly comparable to the val numbers
above; CV folds average in early, thin-history windows that don't
reflect the full model's training population).

**On tuned ECE vs. untuned:** tuning optimized for log loss and
improved it (0.6535 -> 0.6483, odds-covered), but ECE moved the wrong
direction (0.0214 -> 0.0318) — still comfortably inside the ≤0.05
target, but a real, not noise-level, shift. Expected: log loss
rewards confident correct calls, and tuning found a model willing to
be more confident, at some cost to calibration. Wednesday's isotonic/
Platt calibration step exists specifically to correct exactly this.

---

## Week 3 Tuesday — Tier 3 feature evaluation (all cut)

Four feature groups were built and measured on expanding-window CV
folds inside `train` only (`models/feature_deltas.py`). Val was not
read during feature selection. Hyperparameters held fixed at Monday's
Optuna winner across every configuration. Full decision and reasoning:
**ADR-016**.

Pre-registered thresholds, set before results were seen: log loss
|Δ| ≥ 0.002, accuracy |Δ| ≥ 0.005.

**The baseline configuration reproduced Monday's tuned CV log loss to
six decimals (0.660179)** — an end-to-end regression check on the Elo
attach, symmetrization, CV splitter, and differential build.

### Cumulative additions

| config | n_feat | CV log loss | CV accuracy | Δ LL vs. baseline | Δ acc vs. baseline |
|---|---|---|---|---|---|
| baseline (Monday, 32 feat) | 32 | 0.660179 | 0.606076 | 0.000000 | 0.000000 |
| + sos | 34 | 0.660236 | 0.604414 | +0.000057 | -0.001662 |
| + damage | 35 | 0.659999 | 0.605032 | -0.000180 | -0.001044 |
| + interactions | 37 | 0.659913 | 0.607627 | -0.000266 | +0.001551 |
| + weight_change | 38 | 0.660239 | 0.602612 | +0.000060 | -0.003464 |

### Leave-one-out from the full 38-feature model

Positive Δ = removing it made things worse = the group was
contributing.

| dropped group | CV log loss | CV accuracy | Δ LL | Δ acc |
|---|---|---|---|---|
| (none — full 38 feat) | 0.660239 | 0.602612 | 0.000000 | 0.000000 |
| sos | 0.661142 | 0.605147 | +0.000902 | +0.002535 |
| damage | 0.660101 | 0.607818 | -0.000138 | +0.005206 |
| interactions | 0.660130 | 0.606500 | -0.000109 | +0.003888 |
| weight_change | 0.659913 | 0.607627 | -0.000327 | +0.005015 |
| `diff_sos_last_3` only | 0.659646 | 0.605996 | -0.000594 | +0.003385 |
| `diff_sos_last_5` only | 0.660468 | 0.606017 | +0.000229 | +0.003405 |

**Verdict: all four groups cut.** Every log-loss effect in the run
falls between -0.0006 and +0.0009 — the largest under half the 0.002
threshold. On accuracy, all four leave-one-out removals *improved*
the number (+0.0025 to +0.0052), and the cumulative walk lost 0.35
points from baseline to full; those magnitudes sit at or inside the
0.005 accuracy noise floor, so the honest read is "no evidence of
benefit, with unfavorable drift," not "actively harmful."

No val read was performed — with every feature cut, the val
configuration is Monday's unchanged 32-feature model, and its numbers
are already recorded above.

### Fold-by-fold trend — the model saturates on training volume

Baseline (32 feat), oldest to newest:

| val_year | train bouts | val bouts | log loss | accuracy |
|---|---|---|---|---|
| 2011 | 1,291 | 295 | 0.671664 | 0.572881 |
| 2012 | 1,586 | 333 | 0.664527 | 0.600601 |
| 2013 | 1,919 | 376 | 0.644137 | 0.646277 |
| 2014 | 2,295 | 494 | 0.664687 | 0.606275 |
| 2015 | 2,789 | 464 | 0.655693 | 0.602371 |
| 2016 | 3,253 | 483 | 0.649070 | 0.630435 |
| 2017 | 3,736 | 446 | 0.651452 | 0.631166 |
| 2018 | 4,182 | 469 | 0.657615 | 0.608742 |
| 2019 | 4,651 | 506 | 0.680333 | 0.565217 |
| 2020 | 5,157 | 444 | 0.656005 | 0.614865 |
| 2021 | 5,601 | 497 | 0.668792 | 0.578471 |
| 2022 | 6,098 | 506 | 0.658171 | 0.615613 |

`corr(val_year, log_loss) = 0.073` for the baseline, `0.064` for the
full 38-feature model — essentially zero, and marginally *positive*
(later folds slightly worse, not better).

**Training bouts grow ~4.7x across the fold range (1,291 → 6,098)
and buy nothing.** The model saturates on training volume well before
2022's fold. Two implications:

1. Supporting evidence for the modern-only training window idea in
   `IDEAS.md` — if the extra pre-2010 history isn't improving
   anything, dropping it costs less than assumed.
2. The binding constraint is not data volume. It's the feature set,
   or the ceiling of what pre-fight tabular stats can express about a
   fight outcome. Relevant framing for the README and
   `docs/MODEL_CARD.md`.

**2019 is the worst fold on both metrics in both configurations**
(0.6803 log loss, 0.5652 accuracy) and **2013 the best** (0.6441 /
0.6463) despite training on only 1,919 bouts. Not investigated —
most likely year-to-year variance in upset frequency rather than
anything structural, but worth a line in the model card as a known
artifact rather than leaving it to be rediscovered.

## Week 3 Wednesday — Calibration (rejected, v1 ships uncalibrated)

Both isotonic and Platt evaluated as post-hoc calibrators, fit on
out-of-fold train predictions (not val — see ADR-017). Neither passed
all pre-registered gates; **v1 ships uncalibrated**.

### Train-internal holdout (OOF fit: 2011–20, n=8,620 · holdout: 2021–22, n=2,006)

| method | accuracy | log_loss | brier | ece | max_pair_dev | d_log_loss | d_ece |
|---|---|---|---|---|---|---|---|
| uncalibrated | 0.5972 | 0.6634 | 0.2355 | 0.0133 | 0.0702 | — | — |
| platt | 0.5972 | 0.6635 | 0.2355 | 0.0111 | 0.0683 (0.97x) | +0.0000 | −0.0022 |
| isotonic | 0.5992 | 0.6653 | 0.2364 | 0.0057 | 0.1307 (**1.86x**) | +0.0019 | −0.0076 |

Isotonic disqualified on symmetry (Gate C, ADR-004). Platt passes
symmetry but falls short of the required ECE improvement (Gate A).

### Val (the one read — isotonic only, since it was the provisional pick before Gate C existed)

| | raw (shipped v1) | isotonic (rejected) | Δ |
|---|---|---|---|
| accuracy, full val | 0.6224 | 0.6244 | +0.0020 |
| log loss, full val | 0.6529 | 0.6539 | +0.0010 |
| ece, full val | 0.0235 | 0.0307 | +0.0072 |
| accuracy, odds-covered | 0.6305 | 0.6326 | +0.0021 |
| log loss, odds-covered | 0.6483 | 0.6492 | +0.0009 |
| ece, odds-covered | 0.0318 | 0.0352 | +0.0035 |

### ECE sensitivity to bin count (val)

| n_bins | raw, full val | isotonic, full val | raw, odds-covered | isotonic, odds-covered |
|---|---|---|---|---|
| 5 | 0.0150 | 0.0114 | 0.0209 | 0.0132 |
| 10 | 0.0235 | 0.0307 | 0.0318 | 0.0352 |
| 15 | 0.0291 | 0.0337 | 0.0354 | 0.0415 |
| 20 | 0.0253 | 0.0365 | 0.0330 | 0.0462 |

Isotonic only wins at n_bins=5; loses at every finer resolution, gap
widening — a step-function artifact, not a robust improvement. All
other ECE in this project is reported at n_bins=10 (`models/metrics.py`
default); this is the first time that choice has been shown to matter
this much (~2x swing on raw alone).

### Reliability diagrams

`docs/images/reliability_v1_full_val.png`,
`docs/images/reliability_v1_odds_covered.png`

Raw model's curve sits below the diagonal at low predicted probability
and above it past ~0.55 — the shape of **underconfidence** (real
outcomes more decisive than stated probabilities), not the
overconfidence a log-loss-optimized tune would suggest by default.
Platt's small, correctly-signed holdout effect is consistent with this.

Full reasoning, gate definitions, and the Gate C mid-session revision:
**ADR-017**.