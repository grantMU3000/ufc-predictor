Initial baseline check: 1,037 distinct val bouts, and 954 with odds

## Baseline Results — Validation Set (2023–2024)

| Baseline | n | Accuracy | Log Loss | Brier | ECE |
|---|---|---|---|---|---|
| Market (de-vigged closing line) | 1,870 | 0.6936 | 0.5897 | 0.2019 | 0.0360 |
| Logistic Regression (full val) | 2,034 | 0.5998 | 0.6653 | 0.2365 | 0.0212 |
| Logistic Regression (odds-covered subset) | 1,870 | 0.6043 | 0.6634 | 0.2356 | 0.0209 |
| Elo (experience-based K, full val) | 2,034 | 0.5610 | 0.6781 | 0.2426 | 0.0059 |
| Elo (experience-based K, odds-covered subset) | 1,870 | 0.5615 | 0.6788 | 0.2430 | 0.0041 |

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