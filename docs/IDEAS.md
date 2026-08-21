- **Modern-only training window (~2010+)**: hypothesis that 1999-2010
  era fights (thin per-year volume, pre-professionalization per Week 2
  Saturday's KS-test era-drift findings) may be diluting rather than
  helping. **Now has direct supporting evidence** beyond the original
  hypothesis: Week 3 Tuesday's fold-by-fold CV trend
  (`docs/RESULTS.md`, ADR-016) shows `corr(val_year, log_loss) = 0.073`
  — essentially zero — while training bouts grow 4.7x across the fold
  range (1,291 -> 6,098). The model saturates on training volume early,
  so the extra pre-2010 history is measurably not buying anything.
  Still requires rerunning the full baseline suite (Elo/LR/LightGBM)
  on the restricted population for a fair before/after — not a
  same-day bolt-on. Good candidate for buffer week or a documented
  ablation in the model card. **If this is tested, retest the ADR-016
  feature groups on the restricted population at the same time** —
  the three `build_train_val_with_elo` flags make that a
  one-argument change, and those features may behave differently
  without the era-drift effects present.
- **Glicko-2 rating deviation (RD) as a standalone feature**: ADR-014
  proposed it, ADR-015 closed the gate. The bucketed-ECE check fired
  (debut 0.0682, 365-730d layoff 0.0677, both ~2.9x the full-val
  baseline), but a size-matched permutation test showed every
  triggered bucket was indistinguishable from noise — 365-730d, the
  only bucket RD could mechanistically address, landed at the 53rd
  percentile of its own null (p=0.47). Revisit once the test set
  unlocks and larger samples are available, especially the 730d+
  bucket (currently n=42, 87th percentile, p=0.13 — the closest thing
  to a real signal, but badly underpowered). Rerun
  `check_buckets_against_null` rather than the raw bucketed table.
- **Age nonlinearity / `age x total_ufc_fights` mileage interaction**:
  ADR-015's 21+ fight bucket showed `mean_pred` 0.4543 vs.
  `actual_rate` 0.4138 — the model may under-discount high-mileage
  veterans. At n=116 and the 21st percentile of its own null (p=0.79)
  this is not evidence of a real effect. Note that
  `diff_age_x_experience` was subsequently built and cut in ADR-016
  (no measured contribution on either metric), so a future revisit
  should try a *different* encoding (explicit age-decline
  nonlinearity, or age bucketing) rather than re-testing the same
  multiplicative term. Explicitly NOT an RD/Glicko question — RD is
  low for actively-fighting veterans and would reinforce trust in
  their rating rather than discount it.
- **Strength of schedule, `n=5` window only**: if SoS is revisited,
  start from `diff_sos_last_5` alone. ADR-016's leave-one-out found
  dropping *only* `diff_sos_last_3` produced the best log loss of any
  configuration tested (0.659646), while dropping only the 5-window
  made things worse — consistent with two correlated columns where
  the noisier short window dilutes the longer one. Both inside the
  noise threshold, so not actionable alone, but the direction is
  recorded so it isn't re-derived.
- **Short-notice flag (<30 days between announcement and fight)**:
  not buildable with current data — no bout announcement date exists
  in the schema, and neither Greco's CSVs nor Wikipedia's
  `{{MMAevent bout}}` template carries one reliably. Would require a
  new data source (dated MMA news RSS, or manual tagging). Genuinely
  valuable signal (short-notice fighters underperform), so worth
  revisiting if the Week 6 news-ingestion work lands — but that
  pipeline is display-only by design, so promoting it to a model
  input would need its own leakage review first.
- **Style clustering (k-means archetypes on Tier 2 rates)**: planned
  for Week 3 Tuesday, cut for time after the ADR-015 permutation work.
  Never built, so ADR-016 says nothing about it either way. Design
  notes if revisited: cluster on Tier 2 rate features only (not
  physicals — style, not body type); `StandardScaler` + `KMeans` both
  **fit on train only** then `.transform()` applied to val (this is a
  genuinely *fitted* cross-row transform, unlike SoS/damage, so
  fitting on train+val would be a real leak); pick k via elbow +
  silhouette on train only; sanity-check centroids map to recognizable
  archetypes before shipping. Encoding: one-hot then difference (free,
  auto-discovered by `to_differential`) before trying an explicit
  archetype-pair categorical.
- **Increase the learning rate in LightGBM**: This comes at a risk of overfitting, but if we're training on a smaller dataset, it may be worth the risk in order to improve prediction accuracy. However, this could lead us to having to adjust ECE & log loss goals, which would make our model less professional.
- **Have LightGBM do more trials/training sessions**: This may also lead to overfitting, but again, I think these models are worth at least testing to see if we can achieve a market-level model.
- **Try XGBoost**: Less prone to overfitting. Uses more memory, but my dataset is relatively small
- **Train model strictly on people with multiple UFC fights**: Optimize the model for "main-card" worthy fighters, or people who have actually been in the UFC for a while