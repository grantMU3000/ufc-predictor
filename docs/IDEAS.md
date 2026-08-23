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
- **Post-hoc calibration (Platt / beta calibration, not isotonic)**:
  ADR-017 rejected both isotonic (fails ADR-004 symmetry — its step
  function amplifies LightGBM's existing corner-pair asymmetry, 1.86x
  the raw model's own deviation, confirmed on both the train-internal
  holdout and val) and Platt (real, correctly-signed ECE improvement,
  −0.0022, but below the −0.005 gate). Revisit at test unlock with
  more data. Two things to change next time, not re-derive from
  scratch: **(1)** start from Platt or beta calibration (smooth,
  won't shatter symmetry) — isotonic is structurally the wrong family
  for this symmetrized dual-row design, not just under-evidenced.
  **(2)** size the ECE-improvement gate to the *calibration holdout's
  own baseline ECE*, not to val's regressed number — this round's
  gate (0.005) was calibrated against val's 0.0318 while the holdout's
  actual baseline was 0.0133, effectively demanding Platt close ~38%
  of a gap sized for a different population. The underlying signal
  worth chasing: the raw model's reliability diagram shows genuine
  **underconfidence** (below diagonal at low p, above it past ~0.55),
  not overconfidence — a global stretch (which is what Platt does) is
  mechanistically the right tool for that shape, it just isn't earning
  its keep yet at this sample size.

  - **Path to market-level accuracy — a real roadmap, not a single feature**:
  P asked directly whether market parity is achievable long-term (any
  algorithm, any training window, any published competitor). Short
  answer: matching the market's **log loss** looks realistically
  reachable; consistently **beating** the closing line net of vig is a
  much harder, different claim, and most of the honest answer is about
  telling those two apart. Full reasoning below so this doesn't need
  re-deriving.

  **What won't close the gap (already tested or strongly implied by what's already been tested):**
  - *Algorithm swap (XGBoost/CatBoost instead of LightGBM).* Today's
    ensemble result is the direct evidence against this: three
    genuinely different model *shapes* (LR/LightGBM/Elo) blended to
    only -0.002 log loss, missing Gate A. XGBoost is a much closer
    cousin to LightGBM than Elo is to either — expect less than that,
    not more. Worth a 30-minute run for the model card's sake, not a
    real lever.
  - *More data.* Already directly measured and ruled out — ADR-016's
    fold-trend finding: training bouts grew 4.7x (1,291 -> 6,098
    bouts) across the fold range with `corr(val_year, log_loss) =
    0.073`, essentially zero. The model saturates on training volume
    early. This is the single most important diagnostic for this
    question: it means the project is **information-starved, not
    data-starved** — no amount of additional historical rows fixes a
    feature set that's already been fully learned from.
  - *Modern-only (~2010+) training window.* Still worth testing per
    the existing entry above, but the saturation finding cuts both
    ways — if more data doesn't help, less data (within reason)
    probably doesn't hurt much either. Expect a small effect in either
    direction, not a path to market parity on its own.

  **Where the real hope lives, ranked by plausibility:**
  1. **Fighter representation, not fighter statistics — the biggest
     lever.** The current feature set (32 `diff_` differentials) can
     express "who has the better takedown defense" but not "does this
     specific style beat that specific style" — a wrestler beating a
     striker who can't stop takedowns is a matchup interaction, not a
     stat differential, and nothing in the current design can
     represent it. Two converging pieces of evidence: (a) the
     project's own unbuilt style-clustering idea (entry above,
     k-means archetypes on Tier 2 rates), and (b) an external
     reference point found via search — a public UFC prediction site
     (mmamodel.ai) reports 67.6% accuracy / 0.598 log loss / 0.015 ECE
     on a stacked 5-model ensemble, and the detail that stands out is
     a Siamese neural network component (13% meta-weight) that learns
     a fighter-*embedding* space rather than reading hand-built
     stats — a strictly more powerful version of the style-clustering
     idea. (Numbers are self-reported with no visible leakage audit —
     treat as a directional existence proof, not a verified target.)
  2. **Short-notice / camp-disruption flag.** Already logged above as
     blocked on data availability. Restated here because it's a
     **systematic, one-directional** blind spot — the market prices
     this instantly (short-notice fighters underperform and the line
     moves accordingly) and the current model prices it at exactly
     zero. Revisit if Week 6's news-ingestion pipeline lands, with its
     own leakage review before promoting it from display-only to a
     model input.
  3. **Timing — beating the market's clock, not its knowledge.**
     Beating the *closing* line means outpredicting the market's
     final, fully-informed price. Beating the *opening* line only
     means being faster than the crowd. Search turned up a directly
     relevant, specific claim: prelim bouts get far less market
     attention and sharp money moves those lines less aggressively
     than main-card lines, so mispricing persists longer there.
     `docs/PLAN.md` §7 already flagged this as the most plausible
     path to real money; this is corroborating outside evidence, not
     a new idea, but it reframes prelim-specific backtesting as worth
     prioritizing over an all-card backtest.
  4. **Weight-cut / hydration signals.** Partially available already
     (`fighter_red_weigh_in_lbs` / `fighter_blue_weigh_in_lbs` exist
     in the schema, unused as features). Cheap to test relative to
     the other three.

  **The honest ceiling, per the literature (not project-specific,
  general sports-betting market efficiency research found via
  search):** one large multi-sport study found markets broadly
  efficient with no odds-based strategy yielding statistically
  significant long-term profit — but the same study specifically
  flagged UFC underdog bets as showing a *positive* return over an
  extended period, without clearing statistical significance. That's
  the right way to read this project's own realistic ceiling: a real,
  recurring signal that keeps showing up and keeps failing to be
  provable at this sample size. Suggestive that a durable edge may
  exist in a specific slice (underdogs, prelims, short-notice) even if
  it's implausible across the board.

  **Rough near-term target, not a promise:** log loss ~0.62-0.63 with
  style/embedding features added — closing roughly a third of the
  current 0.059 gap to market (0.6483 -> market's 0.5897). Full market
  log-loss parity is a plausible 12+ month goal, most likely to arrive
  through better fighter representation than through more tuning of
  the current feature set. A genuine, bettable edge net of vig is a
  "keep the prediction ledger running and find out honestly" question,
  not something to claim in advance — which is exactly what the
  ledger (`predictions`/`prediction_results`, `docs/PLAN.md` §3 Week 4)
  already exists to answer.

  **Long-term connection worth naming explicitly:** the original
  project scope's long-term goal #2 (fight-footage analysis, cut from
  the 6-week short-term plan as a multi-month problem) is not a
  separate stretch idea from this — it's arguably the *real* long-term
  answer to this exact question. The market's edge ultimately comes
  from thousands of people having watched the fighters fight; a model
  that can watch fight footage is the only version of this project
  that competes with that directly rather than approximating it
  secondhand through box-score-style stats.

  - **Long-term v2: prop/stat prediction instead of (or alongside)
  winner prediction.** Instead of "who wins," predict a distribution
  over an individual fighter's stat (e.g. significant strikes landed)
  for an upcoming bout, and compare it to the market's posted
  over/under line — surfacing where they disagree meaningfully.
  **Why this is promising:** props are priced by far fewer sharp
  bettors than moneylines (the plausible-edge case `docs/PLAN.md` §7
  already names), the project already has per-round `bout_stats`
  currently thrown away by the winner model (strikes/TDs/control time
  by round, by fighter), and a volume target is likely more learnable
  than a binary outcome — strike counts accumulate roughly as a rate
  process, while a fight outcome can flip on one exchange.
  **The hard parts, not yet solved, in rough priority order:**
  - *Censoring*: "over 60 strikes" is really rate x duration jointly —
    a fight ending in round 2 isn't a failure to reach 60, it's a
    truncated observation. Needs a duration model and a rate model
    integrated together (survival-analysis framing), not a
    direct regression.
  - *No historical prop-odds data*: The Odds API's coverage (ADR-008)
    is moneylines only. Without historical prop lines there's no
    backtest, and without a backtest this can't be trusted with real
    money. Needs its own data-source investigation before any modeling
    starts.
  - *No live stat feed for the in-play version*: knowing a live strike
    count mid-round requires a real-time counting source that doesn't
    currently exist and that ufcstats (ADR-006) won't provide.
  - *The originally-imagined trigger was backwards*: alerting when a
    line moves (e.g. -150 -> -200) is chasing a move the market already
    priced, not finding an edge. A real signal compares the model's
    own fair-value estimate against the CURRENT line, not against the
    line's own recent movement.
  - *Latency and regulatory scope*, for the live/alerting version
    specifically — competing on speed against feeds this project won't
    have, and "message the user to place a bet" is a different
    liability surface than a passive prediction ledger.
  **Recommended sequencing when this gets picked up:** start with the
  PRE-FIGHT version only (predict the distribution, compare to the
  posted line, no live infra) — reuses the existing ingestion
  pipeline, point-in-time feature store, and prediction ledger
  almost unchanged, and isolates the real technical problem
  (censored-duration modeling) from the much harder live-data problem.
  The in-play/alerting version is a deployment question to revisit
  only once the pre-fight model is proven out. Method-of-victory props
  are a related, similarly-lazy market worth a look at the same time —
  `method`/`method_detail`/`ending_round` are already in the schema.
  Explicitly out of scope for the current 6-week plan per §0.3's cut
  list; this is a v2/long-term entry, not a Week 4-6 item.

  ### [2026-08-23] Investigate why test log loss beat val (unexpected-direction miss on ADR-020 Decision 5)

**Context:** Shipping artifact (B) scored 0.6333 log loss on test
(odds-covered), beating val's 0.6483 by 0.0150 — just outside the
±0.01 generalization tolerance pre-registered in ADR-020. Per the
project's standing "any 3+ point jump is a leak until proven
otherwise" instinct, an unexpectedly *better* number deserves the same
scrutiny as an unexpectedly worse one, not a pass.

**What to check, for a hypothetical v2 only — never retroactively on
v1 (ADR-020's hard no-further-tuning rule):**
- Year-by-year and weight-class composition of test vs. val — does
  test skew toward fight types (favorites, weight classes, veteran
  fighters) the model already handles well?
- Favorite/underdog mix — `docs/RESULTS.md`'s 2019/2013 fold-variance
  note (worst/best CV folds despite similar training size) suggests
  year-to-year composition swings are already a known source of
  variance; worth confirming test isn't simply an easier year by that
  same mechanism.
- Re-run `train_val_distribution_check()` (the KS-drift tool from the
  [2026-08-16] leakage entry) with test included, now that Preflight
  2/2 confirms test's boundary is clean.

**Priority:** low — the result is favorable, not alarming, and doesn't
block anything. Logged so it isn't lost, not because it's urgent.

---

### [2026-08-23] Calibration retry (Platt/beta) — test result adds evidence, doesn't change the plan

**Status:** Reinforces an idea already logged after ADR-017; not a new
idea, but test provides the first out-of-sample confirmation.

**What test showed:** shipping artifact's ECE moved from 0.0318 (val)
to 0.0511 (test) — narrowly missing the 0.05 target for the first
time, and moving in the same underconfident-past-tuning direction
ADR-017 already documented at the val stage. The gap to the market's
ECE also widened (market 0.0236 vs. model 0.0511, roughly 2.2x).

**Still the right starting point, per the existing note:** Platt or
beta calibration, NOT isotonic (isotonic already failed Gate C's
symmetry check in ADR-017 — a structural incompatibility with this
project's dual-row design, not a magnitude issue that more data would
fix). Any refit should size its ECE gate to the calibration holdout's
own baseline rather than reusing val's 0.0318, since ADR-017 already
found ECE varies ~2x by bin count and context alone.

**Scope:** applies only to a from-scratch, re-validated v2 — ADR-020
forecloses any change to v1 regardless of this evidence.

---

### [2026-08-23] Kelly/backtest result as documented justification for the real-money betting gate

**Context:** `docs/PLAN.md` §0.3 already cuts real-money betting until
3+ months of logged out-of-sample results exist. Test's Kelly
simulation (quarter-Kelly, 5% cap) turned $100 into ~$1.67–1.73 across
all three edge thresholds — a ~98% drawdown — despite the model
having only a moderate calibration gap versus the market (ECE 0.051
vs 0.024).

**Idea, not urgent:** this is strong, concrete evidence for the model
card and README explaining *why* that rule exists, rather than
treating it as a generic disclaimer. Worth a short "what this number
means" callout — a small, real miscalibration compounding into
near-total bankroll loss under compounding stakes is a more persuasive
argument than the rule stated on its own.

**Scope:** documentation only, no code or model change implied.