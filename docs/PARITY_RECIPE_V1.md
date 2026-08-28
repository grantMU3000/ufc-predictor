# Inference Feature Parity Recipe — v1 (32 features)

Traced from source and verified against models/v1/metadata.json
(feature_list order + count matched exactly). This is the
transformation order that produced the matrix model.txt was fit on.
The inference path must reproduce it step for step.

Canonical shipped path: `build_train_val_with_elo()` with all three
Tier 3 flags at defaults (all False — ADR-016 cut them). The
`__main__` block in build_lgbm_matrix.py forces them ON; that is a
build smoke test, NOT the shipped path. Do not copy from it.

Every DB read in Stages A1–A4 goes through a DuckDB connection with
views over data/processed/*.parquet. See Finding A.

---

## Stage A — Per-bout raw feature row

Owner: `features/store.py::build_feature_row(con, bout_id)`

A1. `_get_bout_context(con, bout_id)`
      SELECT b.fighter_red_id, b.fighter_blue_id, e.event_date
      FROM bouts b JOIN events e ON e.id = b.event_id WHERE b.id = ?
    -> raises ValueError if bout_id not found
    -> `as_of_date = event_date`  (THE BOUT'S OWN DATE — see Finding B)

A2. Seed dict, IN THIS ORDER (dict insertion order is load-bearing,
    Finding C):
      {"bout_id": ..., "event_date": as_of_date}

A3. Station 1 — for corner in (red, blue), for each of the 32
    FIGHTER_FEATURES in list order:
      row[f"{corner}_{name}"] = fn(con, fighter_id, as_of_date)

    All 32, in exact order:
      age, height_cm, reach_cm, reach_to_height_ratio, stance,
      slpm, sapm, td_avg_per_15, sub_attempts_per_15,
      striking_accuracy, takedown_accuracy,
      significant_strike_rate, striking_defense, takedown_defense,
      career_win_pct, decision_win_pct, decision_loss_pct,
      submission_win_count, submission_success_rate, finish_rate,
      ko_loss_rate, sub_loss_rate, total_ufc_fights,
      days_since_last_fight, avg_fight_time_seconds,
      title_fight_experience, times_knocked_down, knockdown_rate,
      control_time_pct, time_controlled_pct,
      striking_output_decay, takedown_output_decay

    Tier 1 (5): direct column reads off `fighters`. Not date-
      dependent except `age`, which is `(as_of_date - dob).days /
      365.25`. Missing -> None, never imputed.
    Tier 2 (27): all read via get_prior_bouts / get_prior_bout_stats
      (features/bout_history.py, bout_stats_history.py), which own
      the `event_date < as_of_date` STRICT filter. Debutants return
      None for rates, 0 for true counts (submission_win_count,
      total_ufc_fights, title_fight_experience, times_knocked_down).

A4. Station 2 — stance_matchup, once per corner, self's POV:
      row[f"{corner}_stance_matchup_descriptive"]
      row[f"{corner}_is_open_stance_matchup"]

A5. Station 3 — bout-level, no corner prefix, in BOUT_FEATURES order:
      weight_class, is_title_fight, scheduled_rounds

Resulting key order:
  bout_id, event_date,
  red_<32>, blue_<32>,
  red_stance_matchup_descriptive, red_is_open_stance_matchup,
  blue_stance_matchup_descriptive, blue_is_open_stance_matchup,
  weight_class, is_title_fight, scheduled_rounds

---

## Stage B — Symmetrize

Owner: `features/symmetrize.py::_symmetrize_row(row, red_id,
blue_id, self_fighter_id)`

B1. Pick sources:
      self_fighter_id == red_id  -> self_source="red_",  opp="blue_"
      self_fighter_id == blue_id -> self_source="blue_", opp="red_"
      neither                    -> ValueError

B2. Walk `row.items()` IN ORDER:
      key startswith self_source -> `self_{suffix}`
      key startswith opp_source  -> `opp_{suffix}`, EXCEPT
        suffix in {stance_matchup_descriptive,
                   is_open_stance_matchup} -> DROPPED (redundant
        mirror)
      no corner prefix           -> copied through unchanged

B3. Append, in order: self_fighter_id, opp_fighter_id, source_corner

B4. Training additionally sets `self_won = (self_fighter_id ==
    winner_id)` in build_symmetrized_dataset (NOT inside
    _symmetrize_row). Inference has no label — Finding D.

Training loops corners as `(fighter_red_id, fighter_blue_id)`, so
the RED-perspective row is always built first. Finding C.

---

## Stage C — Split (training only)

`temporal_split` on event_date: train < 2023-01-01, val < 2025-01-01,
test >= 2025-01-01. Artifact B = train + val, i.e. event_date 
2025-01-01. Written to parquet; Stage A/B never re-run after Week 2.

---

## Stage D — Elo

D1. DuckDB views over data/processed/{fighters,events,bouts}.parquet

D2. `labels = get_completed_decided_bouts(con)`
      status='completed' AND winner_id IS NOT NULL
      (draws + NCs out; DQs in)
      ORDER BY event_date ASC

D3. Filter labels to `event_date < TEST_START (2025-01-01)`
      (`_load_labels_and_elo` default cutoff)

D4. `compute_elo_ratings(labels, k_factor=k_fn)`
      k_fn = k_factor_by_experience(count, k_new=80.0,
             k_veteran=24.0, decay_scale=3.0)     [ADR-014]
      initial_rating = 1500.0
      Sequential oldest-first walk. PRE-fight rating recorded BEFORE
      the update. Per-fighter fight_count increments across the walk
      and drives the K decay.
      Raises if event_date not monotonically increasing.
      -> bout_id, red_elo_pre, blue_elo_pre

D5. `attach_by_corner(split, elo_ratings, stems=["elo_pre"])`
      LEFT merge on bout_id; raises on row-count change or unmatched
      is_red = (source_corner == "red")
      self_elo_pre = red if is_red else blue
      opp_elo_pre  = blue if is_red else red
      drops red_elo_pre / blue_elo_pre
      NEW COLUMNS LAND AT THE END OF THE FRAME -> diff_elo_pre is
      the last feature.

## Stage D-skip — Tier 3 NOT in v1

Never called for the shipped model: build_sos_by_bout,
build_recent_damage_by_bout / add_interaction_features,
build_weight_class_change_by_bout. Assert the inference path doesn't
touch features/tier3.py at all.

---

## Stage E — Differencing

Owner: `features/differential.py::to_differential(df)`

E1. y = df["self_won"].astype(int)             [Finding D]
E2. Iterate [c for c in df.columns if c.startswith("self_")] IN
    DATAFRAME COLUMN ORDER. suffix = c[5:]:
      "won"                  -> skip (it's y)
      "fighter_id"           -> skip (identity, ADR-004)
      no opp_{suffix}        -> DROP (self_-only)
      either side non-numeric-> DROP (can't subtract)
      else -> diff_{suffix} = self.astype(float) - opp.astype(float)
E3. Any opp_-only column -> warn + drop
E4. X = pd.DataFrame(diff_data, index=df.index)
      NaNs PRESERVED, no imputation anywhere.
      Column order = dict insertion order = df.columns order.

Confirmed drops for v1 (3 columns, all expected):
  self_stance / opp_stance          -> non-numeric
  self_stance_matchup_descriptive   -> self_-only (opp_ dropped in B2)
  self_is_open_stance_matchup       -> self_-only (opp_ dropped in B2)
  => STANCE CARRIES ZERO WEIGHT IN v1. Not a bug; a consequence.

Never enter X (no self_/opp_ prefix): bout_id, event_date,
weight_class, is_title_fight, scheduled_rounds, source_corner.

Result: 32 diff_ columns. VERIFIED against metadata.json —
count and order both match.

---

## Stage F — Model input

X[list(model.feature_order)]  — explicit positional reindex.
Assert set equality FIRST so a missing/extra feature raises instead
of being silently dropped by the selection.

---

## Inference deltas

| Stage | Training | Inference |
|---|---|---|
| A | DuckDB over parquet, materialized Week 2 | needs a DuckDB con — Finding A |
| A1 | completed bout | scheduled bout; same query works, cols are NOT NULL |
| B | red row then blue row, self_won attached | both rows, no label — Findings C, D |
| C | read from parquet | no parquet row exists |
| D2 | labels only | target bout is `scheduled`, excluded from labels — Finding E |
| D4 | mid-walk pre-fight rating | need FINAL ratings after last completed bout — Finding E |
| D5 | merge on bout_id | no bout_id match; assign self_/opp_ directly from source_corner |