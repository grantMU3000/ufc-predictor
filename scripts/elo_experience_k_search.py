"""
Step 5 (part 2) — grid search over the smooth-decay K-factor's three
parameters (k_new, k_veteran, decay_scale), scored through the same
models.metrics.evaluate harness as everything else this week. This
supersedes the flat-K grid (elo_k_factor_search.py) as the tuning
step whose winner becomes features/elo.py's actual default — flat K
stays in the codebase as a documented, tested option, just not the
one used going forward.

Why tune all THREE together instead of one at a time: they interact.
A small decay_scale only matters if k_new and k_veteran are far
enough apart for the "cooling" to be visible; a big gap between
k_new/k_veteran is wasted if decay_scale is so small everyone's
already cooled down by fight #2. Tuning them jointly is the only way
to not miss a combination that only works well together.

Not part of the pipeline — one-time tuning script, run by hand.
"""

import itertools
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from features.elo import compute_elo_ratings, expected_score, k_factor_by_experience
from features.labels import get_completed_decided_bouts
from features.odds import get_closing_lines
from features.split import TEST_START
from models.metrics import evaluate

# k_new candidates are all comfortably above every k_veteran
# candidate on purpose, so every combination is a genuine decay
# (starts high, ends low) — never an accidental increase.
K_NEW_CANDIDATES = [64, 72, 80]
K_VETERAN_CANDIDATES = [16, 24, 32]
DECAY_SCALE_CANDIDATES = [1, 2, 3, 5]

RESULTS_DIR = Path("data/tuning")


def _score_one_config(
    k_new: float,
    k_veteran: float,
    decay_scale: float,
    labels: pd.DataFrame,
    val: pd.DataFrame,
    closing: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Same scoring shape as elo_k_factor_search.py's _score_one_k, just
    pointed at k_factor_by_experience instead of a bare float. The
    closure over (k_new, k_veteran, decay_scale) is what lets
    compute_elo_ratings call k_fn(fight_count) with exactly one
    argument, matching its Callable[[int], float] contract.
    """
    def k_fn(fight_count: int) -> float:
        return k_factor_by_experience(
            fight_count, k_new=k_new, k_veteran=k_veteran, decay_scale=decay_scale
        )

    elo = compute_elo_ratings(labels, k_factor=k_fn)
    params = {"k_new": k_new, "k_veteran": k_veteran, "decay_scale": decay_scale}
    name = f"elo_kn{k_new}_kv{k_veteran}_ds{decay_scale}"

    merged = val.merge(elo, on="bout_id", how="inner")
    is_red = merged["source_corner"] == "red"
    self_elo = np.where(is_red, merged["red_elo_pre"], merged["blue_elo_pre"])
    opp_elo = np.where(is_red, merged["blue_elo_pre"], merged["red_elo_pre"])
    y_true = merged["self_won"].astype(int).to_numpy()
    y_prob = expected_score(self_elo, opp_elo)
    full_result = evaluate(y_true, y_prob, name=f"{name}_full_val") | params

    covered = merged.merge(
        closing,
        left_on=["bout_id", "self_fighter_id"],
        right_on=["bout_id", "fighter_id"],
        how="inner",
    )
    is_red_c = covered["source_corner"] == "red"
    self_elo_c = np.where(is_red_c, covered["red_elo_pre"], covered["blue_elo_pre"])
    opp_elo_c = np.where(is_red_c, covered["blue_elo_pre"], covered["red_elo_pre"])
    y_true_c = covered["self_won"].astype(int).to_numpy()
    y_prob_c = expected_score(self_elo_c, opp_elo_c)
    covered_result = evaluate(y_true_c, y_prob_c, name=f"{name}_odds_covered") | params

    return full_result, covered_result


def _flag_edge_values(best_row: pd.Series) -> None:
    """
    Same lesson as the flat-K grid: if the winning combo sits at the
    EDGE of a candidate list rather than somewhere in the interior,
    that's a sign the grid didn't actually bracket a peak for that
    parameter — extend it before trusting the result, don't just take
    the best-of-what-we-tried.
    """
    edges = {
        "k_new": (min(K_NEW_CANDIDATES), max(K_NEW_CANDIDATES)),
        "k_veteran": (min(K_VETERAN_CANDIDATES), max(K_VETERAN_CANDIDATES)),
        "decay_scale": (min(DECAY_SCALE_CANDIDATES), max(DECAY_SCALE_CANDIDATES)),
    }
    for param, (lo, hi) in edges.items():
        if best_row[param] in (lo, hi):
            print(f"  NOTE: best {param}={best_row[param]} is at the EDGE of "
                  f"the tested range ({lo}-{hi}) — extend the grid before trusting this.")


def main():
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts", "odds_snapshots"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    labels = get_completed_decided_bouts(con)
    labels = labels[pd.to_datetime(labels["event_date"]) < TEST_START].reset_index(drop=True)

    val = pd.read_parquet("data/processed/val.parquet")
    closing = get_closing_lines(con)

    combos = list(itertools.product(K_NEW_CANDIDATES, K_VETERAN_CANDIDATES, DECAY_SCALE_CANDIDATES))
    print(f"scoring {len(combos)} (k_new, k_veteran, decay_scale) combinations...")

    full_rows, covered_rows = [], []
    for i, (k_new, k_veteran, decay_scale) in enumerate(combos, 1):
        full_result, covered_result = _score_one_config(k_new, k_veteran, decay_scale, labels, val, closing)
        full_rows.append(full_result)
        covered_rows.append(covered_result)
        if i % 6 == 0 or i == len(combos):
            print(f"  {i}/{len(combos)} done")

    full_df = pd.DataFrame(full_rows)
    covered_df = pd.DataFrame(covered_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(RESULTS_DIR / "elo_experience_k_grid_full_val.csv", index=False)
    covered_df.to_csv(RESULTS_DIR / "elo_experience_k_grid_odds_covered.csv", index=False)
    print(f"\nfull {len(combos)}-row results written to {RESULTS_DIR}/elo_experience_k_grid_*.csv")

    cols = ["k_new", "k_veteran", "decay_scale", "n", "accuracy", "log_loss", "brier", "ece"]

    print("\n--- Top 10 by log_loss (odds-covered subset — comparable to market/LR) ---")
    top_ll = covered_df.sort_values("log_loss").head(10)
    print(top_ll[cols].to_string(index=False))
    print()
    _flag_edge_values(top_ll.iloc[0])

    print("\n--- Top 10 by ECE (odds-covered subset) ---")
    top_ece = covered_df.sort_values("ece").head(10)
    print(top_ece[cols].to_string(index=False))
    print()
    _flag_edge_values(top_ece.iloc[0])


if __name__ == "__main__":
    main()