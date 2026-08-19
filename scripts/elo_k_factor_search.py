"""
Step 5 — K-factor grid search: scores several constant K values for
features.elo.compute_elo_ratings against val, through the same
models.metrics.evaluate harness every other baseline uses. Also
carries forward the Islam-vs-Magny diagnostic from the spot check,
since log loss alone might not catch what that comparison actually
tests for — fight-volume compounding vs. per-fight quality, not raw
calibration.

Not part of the pipeline — one-time tuning script, run by hand.
"""

import duckdb
import numpy as np
import pandas as pd

from features.elo import compute_elo_ratings, expected_score
from features.labels import get_completed_decided_bouts
from features.odds import get_closing_lines
from features.split import TEST_START
from models.metrics import evaluate

CANDIDATE_K_VALUES = [8, 16, 24, 32, 40, 48, 64, 80, 96, 128]

# Same two fighters from the spot check — a second, non-log-loss lens
# on each K: does raising K widen or narrow the "prolific-but-average
# fighter nearly matches a true elite" gap, or does it just make
# BOTH curves swing harder without changing the relationship?
DIAGNOSTIC_FIGHTERS = ["Islam Makhachev", "Neil Magny"]


def _resolve_fighter_id(con: duckdb.DuckDBPyConnection, name: str) -> int | None:
    """Exact match first, falls back to a printed ILIKE candidate list on a miss."""
    exact = con.execute(
        "SELECT id FROM fighters WHERE real_name = $name", {"name": name}
    ).fetchone()
    if exact:
        return exact[0]
    candidates = con.execute(
        "SELECT id, real_name FROM fighters WHERE real_name ILIKE $pattern",
        {"pattern": f"%{name}%"},
    ).df()
    print(f"    no exact match for '{name}'" + (
        f"; candidates:\n{candidates.to_string(index=False)}" if not candidates.empty else ""
    ))
    return None


def _score_one_k(
    k: float, labels: pd.DataFrame, val: pd.DataFrame, closing: pd.DataFrame
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Computes Elo at this K, scores it on full val AND on the
    odds-covered subset (same join market_baseline uses — directly
    comparable to the market/LR rows already in docs/RESULTS.md).
    Returns the raw elo table too, so main() doesn't recompute it for
    the diagnostic check below.
    """
    elo = compute_elo_ratings(labels, k_factor=k)

    merged = val.merge(elo, on="bout_id", how="inner")
    is_red = merged["source_corner"] == "red"
    self_elo = np.where(is_red, merged["red_elo_pre"], merged["blue_elo_pre"])
    opp_elo = np.where(is_red, merged["blue_elo_pre"], merged["red_elo_pre"])

    y_true = merged["self_won"].astype(int).to_numpy()
    y_prob = expected_score(self_elo, opp_elo)
    full_result = evaluate(y_true, y_prob, name=f"elo_k{k}_full_val")

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
    covered_result = evaluate(y_true_c, y_prob_c, name=f"elo_k{k}_odds_covered")

    return full_result, covered_result, elo


def main():
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts", "odds_snapshots"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    labels = get_completed_decided_bouts(con)
    # Train+val only — same leakage discipline as every step this week.
    labels = labels[pd.to_datetime(labels["event_date"]) < TEST_START].reset_index(drop=True)

    val = pd.read_parquet("data/processed/val.parquet")
    closing = get_closing_lines(con)

    fighter_ids = {name: _resolve_fighter_id(con, name) for name in DIAGNOSTIC_FIGHTERS}

    full_rows, covered_rows = [], []
    for k in CANDIDATE_K_VALUES:
        full_result, covered_result, elo = _score_one_k(k, labels, val, closing)
        full_rows.append(full_result)
        covered_rows.append(covered_result)

        # Diagnostic: each fighter's peak PRE-fight rating at this K.
        merged_labels = labels.merge(elo, on="bout_id")
        gap_bits = []
        for name, fid in fighter_ids.items():
            if fid is None:
                continue
            rows = merged_labels[
                (merged_labels["fighter_red_id"] == fid) | (merged_labels["fighter_blue_id"] == fid)
            ]
            self_pre = np.where(
                rows["fighter_red_id"] == fid, rows["red_elo_pre"], rows["blue_elo_pre"]
            )
            peak = self_pre.max() if len(self_pre) else float("nan")
            gap_bits.append(f"{name}={peak:.1f}")
        print(f"K={k:>3}  " + " | ".join(gap_bits))

    print("\n--- Full val ---")
    print(pd.DataFrame(full_rows).to_string(index=False))
    print("\n--- Odds-covered subset (comparable to market/LR baselines) ---")
    print(pd.DataFrame(covered_rows).to_string(index=False))


if __name__ == "__main__":
    main()