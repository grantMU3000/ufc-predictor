"""
Baseline models for Week 2 Thursday (docs/PLAN.md Section 3). Each
baseline function returns (y_true, y_prob) in the same shape, so every
one of them can be scored through the exact same models.metrics.evaluate
call — no baseline gets graded on a curve.

Today's baseline: the market itself. Tomorrow (Elo) and the LR baseline
get added to this same file, so docs/RESULTS.md eventually pulls every
row from one script, one run, one consistent evaluation pass.
"""

import duckdb
import numpy as np
import pandas as pd

from features.odds import get_closing_lines


def market_baseline(
    val: pd.DataFrame, closing: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Turns the market's de-vigged closing probability into predictions
    for the "always pick the favorite" baseline — docs/PLAN.md Section
    0.1's actual bar to clear, not the naive 60% in the original scope.

    Simple version: for every fight in val, ask "what did the market
    think self_fighter_id's odds of winning were, right before the
    fight?" That number IS the prediction. No model, no features —
    just the closing line, scored the same way everything else will be.

    Join key: bout_id + self_fighter_id <-> bout_id + fighter_id. This
    is exactly why self_fighter_id was added to _symmetrize_row back in
    Wednesday's session — without it, there'd be no clean way to attach
    a per-fighter market probability onto a self_/opp_ row.

    Parameters
    ----------
    val : pd.DataFrame
        The symmetrized validation split (data/processed/val.parquet).
        Needs bout_id, self_fighter_id, self_won.
    closing : pd.DataFrame
        Output of features.odds.get_closing_lines — bout_id, fighter_id,
        market_prob.

    Returns
    -------
    (y_true, y_prob, merged)
        y_true : 0/1 array, whether self_fighter_id actually won
        y_prob : float array, market's de-vigged probability self_fighter_id wins
        merged : the joined DataFrame itself, kept around for debugging
                 and for the coverage check below — NOT every val row
                 survives this join, only ones with odds coverage.

    Note: INNER join, deliberately. A bout with no odds coverage has no
    market opinion to grade — there's no sensible y_prob to invent for
    it, so it's correctly absent from the baseline's n, not silently
    filled with a guess.
    """
    merged = val.merge(
        closing,
        left_on=["bout_id", "self_fighter_id"],
        right_on=["bout_id", "fighter_id"],
        how="inner",
    )

    y_true = merged["self_won"].astype(int).to_numpy()
    y_prob = merged["market_prob"].to_numpy()

    return y_true, y_prob, merged


def _load_val_and_odds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the symmetrized val split and the closing-line odds table
    from local Parquet — the standard "wire up a DuckDB connection to
    the snapshot" pattern from split.py's __main__ block, reused here
    rather than duplicated by hand.

    Returns
    -------
    (val, closing) — ready to pass straight into market_baseline.
    """
    con = duckdb.connect()
    for table in ["odds_snapshots"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    val = pd.read_parquet("data/processed/val.parquet")
    closing = get_closing_lines(con)
    con.close()

    return val, closing


if __name__ == "__main__":
    from models.metrics import evaluate

    val, closing = _load_val_and_odds()
    y_true, y_prob, merged = market_baseline(val, closing)

    # Coverage check — reports what fraction of val actually got
    # scored, since the inner join silently drops uncovered bouts.
    # val is symmetrized (2 rows/bout), so compare row counts, not
    # bout counts, to match what evaluate() actually sees.
    print(f"val rows: {len(val)} | scored rows: {len(merged)} "
          f"({len(merged) / len(val):.1%} coverage)")

    result = evaluate(y_true, y_prob, name="market_baseline")
    print(result)