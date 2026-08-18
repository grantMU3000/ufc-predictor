"""
Extracts one de-vigged market win probability per (bout_id, fighter_id)
from odds_snapshots — the closing line. This is the "market baseline":
what did the sportsbooks think would happen, right before it happened?

Simple version: a sportsbook's moneyline isn't a pure prediction — it's
a pure prediction PLUS the house's cut (the "vig"), baked in so the book
profits regardless of outcome. Before you can compare "the market" to
your own model on a fair footing, you have to peel the vig back off.
De-vigging turns "what the book quoted you" back into "what the book
actually believed."
"""

import duckdb
import pandas as pd


def _moneyline_to_raw_prob(moneyline: pd.Series) -> pd.Series:
    """
    Converts American moneyline odds into a raw (still vig-inflated)
    implied probability.

    Two formulas, because moneylines are quoted two different ways
    depending on which side is favored:
      - Negative line (favorite, e.g. -150): you'd have to bet $150
        to win $100. Implied prob = 150 / (150 + 100) = 0.60
      - Positive line (underdog, e.g. +130): a $100 bet wins $130.
        Implied prob = 100 / (130 + 100) = 0.435

    "Raw" because the two fighters' numbers, for the same bout, will
    sum to slightly MORE than 1.0 — that gap is the book's margin,
    removed in the de-vig step of get_closing_lines, not here.
    """
    is_favorite = moneyline < 0
    raw_prob = pd.Series(index=moneyline.index, dtype=float)
    raw_prob[is_favorite] = -moneyline[is_favorite] / (-moneyline[is_favorite] + 100)
    raw_prob[~is_favorite] = 100 / (moneyline[~is_favorite] + 100)
    return raw_prob


def get_closing_lines(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    One de-vigged market win probability per (bout_id, fighter_id) —
    the market's final word right before the fight, with the
    sportsbook's margin removed.

    Four steps:
      1. Per (bout_id, fighter_id, sportsbook), keep only the LATEST
         snapshot (max collected_at) — that book's closing line.
         Earlier snapshots are line-movement history, not needed here.
      2. Convert each closing moneyline to a raw implied probability.
      3. DE-VIG: within each (bout_id, sportsbook), the two fighters'
         raw probs are renormalized to sum to exactly 1.0 — removing
         the book's built-in margin so what's left is belief, not
         belief plus markup.
      4. Where multiple sportsbooks cover the same bout, take the
         MEDIAN de-vigged probability across books. Median is more
         robust to one outlier book's bad line than a mean would be.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with an odds_snapshots view/table available.

    Returns
    -------
    pd.DataFrame: bout_id, fighter_id, market_prob.
    One row per (bout_id, fighter_id) with at least one odds snapshot.
    Bouts with no coverage simply don't appear — callers must inner-
    join, never assume every bout has a row here.
    """
    # Step 1: latest snapshot per (bout_id, fighter_id, sportsbook).
    # Done in DuckDB with a window function rather than pulling every
    # snapshot into pandas first — cheaper once 6x/day collection has
    # piled up months of history.
    query = """
        SELECT bout_id, fighter_id, sportsbook, moneyline
        FROM (
            SELECT
                bout_id, fighter_id, sportsbook, moneyline, collected_at,
                ROW_NUMBER() OVER (
                    PARTITION BY bout_id, fighter_id, sportsbook
                    ORDER BY collected_at DESC
                ) AS rn
            FROM odds_snapshots
        )
        WHERE rn = 1
    """
    closing = con.execute(query).df()

    # Step 2: moneyline -> raw implied probability.
    closing["raw_prob"] = _moneyline_to_raw_prob(closing["moneyline"])

    # Step 3: de-vig within each (bout_id, sportsbook) pair. A clean
    # pair sums to ~1.05; dividing each side by that sum rescales
    # both back to exactly 1.0 together.
    group_sum = closing.groupby(["bout_id", "sportsbook"])["raw_prob"].transform("sum")
    closing["market_prob"] = closing["raw_prob"] / group_sum

    # Step 4: median across books per (bout_id, fighter_id).
    result = (
        closing.groupby(["bout_id", "fighter_id"])["market_prob"].median().reset_index()
    )

    return result
