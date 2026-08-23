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

def _latest_snapshots(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    The closing snapshot for every (bout_id, fighter_id, sportsbook) —
    the shared first step behind both get_closing_lines (belief) and
    get_closing_prices (payout).

    Pulled out as its own function so the two consumers can never
    drift apart on WHICH snapshot counts as "closing." If that
    definition ever changes (e.g. "last snapshot before the scheduled
    start" instead of "last snapshot, period"), it changes in exactly
    one place and both the market baseline and the ROI backtest move
    together.

    Returns
    -------
    pd.DataFrame: bout_id, fighter_id, sportsbook, moneyline
    """
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
    return con.execute(query).df()


def american_to_decimal(moneyline: pd.Series) -> pd.Series:
    """
    American moneyline -> decimal odds (total return per 1 unit
    staked, stake included).

      -150  ->  1 + 100/150  =  1.6667   (bet 1, get back 1.67 on a win)
      +130  ->  1 + 130/100  =  2.3000   (bet 1, get back 2.30 on a win)

    Decimal is the format the rest of the backtest works in because
    profit-per-unit is just (decimal - 1), with no favorite/underdog
    branching anywhere downstream. American odds are how the book
    quotes it; decimal is how you actually do arithmetic with it.
    """
    is_favorite = moneyline < 0
    decimal = pd.Series(index=moneyline.index, dtype=float)
    decimal[is_favorite] = 1 + 100 / -moneyline[is_favorite]
    decimal[~is_favorite] = 1 + moneyline[~is_favorite] / 100
    return decimal


def get_closing_prices(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    The actual PRICE you'd have been paid at closing, per
    (bout_id, fighter_id) — vig included, deliberately.

    This is the counterpart to get_closing_lines(), and the two are
    used for different jobs that must not be mixed up:
      - market_prob (get_closing_lines) = what the book BELIEVED,
        vig stripped out. Used to compare model vs. market and to
        compute edge.
      - decimal_odds (here) = what the book PAYS, vig left in. Used
        to compute money won and lost.
    Using de-vigged probability as a payout price would invent a
    profit that never existed — it's the single most common way a
    betting backtest fabricates an edge.

    AGGREGATION: median is taken over DECIMAL odds, not over American
    moneylines. American odds are a discontinuous scale (there's no
    such thing as a line between -100 and +100), so a median computed
    on them isn't a meaningful average price. Decimal is continuous,
    so the median there is a real number.

    `best_decimal_odds` is also returned but is NOT the default for
    the backtest. Always taking the best available price across books
    assumes you had accounts at every sportsbook and always shopped
    perfectly — realistic for a sharp bettor, optimistic for an
    honest backtest. Median is the conservative choice, matching how
    get_closing_lines already aggregates belief. Reported alongside
    so the gap between the two is visible rather than assumed.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection with an odds_snapshots view/table available.

    Returns
    -------
    pd.DataFrame: bout_id, fighter_id, decimal_odds, best_decimal_odds,
    n_books. Same coverage as get_closing_lines — inner-join, never
    assume every bout appears.
    """
    closing = _latest_snapshots(con)
    closing["decimal_odds"] = american_to_decimal(closing["moneyline"])

    result = (
        closing.groupby(["bout_id", "fighter_id"])
        .agg(
            decimal_odds=("decimal_odds", "median"),
            best_decimal_odds=("decimal_odds", "max"),
            n_books=("sportsbook", "nunique"),
        )
        .reset_index()
    )
    return result