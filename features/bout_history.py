"""
The single function almost every feature calls: "give me everything
this fighter did BEFORE this date, and nothing after."

Simple version: this is the fighter's scouting report, frozen at a
moment in time — like a photo that can't show anything that happened
after the shutter clicked. Every Tier 2 feature (strike accuracy,
takedown defense, win streak...) is really just "look at this photo
and do some math." If 20 feature functions each took their own photo,
that's 20 chances for one of them to sneak a future fight into frame
— exactly the "career-aggregate leakage" trap in docs/PLAN.md §0.2.
One trusted, tested function here means that mistake can only happen
once, not 20 times.
"""

from datetime import date

import duckdb
import pandas as pd


def get_prior_bouts(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> pd.DataFrame:
    """
    Every completed bout `fighter_id` fought strictly before `as_of_date`,
    oldest first. Columns are normalized to self_/opp_ so callers never
    have to think about red/blue corner.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection pointed at the local Parquet snapshot (see snapshot.py).
    fighter_id : int
        The fighter whose history we want.
    as_of_date : date
        The cutoff. Only fights strictly before this date are included.
        For a fight ON 2024-06-15, pass 2024-06-15 — never let the fight
        see itself.

    Returns
    -------
    pd.DataFrame — one row per prior bout. Empty DataFrame for a debut
    fighter (this is the signal your min_prior_fights guard checks for).
    """
    query = """
        SELECT
            b.id AS bout_id,
            e.event_date,
            CASE WHEN b.fighter_red_id = $fighter_id
                 THEN b.fighter_red_id ELSE b.fighter_blue_id END AS self_id,
            CASE WHEN b.fighter_red_id = $fighter_id
                 THEN b.fighter_blue_id ELSE b.fighter_red_id END AS opp_id,
            CASE WHEN b.winner_id = $fighter_id THEN true
                 WHEN b.winner_id IS NULL THEN NULL
                 ELSE false END AS self_won,
            b.method,
            b.ending_round,
            b.ending_time_seconds
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE (b.fighter_red_id = $fighter_id OR b.fighter_blue_id = $fighter_id)
          AND b.status = 'completed'
          AND e.event_date < $as_of_date
        ORDER BY e.event_date ASC
    """
    return con.execute(query, {"fighter_id": fighter_id, "as_of_date": as_of_date}).df()