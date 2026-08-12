"""
Gets a fighter's ROUND-BY-ROUND performance stats (strikes, takedowns,
submissions, control time) for every fight strictly before a given
date. Pairs with bout_history.py, which only knows *who won* — this
file knows *how the fight actually went*, round by round.

Simple version: bout_history.py is the final score of a basketball
game. This file is the full box score — points, rebounds, assists,
quarter by quarter. You need the box score to answer questions like
"how good is this player's 3-point shooting," not just "did they win."

Same rule as bout_history.py applies here, and matters just as much:
only rows from BEFORE as_of_date are ever returned. A fighter's stats
from a fight next month can't leak into "how good were they last
week" — that's the same career-aggregate leakage trap from
docs/PLAN.md Section 0.2, just showing up in the stats table instead
of the results table.
"""

from datetime import date

import duckdb
import pandas as pd


def get_prior_bout_stats(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> pd.DataFrame:
    """
    Every round of every completed bout `fighter_id` fought strictly
    before `as_of_date`, with the fighter's own numbers in self_*
    columns and the opponent's in opp_* columns — so callers (Tier 2
    feature functions) never have to figure out red/blue corner or
    do their own self-join.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection pointed at the local Parquet snapshot.
    fighter_id : int
        The fighter whose stat history we want.
    as_of_date : date
        Cutoff — only rounds from fights strictly before this date.

    Returns
    -------
    pd.DataFrame — one row per (bout, round). Empty for a debut
    fighter or a fighter with only draws/no-contests so far.

    Why one row per ROUND, not one row per fight: some Tier 2
    features want per-fight totals (e.g. slpm = strikes landed per
    minute across the whole fight), but others might eventually care
    about round-level patterns (e.g. "fades in later rounds"). It's
    easy to sum round-rows up into fight-level totals inside a
    feature function; it's impossible to go the other way and split
    a fight-level total back into rounds. Returning the more detailed
    shape keeps every option open for later Tier 2/3 features.
    """
    query = """
        SELECT
            self_stats.bout_id,
            e.event_date,
            self_stats.round_number,
            self_stats.sig_strikes_landed      AS self_sig_strikes_landed,
            self_stats.sig_strikes_attempted   AS self_sig_strikes_attempted,
            self_stats.total_strikes_landed    AS self_total_strikes_landed,
            self_stats.total_strikes_attempted AS self_total_strikes_attempted,
            self_stats.takedowns_landed        AS self_takedowns_landed,
            self_stats.takedowns_attempted     AS self_takedowns_attempted,
            self_stats.sub_attempts            AS self_sub_attempts,
            self_stats.control_time_seconds    AS self_control_time_seconds,
            self_stats.knockdowns              AS self_knockdowns,
            self_stats.reversals               AS self_reversals,
            self_stats.head_strikes_landed     AS self_head_strikes_landed,
            self_stats.head_strikes_attempted  AS self_head_strikes_attempted,
            self_stats.body_strikes_landed     AS self_body_strikes_landed,
            self_stats.body_strikes_attempted  AS self_body_strikes_attempted,
            self_stats.leg_strikes_landed      AS self_leg_strikes_landed,
            self_stats.leg_strikes_attempted   AS self_leg_strikes_attempted,
            self_stats.distance_strikes_landed AS self_distance_strikes_landed,
            self_stats.distance_strikes_attempted AS self_distance_strikes_attempted,
            self_stats.clinch_strikes_landed   AS self_clinch_strikes_landed,
            self_stats.clinch_strikes_attempted AS self_clinch_strikes_attempted,
            self_stats.ground_strikes_landed   AS self_ground_strikes_landed,
            self_stats.ground_strikes_attempted AS self_ground_strikes_attempted,
            opp_stats.sig_strikes_landed       AS opp_sig_strikes_landed,
            opp_stats.sig_strikes_attempted    AS opp_sig_strikes_attempted,
            opp_stats.total_strikes_landed     AS opp_total_strikes_landed,
            opp_stats.total_strikes_attempted  AS opp_total_strikes_attempted,
            opp_stats.takedowns_landed         AS opp_takedowns_landed,
            opp_stats.takedowns_attempted      AS opp_takedowns_attempted,
            opp_stats.sub_attempts             AS opp_sub_attempts,
            opp_stats.control_time_seconds     AS opp_control_time_seconds,
            opp_stats.knockdowns               AS opp_knockdowns,
            opp_stats.reversals                AS opp_reversals,
            opp_stats.head_strikes_landed      AS opp_head_strikes_landed,
            opp_stats.head_strikes_attempted   AS opp_head_strikes_attempted,
            opp_stats.body_strikes_landed      AS opp_body_strikes_landed,
            opp_stats.body_strikes_attempted   AS opp_body_strikes_attempted,
            opp_stats.leg_strikes_landed       AS opp_leg_strikes_landed,
            opp_stats.leg_strikes_attempted    AS opp_leg_strikes_attempted,
            opp_stats.distance_strikes_landed  AS opp_distance_strikes_landed,
            opp_stats.distance_strikes_attempted AS opp_distance_strikes_attempted,
            opp_stats.clinch_strikes_landed    AS opp_clinch_strikes_landed,
            opp_stats.clinch_strikes_attempted AS opp_clinch_strikes_attempted,
            opp_stats.ground_strikes_landed    AS opp_ground_strikes_landed,
            opp_stats.ground_strikes_attempted AS opp_ground_strikes_attempted
        FROM bout_stats self_stats
        JOIN bouts b ON b.id = self_stats.bout_id
        JOIN events e ON e.id = b.event_id
        JOIN bout_stats opp_stats
            ON opp_stats.bout_id = self_stats.bout_id
            AND opp_stats.round_number = self_stats.round_number
            AND opp_stats.fighter_id != $fighter_id
        WHERE self_stats.fighter_id = $fighter_id
          AND b.status = 'completed'
          AND e.event_date < $as_of_date
        ORDER BY e.event_date ASC, self_stats.round_number ASC
    """
    return con.execute(query, {"fighter_id": fighter_id, "as_of_date": as_of_date}).df()