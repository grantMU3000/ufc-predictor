"""
Tier 2 features: point-in-time career RATES, computed only from a
fighter's fights strictly before the one being predicted. Everything
here builds on bout_history.py / bout_stats_history.py — never
queries bouts/bout_stats directly, so the "only look at the past"
rule only has to be gotten right in one place.
"""

from datetime import date

import duckdb

from features.bout_history import get_prior_bouts

from features.bout_stats_history import get_prior_bout_stats

ROUND_LENGTH_SECONDS = 300  # 5-minute UFC rounds. Safe across this
# entire dataset — pre-Unified-Rules-era bouts (which used different
# round lengths) are already excluded at ingestion (PLAN_ADDENDUM §7),
# so every bout this function will ever see follows the modern format.


def get_fight_duration_seconds(
    method: str | None,
    ending_round: int | None,
    ending_time_seconds: int | None,
    scheduled_rounds: int,
) -> float | None:
    """
    How many seconds a single fight actually lasted.

    Two cases, and they need different math:
    - Decision (fight went the full distance): every scheduled round
      happened in full, so duration = scheduled_rounds * 5 minutes.
    - Finish (KO/TKO/Submission, fight ended early): duration = every
      full round before the finish, plus the exact time into the
      round it ended in.

    ASSUMPTION TO CONFIRM: this checks whether `method` starts with
    "Decision" to tell the two cases apart. If your real data uses a
    different convention, this needs adjusting before it's trusted.

    Parameters
    ----------
    method : str | None
        e.g. "Decision - Unanimous", "KO/TKO", "Submission".
    ending_round : int | None
        Round the fight ended in. Only used for finishes.
    ending_time_seconds : int | None
        Time within ending_round the fight ended. Only used for finishes.
    scheduled_rounds : int
        How many rounds this bout was scheduled for (3 or 5).

    Returns
    -------
    float | None — duration in seconds, or None if this bout's
    finish data is incomplete (a real data gap, not a debut).
    """
    if method is not None and method.strip().lower().startswith("decision"):
        return float(scheduled_rounds * ROUND_LENGTH_SECONDS)

    if ending_round is None or ending_time_seconds is None:
        return None  # genuine data gap on this one bout

    return float((ending_round - 1) * ROUND_LENGTH_SECONDS + ending_time_seconds)


def get_total_seconds_fought(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float:
    """
    Total seconds this fighter has spent actually fighting, across
    every completed prior bout — the shared denominator behind slpm,
    sapm, td_avg_per15, sub_avg_per15, control_time_pct, and
    time_controlled_pct.

    Returns 0.0 for a debutant — NOT None. This is a real, true zero
    (they genuinely haven't fought yet), not a gap in our knowledge.
    Every rate feature built on top of this is responsible for its
    own "what if the denominator is zero" guard — dividing by 0.0
    honestly here would crash or produce garbage, so that check
    belongs in each rate function, not hidden inside this one.

    A bout with incomplete finish data (get_fight_duration_seconds
    returns None for it) is skipped in the sum rather than crashing
    the whole calculation — this slightly understates career time for
    a fighter with a genuine data gap on one old fight, which is an
    honest tradeoff: a slightly-low real number beats no number at all.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    fighter_id : int
    as_of_date : date

    Returns
    -------
    float — total seconds fought. 0.0 for a debutant.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)

    if prior_bouts.empty:
        return 0.0

    total_seconds = 0.0
    for _, bout in prior_bouts.iterrows():
        duration = get_fight_duration_seconds(
            bout["method"],
            bout["ending_round"],
            bout["ending_time_seconds"],
            bout["scheduled_rounds"],
        )
        if duration is not None:
            total_seconds += duration

    return total_seconds

def _rate_per_time_window(
    con: duckdb.DuckDBPyConnection,
    fighter_id: int,
    as_of_date: date,
    numerator_column: str,
    window_seconds: int,
) -> float | None:
    """
    Shared plumbing behind every Tier 2 "X per unit of fight time"
    feature (slpm, sapm, td_avg_per15, sub_avg_per15, ...). Each of
    those is really the same three steps — sum a column from
    get_prior_bout_stats, divide by total seconds fought, scale to
    the desired time window — differing only in WHICH column and
    WHICH window. This function is that shared machine; the public
    feature functions below are just the labeled buttons on top of it.

    Centralizing this means the "debutant -> None, not a fake 0.0"
    guard only has to be written correctly ONCE. Without this helper,
    that guard gets copy-pasted into every rate function separately —
    and copy-pasted logic is exactly the kind of thing that quietly
    drifts (someone fixes a bug in one copy, forgets the other four).

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    fighter_id : int
    as_of_date : date
    numerator_column : str
        Column name in get_prior_bout_stats' output to sum — e.g.
        "self_sig_strikes_landed", "opp_sig_strikes_landed",
        "self_takedowns_landed".
    window_seconds : int
        Size of the time window to scale to — 60 for "per minute",
        900 (15 * 60) for "per 15 minutes".

    Returns
    -------
    float | None — None if this fighter has 0 seconds of career fight
    time (a debutant), since a rate genuinely can't be computed from
    a zero denominator.
    """
    total_seconds = get_total_seconds_fought(con, fighter_id, as_of_date)
    if total_seconds == 0:
        return None

    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    total_events = prior_stats[numerator_column].sum()

    return total_events * window_seconds / total_seconds


def strikes_landed_per_minute(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Significant strikes landed per minute of actual fight time (SLpM)."""
    return _rate_per_time_window(
        con, fighter_id, as_of_date, "self_sig_strikes_landed", window_seconds=60
    )


def strikes_absorbed_per_minute(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Significant strikes ABSORBED per minute (SAPM) — uses
    opp_sig_strikes_landed, since "strikes my opponent landed" IS
    "strikes I absorbed." Mirror of strikes_landed_per_minute.
    """
    return _rate_per_time_window(
        con, fighter_id, as_of_date, "opp_sig_strikes_landed", window_seconds=60
    )


def takedowns_landed_per_15(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Takedowns landed per 15 minutes of fight time. 15-minute window
    (not per-minute) matches ufcstats.com's own "TD Avg" convention —
    takedowns are rare enough that per-minute would produce hard-to-
    read decimals like 0.02.
    """
    return _rate_per_time_window(
        con, fighter_id, as_of_date, "self_takedowns_landed", window_seconds=900
    )