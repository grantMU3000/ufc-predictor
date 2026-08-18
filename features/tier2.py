"""
Tier 2 features: point-in-time career RATES, computed only from a
fighter's fights strictly before the one being predicted. Everything
here builds on bout_history.py / bout_stats_history.py — never
queries bouts/bout_stats directly, so the "only look at the past"
rule only has to be gotten right in one place.
"""

from datetime import date

import duckdb
import pandas as pd

from features.bout_history import get_prior_bouts
from features.bout_stats_history import get_prior_bout_stats

ROUND_LENGTH_SECONDS = 300  # 5-minute UFC rounds. Safe across this
# entire dataset — pre-Unified-Rules-era bouts (which used different
# round lengths) are already excluded at ingestion (PLAN_ADDENDUM §7),
# so every bout this function will ever see follows the modern format.


def _is_decision_method(method: str | None) -> bool:
    """
    True if a bout's method string represents a decision (any of
    "Decision - Unanimous/Split/Majority"), regardless of who won or
    whether it was a draw. Pulled out as its own function since both
    get_fight_duration_seconds and this batch's decision_win/loss
    features need the exact same check — one place to get it right,
    not two copies that could quietly drift apart later.
    """
    return method is not None and method.strip().lower().startswith("decision")


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
    if _is_decision_method(method):
        return float(scheduled_rounds * ROUND_LENGTH_SECONDS)

    if ending_round is None or ending_time_seconds is None:
        return None  # genuine data gap on this one bout

    return float((ending_round - 1) * ROUND_LENGTH_SECONDS + ending_time_seconds)


def _get_prior_fight_durations(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> list[float]:
    """
    Per-fight durations (seconds) for every prior bout with complete
    finish data — the shared list behind both get_total_seconds_fought
    (sums it) and average_fight_time_seconds (averages it). Bouts
    with incomplete finish data are skipped, not zero-filled — same
    "understate slightly rather than fabricate" tradeoff as before.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    durations = []
    for _, bout in prior_bouts.iterrows():
        duration = get_fight_duration_seconds(
            bout["method"],
            bout["ending_round"],
            bout["ending_time_seconds"],
            bout["scheduled_rounds"],
        )
        if duration is not None:
            durations.append(duration)
    return durations


def get_total_seconds_fought(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float:
    """Total seconds fought across all prior bouts. 0.0 for a debutant — see original docstring for why that's a true zero, not a gap."""
    return sum(_get_prior_fight_durations(con, fighter_id, as_of_date), 0.0)


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


def submissions_attempted_per_15(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Submission attempts per 15 minutes of fight time."""
    return _rate_per_time_window(
        con, fighter_id, as_of_date, "self_sub_attempts", window_seconds=900
    )


def _column_ratio(
    con: duckdb.DuckDBPyConnection,
    fighter_id: int,
    as_of_date: date,
    numerator_column: str,
    denominator_column: str,
) -> float | None:
    """
    Shared plumbing behind every Tier 2 "one stat as a fraction of
    another" feature (str_acc, td_acc, sig_strike_rate, ...). Sums
    two columns from get_prior_bout_stats and divides. Deliberately
    more general than "landed / attempted" — sig_strike_rate divides
    landed by landed (two different columns, not an attempt count),
    so this helper doesn't assume anything about what the two columns
    mean, only that one gets summed and divided by the other.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    fighter_id : int
    as_of_date : date
    numerator_column : str
        Column in get_prior_bout_stats' output to sum for the top.
    denominator_column : str
        Column to sum for the bottom.

    Returns
    -------
    float | None — None if the denominator sums to 0 (either a
    debutant with no bout_stats rows at all, or the rarer case of a
    fighter whose recorded fights show zero attempts in this
    category — either way, a real ratio can't be computed).
    """
    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)

    denominator_total = prior_stats[denominator_column].sum()
    if denominator_total == 0:
        return None

    numerator_total = prior_stats[numerator_column].sum()
    return numerator_total / denominator_total


def striking_accuracy(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Significant strikes landed ÷ attempted — this fighter's own striking accuracy."""
    return _column_ratio(
        con,
        fighter_id,
        as_of_date,
        "self_sig_strikes_landed",
        "self_sig_strikes_attempted",
    )


def takedown_accuracy(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Takedowns landed ÷ attempted — this fighter's own takedown accuracy."""
    return _column_ratio(
        con, fighter_id, as_of_date, "self_takedowns_landed", "self_takedowns_attempted"
    )


def significant_strike_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    What fraction of this fighter's landed strikes counted as
    "significant" (meaningful/damage-relevant) rather than just
    total volume. Distinct from striking_accuracy — accuracy asks
    "landed vs. attempted," this asks "of what landed, how much
    actually mattered." A fighter with a lot of low-impact clinch/
    grappling strikes will show a lower rate here than a clean
    striker, even with similar overall accuracy.
    """
    return _column_ratio(
        con,
        fighter_id,
        as_of_date,
        "self_sig_strikes_landed",
        "self_total_strikes_landed",
    )


def striking_defense(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of opponents' significant strikes that did NOT land — i.e. this
    fighter's ability to avoid damage. Computed as 1 minus the
    OPPONENT's striking accuracy against this fighter (opp_ columns),
    not anything from self_. Defense is inherently about what the
    other fighter failed to do, not a self-stat.

    None if opponents combined never attempted a significant strike
    against this fighter (only realistic for a debutant) — same
    "can't compute a ratio from 0" guard as _column_ratio.
    """
    opponent_accuracy = _column_ratio(
        con,
        fighter_id,
        as_of_date,
        "opp_sig_strikes_landed",
        "opp_sig_strikes_attempted",
    )
    if opponent_accuracy is None:
        return None
    return 1 - opponent_accuracy


def takedown_defense(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of opponents' takedown attempts that did NOT land — this
    fighter's takedown defense. Same "invert the opponent's own rate"
    logic as striking_defense.
    """
    opponent_accuracy = _column_ratio(
        con, fighter_id, as_of_date, "opp_takedowns_landed", "opp_takedowns_attempted"
    )
    if opponent_accuracy is None:
        return None
    return 1 - opponent_accuracy


def _decided_bouts(prior_bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Filters get_prior_bouts' output down to fights with a real
    winner — drops no-contests AND draws (both show self_won as None,
    since winner_id is NULL for either). A draw is neither a win nor
    a loss, so it shouldn't silently deflate a win/loss rate by
    sitting in the denominator without ever matching either side —
    same spirit as ADR-003 excluding draws/NCs from training labels,
    applied here to a fighter's own career-rate features instead.

    DQ results are unaffected by this filter — they still have a
    real winner_id (whoever wasn't disqualified), so they pass
    through as a normal decided bout.
    """
    return prior_bouts[prior_bouts["self_won"].notna()]


def career_win_percentage(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of decided prior UFC bouts this fighter won. Draws/no-contests
    excluded from both numerator and denominator (see _decided_bouts)
    — a draw isn't a loss, so it shouldn't count against this number.

    None if this fighter has zero decided prior bouts (a debutant, or
    a fighter whose only prior fights were draws/no-contests).
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    decided = _decided_bouts(prior_bouts)

    if decided.empty:
        return None

    win_count = int(decided["self_won"].astype(bool).sum())
    return win_count / len(decided)


def decision_win_percentage(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of decided prior bouts this fighter won BY DECISION specifically
    (not overall win rate) — a fighter who wins mostly by decision is
    a meaningfully different profile than one who wins mostly by
    finish, even at the same overall win percentage.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    decided = _decided_bouts(prior_bouts)

    if decided.empty:
        return None

    is_decision_win = decided["self_won"].astype(bool) & decided["method"].apply(
        _is_decision_method
    )
    return int(is_decision_win.sum()) / len(decided)


def decision_loss_percentage(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of decided prior bouts this fighter LOST by decision — pairs
    with decision_win_percentage. A fighter who loses a lot of close
    decisions is a different risk profile than one who gets finished.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    decided = _decided_bouts(prior_bouts)

    if decided.empty:
        return None

    is_decision_loss = (~decided["self_won"].astype(bool)) & decided["method"].apply(
        _is_decision_method
    )
    return int(is_decision_loss.sum()) / len(decided)


def submission_win_count(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> int:
    """
    Total career wins by submission. A raw count, not a rate — pairs
    with submission_success_rate below as the numerator.

    Returns 0 for a debutant, not None — same "true zero, not a
    gap" reasoning as get_total_seconds_fought. A fighter with no
    submission wins genuinely has zero, whether that's because they
    haven't fought yet or because they've fought and never finished
    one that way.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    if prior_bouts.empty:
        return 0

    is_submission_win = prior_bouts["self_won"].astype("boolean") & (
        prior_bouts["method"] == "Submission"
    )
    return int(is_submission_win.sum())


def submission_success_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Successful submissions ÷ submission ATTEMPTS. Note this one
    crosses tables — the numerator (submission_win_count) comes from
    get_prior_bouts (outcomes), the denominator comes from
    get_prior_bout_stats (self_sub_attempts, summed). Every other
    function in this batch stays within one table; this is the
    exception, worth remembering if this number ever looks off and
    you're hunting for why.

    None if this fighter has zero recorded submission attempts —
    can't compute a success rate with nothing attempted.
    """
    win_count = submission_win_count(con, fighter_id, as_of_date)

    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    if prior_stats.empty:
        return None

    total_attempts = prior_stats["self_sub_attempts"].sum()
    if total_attempts == 0:
        return None

    return win_count / total_attempts


def _wins_only(prior_bouts: pd.DataFrame) -> pd.DataFrame:
    """Decided prior bouts this fighter WON (see _decided_bouts) — includes DQ wins."""
    decided = _decided_bouts(prior_bouts)
    return decided[decided["self_won"].astype(bool)]


def _non_dq_losses(prior_bouts: pd.DataFrame) -> pd.DataFrame:
    """
    Decided prior bouts this fighter LOST, excluding DQ losses. A
    loss by disqualification (illegal strike, weight miss, etc.)
    isn't a durability or finishing-vulnerability signal, so it's
    excluded from the denominator of ko_loss_rate/sub_loss_rate
    specifically — those two exist to measure HOW a fighter tends to
    actually lose fights, and a DQ isn't really "losing a fight" in
    that sense.
    """
    decided = _decided_bouts(prior_bouts)
    losses = decided[~decided["self_won"].astype(bool)]
    return losses[losses["method"] != "DQ"]


def finish_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of this fighter's wins that came by finish (KO/TKO, Submission,
    Could Not Continue, TKO - Doctor's Stoppage) rather than decision.

    DQ wins are excluded from the NUMERATOR only (a DQ win isn't
    really "finishing" an opponent — your opponent didn't get beaten,
    they got disqualified) but still count in the denominator, since
    it's still a real win. Could Not Continue / TKO - Doctor's
    Stoppage wins DO count as finishes here — the ambiguity you
    flagged for those methods was specifically about the LOSING
    fighter's chin/durability (a stoppage might reflect an unrelated
    injury, not damage taken), which doesn't apply from the winning
    fighter's side — a win by those methods still means this fighter
    ended the fight early.

    None if this fighter has zero prior wins.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    wins = _wins_only(prior_bouts)

    if wins.empty:
        return None

    is_finish = ~wins["method"].apply(_is_decision_method) & (wins["method"] != "DQ")
    return int(is_finish.sum()) / len(wins)


def ko_loss_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of this fighter's non-DQ losses that came by KO/TKO
    specifically. Strictly method == "KO/TKO" — Could Not Continue
    and TKO - Doctor's Stoppage are deliberately NOT counted here
    (deferred to a future cuts/injury feature per your call), since
    those can reflect an unrelated injury or accidental cut rather
    than genuine damage-taken durability, and folding them in would
    contaminate this as a chin proxy.

    None if this fighter has zero non-DQ prior losses.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    losses = _non_dq_losses(prior_bouts)

    if losses.empty:
        return None

    is_ko_loss = losses["method"] == "KO/TKO"
    return int(is_ko_loss.sum()) / len(losses)


def sub_loss_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    % of this fighter's non-DQ losses that came by submission.
    Same denominator (_non_dq_losses) as ko_loss_rate, so the two
    are directly comparable — e.g. a fighter with ko_loss_rate=0.6
    and sub_loss_rate=0.1 has a clear "how they tend to lose" profile.

    None if this fighter has zero non-DQ prior losses.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    losses = _non_dq_losses(prior_bouts)

    if losses.empty:
        return None

    is_sub_loss = losses["method"] == "Submission"
    return int(is_sub_loss.sum()) / len(losses)


def total_ufc_fights(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> int:
    """
    Total prior UFC bouts, period — draws, no-contests, DQs all
    count. This measures EXPERIENCE, not record, so nothing gets
    filtered out the way _decided_bouts filters for win-rate features.
    0 for a debutant (a true, honest zero).
    """
    return len(get_prior_bouts(con, fighter_id, as_of_date))


def days_since_last_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> int | None:
    """
    Days since this fighter's most recent prior bout (layoff).
    get_prior_bouts is sorted oldest-first, so the last row is the
    most recent fight. None for a debutant — unlike total_seconds_
    fought, there's no honest "zero" here; layoff is genuinely
    undefined without a prior fight to measure from.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    if prior_bouts.empty:
        return None
    last_fight_date = prior_bouts.iloc[-1]["event_date"].date()
    return (as_of_date - last_fight_date).days


def average_fight_time_seconds(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Average duration (seconds) of this fighter's prior fights. None
    for a debutant, or if every prior bout had incomplete finish data.
    """
    durations = _get_prior_fight_durations(con, fighter_id, as_of_date)
    if not durations:
        return None
    return sum(durations) / len(durations)


def title_fight_experience(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> int:
    """
    Count of prior bouts that were title fights (not a boolean — a
    fighter with 3 title fights is a different case than one with 1).
    0 for a debutant. is_title_fight is NOT NULL at the DB level, so
    no missing-data handling needed here.
    """
    prior_bouts = get_prior_bouts(con, fighter_id, as_of_date)
    if prior_bouts.empty:
        return 0
    return int(prior_bouts["is_title_fight"].sum())


def times_knocked_down(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> int:
    """
    Total times this fighter has been knocked down (opp_knockdowns —
    "knockdowns MY OPPONENT scored" IS "times I got dropped," per
    the durability-proxy discussion). 0 for a debutant.
    """
    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    if prior_stats.empty:
        return 0
    return int(prior_stats["opp_knockdowns"].sum())


def knockdown_rate(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Knockdowns SCORED per 15 minutes of fight time — reuses _rate_per_time_window, self_knockdowns as numerator."""
    return _rate_per_time_window(
        con, fighter_id, as_of_date, "self_knockdowns", window_seconds=900
    )


def _time_percentage(
    con: duckdb.DuckDBPyConnection,
    fighter_id: int,
    as_of_date: date,
    seconds_column: str,
) -> float | None:
    """Shared plumbing for control_time_percentage/time_controlled_percentage — a column of seconds, divided by total seconds fought. None if the fighter has 0 seconds fought."""
    total_seconds = get_total_seconds_fought(con, fighter_id, as_of_date)
    if total_seconds == 0:
        return None
    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    return prior_stats[seconds_column].sum() / total_seconds


def control_time_percentage(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """% of fight time this fighter spent CONTROLLING opponents (self_control_time_seconds)."""
    return _time_percentage(con, fighter_id, as_of_date, "self_control_time_seconds")


def time_controlled_percentage(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """% of fight time this fighter spent BEING CONTROLLED (opp_control_time_seconds)."""
    return _time_percentage(con, fighter_id, as_of_date, "opp_control_time_seconds")


def _round_split_decay(prior_stats: pd.DataFrame, column_name: str) -> float | None:
    """
    Shared engine behind every "does this fighter's output drop off
    in later rounds" feature. Splits a fighter's round-by-round stats
    into early (rounds 1-2) and late (rounds 3+) buckets, and returns
    late-average minus early-average for the given column. Negative =
    fades late. Positive = holds steady or ramps up.

    Kept as ONE function so the early/late round threshold only has
    to be decided in one place — if striking_output_decay and
    takedown_output_decay each redefined "early vs late" separately,
    a future tweak to one could quietly drift from the other.

    KNOWN SIMPLIFICATION (applies to every column this is used with):
    compares per-ROUND averages, not per-exact-minute. Every round is
    a full 5 min except a fight's final round if it ends by finish,
    which is partial. If that partial round lands in the late bucket,
    it slightly understates late-round output for that column — real
    decay may look a bit more pronounced than fully time-corrected
    math would show. Flagged, not fixed, here.

    Parameters
    ----------
    prior_stats : pd.DataFrame
        Output of get_prior_bout_stats for one fighter.
    column_name : str
        Which self_* column to compare early vs. late.

    Returns
    -------
    float | None — None if no prior fight reached round 3+.
    """
    early_rounds = prior_stats[prior_stats["round_number"] <= 2]
    late_rounds = prior_stats[prior_stats["round_number"] >= 3]

    if late_rounds.empty:
        return None

    return late_rounds[column_name].mean() - early_rounds[column_name].mean()


def striking_output_decay(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Late-round minus early-round significant strikes landed per round. See _round_split_decay for the shared logic and its known simplification."""
    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    if prior_stats.empty:
        return None
    return _round_split_decay(prior_stats, "self_sig_strikes_landed")


def takedown_output_decay(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Late-round minus early-round takedowns landed per round — a grappling-specific fade signal, distinct from striking_output_decay."""
    prior_stats = get_prior_bout_stats(con, fighter_id, as_of_date)
    if prior_stats.empty:
        return None
    return _round_split_decay(prior_stats, "self_takedowns_landed")
