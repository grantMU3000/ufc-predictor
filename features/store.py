"""
Assembles one row of raw features per bout — think of it like
building a trading card for a fight, with the red corner's stats on
one half, the blue corner's stats on the other half, and a few facts
stamped across the middle that belong to the fight itself (weight
class, title fight, rounds).

Scope note: this produces ONE row per bout, red_/blue_ prefixed —
NOT the symmetrized dual-row (self_/opp_, flipped labels, both
orderings) version described in docs/PLAN.md Section 0.2 / Week 2
Wednesday. That's a deliberate separation: symmetrization is a
transformation applied AFTER a row is correctly assembled, not
something tangled into assembly itself. Keeping them apart means a
bug in one is never confused for a bug in the other.
"""

import duckdb

from features.tier1 import (
    age_at_fight,
    height_at_fight,
    is_title_fight_at_bout,
    reach_at_fight,
    reach_to_height_ratio,
    scheduled_rounds_at_bout,
    stance_at_fight,
    stance_matchup,
    weight_class_at_bout,
)
from features.tier2 import (
    average_fight_time_seconds,
    career_win_percentage,
    control_time_percentage,
    days_since_last_fight,
    decision_loss_percentage,
    decision_win_percentage,
    finish_rate,
    knockdown_rate,
    ko_loss_rate,
    significant_strike_rate,
    strikes_absorbed_per_minute,
    strikes_landed_per_minute,
    striking_accuracy,
    striking_defense,
    striking_output_decay,
    sub_loss_rate,
    submission_success_rate,
    submission_win_count,
    submissions_attempted_per_15,
    takedown_accuracy,
    takedown_defense,
    takedown_output_decay,
    takedowns_landed_per_15,
    time_controlled_percentage,
    times_knocked_down,
    title_fight_experience,
    total_ufc_fights,
)

FIGHTER_FEATURES = [
    # Tier 1 — physical/static
    ("age", age_at_fight),
    ("height_cm", height_at_fight),
    ("reach_cm", reach_at_fight),
    ("reach_to_height_ratio", reach_to_height_ratio),
    ("stance", stance_at_fight),
    # Tier 2 — career rates, point-in-time
    ("slpm", strikes_landed_per_minute),
    ("sapm", strikes_absorbed_per_minute),
    ("td_avg_per_15", takedowns_landed_per_15),
    ("sub_attempts_per_15", submissions_attempted_per_15),
    ("striking_accuracy", striking_accuracy),
    ("takedown_accuracy", takedown_accuracy),
    ("significant_strike_rate", significant_strike_rate),
    ("striking_defense", striking_defense),
    ("takedown_defense", takedown_defense),
    ("career_win_pct", career_win_percentage),
    ("decision_win_pct", decision_win_percentage),
    ("decision_loss_pct", decision_loss_percentage),
    ("submission_win_count", submission_win_count),
    ("submission_success_rate", submission_success_rate),
    ("finish_rate", finish_rate),
    ("ko_loss_rate", ko_loss_rate),
    ("sub_loss_rate", sub_loss_rate),
    ("total_ufc_fights", total_ufc_fights),
    ("days_since_last_fight", days_since_last_fight),
    ("avg_fight_time_seconds", average_fight_time_seconds),
    ("title_fight_experience", title_fight_experience),
    ("times_knocked_down", times_knocked_down),
    ("knockdown_rate", knockdown_rate),
    ("control_time_pct", control_time_percentage),
    ("time_controlled_pct", time_controlled_percentage),
    ("striking_output_decay", striking_output_decay),
    ("takedown_output_decay", takedown_output_decay),
]

# Station 3: bout_id only, same value stamped for both corners.
BOUT_FEATURES = [
    ("weight_class", weight_class_at_bout),
    ("is_title_fight", is_title_fight_at_bout),
    ("scheduled_rounds", scheduled_rounds_at_bout),
]


def _get_bout_context(con: duckdb.DuckDBPyConnection, bout_id: int) -> dict:
    """
    Pulls the handful of facts every other function in this file
    needs before it can do anything: who's fighting, and when.

    Simple version: before you can fill out the trading card, you
    need to know which two fighters it's even for, and what date to
    stamp on it. This is that lookup, done once, so nothing else in
    this file has to repeat it.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int

    Returns
    -------
    dict with keys: fighter_red_id, fighter_blue_id, event_date.

    Raises
    ------
    ValueError if bout_id doesn't exist — fails loudly here rather
    than letting every downstream feature function quietly return
    None and produce a card built for a fight that doesn't exist.
    """
    query = """
        SELECT b.fighter_red_id, b.fighter_blue_id, e.event_date
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE b.id = $bout_id
    """
    result = con.execute(query, {"bout_id": bout_id}).fetchone()

    if result is None:
        raise ValueError(f"bout_id {bout_id} not found")

    fighter_red_id, fighter_blue_id, event_date = result
    return {
        "fighter_red_id": fighter_red_id,
        "fighter_blue_id": fighter_blue_id,
        "event_date": event_date,
    }


def build_feature_row(con: duckdb.DuckDBPyConnection, bout_id: int) -> dict:
    """
    Builds one raw feature row for a single bout: red corner's
    features, blue corner's features, and the bout-level facts that
    apply to both — all in one flat dict, ready to become one row of
    a pandas DataFrame.

    NOT yet symmetrized — this is red_/blue_, not the self_/opp_
    dual-row-with-flipped-labels version from the Week 2 Wednesday
    plan. That transformation happens to this function's OUTPUT,
    later, not inside this function.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int

    Returns
    -------
    dict — flat, one row's worth of features. Example shape:
        {
            "bout_id": 123,
            "event_date": date(2024, 6, 15),
            "red_age": 29.4, "blue_age": 31.1,
            "red_height_cm": 180.0, "blue_height_cm": 175.0,
            ...
            "red_stance_matchup_descriptive": "orthodox vs southpaw",
            "red_is_open_stance_matchup": True,
            "blue_stance_matchup_descriptive": "southpaw vs orthodox",
            "blue_is_open_stance_matchup": True,
            "weight_class": "Lightweight",
            "is_title_fight": False,
            "scheduled_rounds": 3,
        }
    """
    context = _get_bout_context(con, bout_id)
    fighter_red_id = context["fighter_red_id"]
    fighter_blue_id = context["fighter_blue_id"]
    as_of_date = context["event_date"]

    row = {"bout_id": bout_id, "event_date": as_of_date}

    # Station 1: same feature function, called once per corner.
    for corner_prefix, this_fighter_id in [
        ("red", fighter_red_id),
        ("blue", fighter_blue_id),
    ]:
        for fighter_feature_name, fighter_feature_fn in FIGHTER_FEATURES:
            row[f"{corner_prefix}_{fighter_feature_name}"] = fighter_feature_fn(
                con, this_fighter_id, as_of_date
            )

    # Station 2: stance_matchup needs BOTH IDs and cares who's
    # "self" — called once per corner, each time from that corner's
    # own point of view, so both halves of the card read correctly
    # from their own perspective.
    for corner_prefix, self_id, opp_id in [
        ("red", fighter_red_id, fighter_blue_id),
        ("blue", fighter_blue_id, fighter_red_id),
    ]:
        descriptive, is_open_stance = stance_matchup(con, self_id, opp_id, as_of_date)
        row[f"{corner_prefix}_stance_matchup_descriptive"] = descriptive
        row[f"{corner_prefix}_is_open_stance_matchup"] = is_open_stance

    # Station 3: bout-level facts, computed once, no corner prefix —
    # both fighters share the exact same weight class/title-fight
    # status/round count.
    for bout_feature_name, bout_feature_fn in BOUT_FEATURES:
        row[bout_feature_name] = bout_feature_fn(con, bout_id)

    return row