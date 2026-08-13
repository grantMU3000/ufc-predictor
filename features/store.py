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

from datetime import date

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

# Station 1: same fighter_id/as_of_date shape every time. Called once
# per corner (red, then blue). Adding a new Tier 1 fighter-level
# feature later means adding ONE line here — nothing else changes.
FIGHTER_FEATURES = [
    ("age_at_fight", age_at_fight),
    ("height_cm", height_at_fight),
    ("reach_cm", reach_at_fight),
    ("reach_to_height_ratio", reach_to_height_ratio),
    ("stance", stance_at_fight),
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
        for feature_name, feature_fn in FIGHTER_FEATURES:
            row[f"{corner_prefix}_{feature_name}"] = feature_fn(
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
    for feature_name, feature_fn in BOUT_FEATURES:
        row[feature_name] = feature_fn(con, bout_id)

    return row