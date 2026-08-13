"""
Tier 1 features: static/physical stuff you could know about a fighter
just by looking at them, before they've thrown a single punch. No
fight history needed here — that's Tier 2's job (uses bout_history.py
/ bout_stats_history.py instead).

Every function in this file follows the same shape:
    feature(con, fighter_id, as_of_date) -> value
so the feature store (store.py) can call any of them the same way,
without needing to know what's happening inside.
"""

from datetime import date

import duckdb


_FIGHTER_COLUMNS = {"dob", "height_cm", "reach_cm", "stance"}


def _fetch_fighter_column(
    con: duckdb.DuckDBPyConnection, fighter_id: int, column_name: str
):
    """
    Shared plumbing behind every Tier 1 feature that's just "look up
    one column on the fighters table." Not a feature function itself
    — the leading underscore is the same signal as a "staff only"
    door: store.py should never call this directly, only the real
    feature functions below should.

    column_name is restricted to a fixed whitelist (_FIGHTER_COLUMNS)
    instead of accepted as free text. Two reasons: a typo like
    "hieght_cm" turns into a clear error right here instead of a
    confusing DuckDB failure buried in a query string, and — since
    this value gets pasted directly into SQL — it closes off SQL
    injection as a concern even though every caller today is our own
    hardcoded code, never user input.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    fighter_id : int
    column_name : str
        Must be one of _FIGHTER_COLUMNS.

    Returns
    -------
    The raw column value, or None if the fighter doesn't exist or the
    column is NULL. Callers convert to the right type themselves
    (float(), etc.) — this function's only job is fetching.
    """
    if column_name not in _FIGHTER_COLUMNS:
        raise ValueError(f"Unexpected fighters column: {column_name!r}")

    query = f"SELECT {column_name} FROM fighters WHERE id = $fighter_id"
    result = con.execute(query, {"fighter_id": fighter_id}).fetchone()

    if result is None or result[0] is None:
        return None

    return result[0]


def age_at_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Fighter's age, in years, on the date of the fight. 365.25 (not
    365) accounts for leap years. Missing dob returns None — see
    module-level reasoning: LightGBM learns from real gaps, a guessed
    average would hide one.
    """
    dob = _fetch_fighter_column(con, fighter_id, "dob")
    if dob is None:
        return None
    return (as_of_date - dob).days / 365.25


def height_at_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date  # noqa: ARG001
) -> float | None:
    """Fighter's height in cm. Not date-dependent; as_of_date kept for signature consistency."""
    value = _fetch_fighter_column(con, fighter_id, "height_cm")
    return float(value) if value is not None else None


def reach_at_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date  # noqa: ARG001
) -> float | None:
    """Fighter's reach in cm. Not date-dependent; as_of_date kept for signature consistency."""
    value = _fetch_fighter_column(con, fighter_id, "reach_cm")
    return float(value) if value is not None else None


def reach_to_height_ratio(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """Reach / height. None if either is unknown — see prior discussion."""
    height = height_at_fight(con, fighter_id, as_of_date)
    reach = reach_at_fight(con, fighter_id, as_of_date)
    return reach / height if height is not None and reach is not None else None


def stance_at_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date  # noqa: ARG001
) -> str | None:
    """
    Fighter's fighting stance ("Orthodox", "Southpaw", "Switch", and
    at least two more values seen in real data — see stance_matchup
    below for why that matters).
    """
    return _fetch_fighter_column(con, fighter_id, "stance")


_OPEN_STANCE_PAIRS = {
    frozenset({"orthodox", "southpaw"}),
}


def stance_matchup(
    con: duckdb.DuckDBPyConnection, fighter_id: int, opponent_id: int, as_of_date: date
) -> tuple[str | None, bool | None]:
    """
    Describes how this fighter's stance lines up against their
    opponent's, from self's perspective.

    Breaks the single-fighter signature every other Tier 1 function
    uses — this is inherently a two-fighter (bout-level) question,
    not a one-fighter question, so it needs opponent_id. store.py's
    assembly logic will need to call this one differently from the
    rest (both corners' IDs from the bout), not loop fighter-by-fighter.

    Returns a (descriptive, is_open_stance) pair:
      - descriptive: "Orthodox vs Southpaw" — self's stance first, so
        it flips consistently on the symmetrized dual row (see
        docs/DECISIONS.md ADR-004). Handles any stance value found in
        the data, including the unconstrained "Open Stance"/
        "Sideways" values noted in DECISIONS.md — not just the three
        you'd expect.
      - is_open_stance: True only for a clean orthodox/southpaw
        mismatch (the tactically meaningful case in MMA — different
        lead hands change which angles are open). None if either
        stance is missing OR is one of the non-standard values, since
        "open stance" isn't a well-defined concept for those.

    Returns (None, None) if either fighter's stance is unknown.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    fighter_id : int
        The "self" fighter — appears first in the descriptive string.
    opponent_id : int
        The fighter self is facing in this bout.
    as_of_date : date
        Passed through to stance_at_fight for signature consistency.

    Returns
    -------
    tuple[str | None, bool | None]
    """
    self_stance = stance_at_fight(con, fighter_id, as_of_date)
    opp_stance = stance_at_fight(con, opponent_id, as_of_date)

    if self_stance is None or opp_stance is None:
        return None, None

    descriptive = f"{self_stance} vs {opp_stance}"
    is_open_stance = frozenset({self_stance, opp_stance}) in _OPEN_STANCE_PAIRS

    return descriptive, is_open_stance

_BOUT_COLUMNS = {"weight_class", "is_title_fight", "scheduled_rounds"}


def _fetch_bout_column(con: duckdb.DuckDBPyConnection, bout_id: int, column_name: str):
    """
    Shared plumbing behind every Tier 1 feature that's a fact about
    the BOUT itself, not about either fighter — weight_class,
    is_title_fight, scheduled_rounds. Mirrors _fetch_fighter_column
    above, just pointed at a different table.

    Why this is a SEPARATE helper from _fetch_fighter_column, rather
    than one generic "fetch any column from any table" function:
    these two helpers answer fundamentally different questions
    (fighter_id -> fact about a person, vs. bout_id -> fact about a
    specific matchup). Merging them into one "fetch anything" helper
    would make every call site less readable — you'd have to check
    which table name got passed in to know what's actually being
    asked. Two small, clearly-named helpers beat one clever generic
    one here.

    Same whitelist safety reasoning as _fetch_fighter_column: a typo
    fails loudly right here, and string interpolation into SQL never
    touches anything outside a fixed, known set of column names.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int
    column_name : str
        Must be one of _BOUT_COLUMNS.

    Returns
    -------
    The raw column value, or None if the bout doesn't exist or the
    column is NULL.
    """
    if column_name not in _BOUT_COLUMNS:
        raise ValueError(f"Unexpected bouts column: {column_name!r}")

    query = f"SELECT {column_name} FROM bouts WHERE id = $bout_id"
    result = con.execute(query, {"bout_id": bout_id}).fetchone()

    if result is None or result[0] is None:
        return None

    return result[0]


def weight_class_at_bout(con: duckdb.DuckDBPyConnection, bout_id: int) -> str | None:
    """
    The weight class this specific bout was contested at — e.g.
    "Lightweight", "Welterweight".

    Deliberately takes bout_id, NOT (fighter_id, as_of_date) like
    every other Tier 1 function so far. This isn't really a property
    of a fighter at all — it's a property of the matchup itself,
    decided in advance as part of how the fight was booked. Both
    fighters in the bout share the exact same answer, so there's
    nothing to look up per-fighter, and no "as of" cutoff needed:
    the weight class of the fight being predicted was never a
    "future" fact in the leakage sense — it's a known part of the
    matchup itself, the same as scheduled_rounds/is_title_fight
    below.

    store.py's assembly step needs to call this differently from
    per-fighter features: once per bout, not once per corner, then
    applied to both sides of the training row.

    Schema note: bouts.weight_class is NOT NULL (see
    38e3db2c80a5_create_initial_schema.py), so a None return here
    would mean the bout_id itself doesn't exist — not a real gap in
    weight-class data.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int

    Returns
    -------
    str | None
    """
    return _fetch_bout_column(con, bout_id, "weight_class")


def is_title_fight_at_bout(con: duckdb.DuckDBPyConnection, bout_id: int) -> bool | None:
    """
    Whether this bout was a title fight. Same bout-level reasoning as
    weight_class_at_bout — a fact about the matchup, not about either
    fighter individually.

    Schema note: NOT NULL with a false default (see
    38e3db2c80a5_create_initial_schema.py) — every real bout has a
    real True/False here. A None return means the bout_id itself
    wasn't found.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int

    Returns
    -------
    bool | None
    """
    return _fetch_bout_column(con, bout_id, "is_title_fight")


def scheduled_rounds_at_bout(con: duckdb.DuckDBPyConnection, bout_id: int) -> int | None:
    """
    How many rounds this bout was scheduled for (3 or 5). Same
    bout-level reasoning as weight_class_at_bout.

    Schema note: NOT NULL, constrained to exactly 3 or 5 at the DB
    level (ck_bouts_scheduled_rounds). A None return means the
    bout_id itself wasn't found, never a data gap.

    Worth remembering when this value gets used downstream: per
    ADR-012, ~70 bouts are 3-round title fights (TUF/Road to UFC
    finales, plus one historical exception) — this function just
    reports what's actually stored, which is already correct for
    those rows. Nothing extra to handle here; the correction already
    happened at the data layer.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
    bout_id : int

    Returns
    -------
    int | None
    """
    return _fetch_bout_column(con, bout_id, "scheduled_rounds")