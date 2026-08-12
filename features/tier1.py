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


def age_at_fight(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date: date
) -> float | None:
    """
    Fighter's age, in years, on the date of the fight.

    Simple version: take the fight date, take the birthday, subtract.
    365.25 (not 365) accounts for leap years so age doesn't drift
    slightly wrong over a fighter's career.

    Missing date_of_birth returns None (pandas will show this as NaN),
    NOT a guessed average age. Guessing a number like "27" would
    quietly lie to the model — it couldn't tell a real 27-year-old
    apart from "we don't actually know." LightGBM (the primary model)
    handles missing values natively and can learn something useful
    from the fact that it's missing, which a fake number would erase.
    If a model that can't handle NaN (the logistic regression
    baseline) needs this filled in later, that's a decision made
    explicitly in that baseline's own prep code — never here.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to the local Parquet snapshot.
    fighter_id : int
        Fighter to compute age for.
    as_of_date : date
        The fight date.

    Returns
    -------
    float | None — age in years (e.g. 27.4), or None if date_of_birth
    is missing for this fighter.
    """
    result = con.execute(
        "SELECT dob FROM fighters WHERE id = $fighter_id",
        {"fighter_id": fighter_id},
    ).fetchone()

    if result is None or result[0] is None:
        return None

    dob = result[0]
    return (as_of_date - dob).days / 365.25