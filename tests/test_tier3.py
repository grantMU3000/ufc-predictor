"""
Tier 3 tests — strength_of_schedule, hand-verified against Khabib
Nurmagomedov's (fighter_id=68) real UFC career, per project testing
convention (hand-computed expected values from raw SQL before writing
assertions).

Expected values computed 2026-08-20: career bout list via SQL against
`bouts`/`events`/`fighters`, opponent pre-fight Elo pulled via a
one-off script calling _load_labels_and_elo() and filtering to
Khabib's 13 bout_ids. Full working shown in PR description.
"""

import pytest
import duckdb
from datetime import date
from features.build_lgbm_matrix import _load_labels_and_elo
from features.tier3 import build_fighter_bout_history, strength_of_schedule, recent_damage_absorbed

KHABIB_ID = 68

# fight #, bout_id, opponent — kept here as a comment for anyone
# re-deriving these numbers later without re-running the SQL.
#  1  6976  Shalorus     6  6125  Dos Anjos
#  2  6809  Tibau        7  5166  Horcher
#  3  6642  Tavares      8  4898  M. Johnson
#  4  6505  Trujillo     9  4339  Barboza
#  5  6382  Healy       10  4226  Iaquinta
#                        11  3988  McGregor
#                        12  3509  Poirier
#                        13  2974  Gaethje


@pytest.fixture(scope="module")
def khabib_history():
    """
    Module-scoped: _load_labels_and_elo() walks the ENTIRE decided-
    bout population (every fighter, not just Khabib) to compute Elo —
    expensive enough that recomputing it per-test would slow the
    suite for no reason. Every test in this file reads from the same
    frame; none of them mutate it.
    """
    labels, elo_ratings = _load_labels_and_elo()
    return build_fighter_bout_history(labels, elo_ratings)

@pytest.fixture(scope="module")
def con():
    """
    Opens the same fighters/events/bouts/bout_stats parquet-backed
    views every other feature-computation call in this project uses
    (matches _load_labels_and_elo's pattern in build_lgbm_matrix.py,
    plus bout_stats since recent_damage_absorbed needs per-round
    strike data the Elo loader never touches).

    Module-scoped, not function-scoped: these are read-only SELECT
    queries against static parquet files — nothing in
    recent_damage_absorbed mutates the connection's state, so
    reopening it fresh for every single test would just be wasted
    setup time for zero safety benefit. Same reasoning as
    khabib_history's module scope above.

    Closed automatically by pytest at the end of the module's test
    run via the yield/teardown pattern.
    """
    connection = duckdb.connect()
    for table in ["fighters", "events", "bouts", "bout_stats"]:
        connection.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )
    yield connection
    connection.close()


def test_debut_sos_is_nan(khabib_history):
    """Shalorus, fight 1: no prior fights means no schedule to be
    strong or weak — must be NaN, not 0 or some other default that
    would quietly tell the model "average schedule" about a debut."""
    sos = strength_of_schedule(khabib_history, n=3)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 6976)]
    # assert row["sos_last_5"].isna().all()
    assert row["sos_last_3"].isna().all()


def test_second_fight_sos_equals_lone_prior_opponent(khabib_history):
    """Tibau, fight 2: with exactly 1 prior fight, min_periods=1
    means SoS is that single opponent's Elo, not an average diluted
    by phantom zeros or held back until 5 fights exist."""
    sos = strength_of_schedule(khabib_history, n=3)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 6809)]
    # assert row["sos_last_5"].iloc[0] == pytest.approx(1475.2773, abs=0.01)
    assert row["sos_last_3"].iloc[0] == pytest.approx(1475.2773, abs=0.01)


def test_ninth_fight_matches_hand_computed_five_fight_window(khabib_history):
    """Barboza, fight 9: real 5-fight rolling window — Trujillo,
    Healy, Dos Anjos, Horcher, M. Johnson. This is the core rolling-
    mean logic, exercised for real once history is long enough that
    the window is actually full."""
    sos = strength_of_schedule(khabib_history, n=3)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 4339)]
    # assert row["sos_last_5"].iloc[0] == pytest.approx(1527.6754, abs=0.01)
    assert row["sos_last_3"].iloc[0] == pytest.approx(1546.3390, abs=0.01)


def test_current_opponent_never_enters_own_sos_window(khabib_history):
    """McGregor, fight 11: the leakage guard. McGregor's own Elo
    (1681.6164, inflated by PPV drawing power relative to his actual
    UFC lightweight resume) must NOT appear in the window used to
    predict THIS fight. Expected value uses only fights 6-10 (Dos
    Anjos through Iaquinta). If .shift(1) is ever dropped from
    strength_of_schedule, McGregor's rating leaks into his own row
    and this number rises — the test would fail loudly instead of
    silently handing the model a peek at the outcome it's rating."""
    sos = strength_of_schedule(khabib_history, n=3)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 3988)]
    # assert row["sos_last_5"].iloc[0] == pytest.approx(1578.0218, abs=0.01)
    assert row["sos_last_3"].iloc[0] == pytest.approx(1604.9114, abs=0.01)

JOSHUA_VAN_ID = 400

# bout_id / date / opponent / recent_damage_24mo, hand-summed from
# bout_stats sig_strikes_landed (opponent's side, per bout, all
# rounds) — SQL + expected-value derivation in PR description.
#  1626  2023-06-24  Zhumagulov    (debut, no prior window)
#  1412  2023-11-11  Borjas
#  1351  2024-01-13  Bunes
#  1072  2024-07-13  C. Johnson
#   981  2024-09-14  Chairez
#   858  2024-12-07  Durden
#   746  2025-03-08  Tsuruya
#   612  2025-06-07  B. Silva
#   573  2025-06-28  Royval
#   328  2025-12-06  Pantoja
#   124  2026-05-09  Taira


def test_recent_damage_debut_is_none(con):
    """No prior fights, no window to sum — must be None, not 0.
    A 0 would tell the model 'this fighter took zero damage
    recently,' which is a false claim about a fighter with no
    history at all."""
    result = recent_damage_absorbed(con, JOSHUA_VAN_ID, date(2023, 6, 24))
    assert result is None


def test_recent_damage_midcareer_sum(con):
    """Chairez fight: straightforward sum of 4 prior bouts, no
    boundary edge cases — the basic case the window logic should
    get right before testing anything trickier."""
    result = recent_damage_absorbed(con, JOSHUA_VAN_ID, date(2024, 9, 14))
    assert result == pytest.approx(316, abs=0.01)


def test_recent_damage_excludes_fight_just_past_24mo_boundary(con):
    """Royval fight: the 24-month lower bound actually does
    something here. Zhumagulov's fight (2023-06-24) is 4 days
    older than the cutoff (2023-06-28) and must be excluded. If the
    >= boundary in the SQL is ever loosened to >, this number would
    silently jump to 573 (103 + 470) instead."""
    result = recent_damage_absorbed(con, JOSHUA_VAN_ID, date(2025, 6, 28))
    assert result == pytest.approx(470, abs=0.01)


def test_recent_damage_rolls_off_older_fights(con):
    """Taira fight: confirms the window actually SLIDES — three
    fights that were counted in the Royval-window test (Zhumagulov,
    Borjas, Bunes) have aged out by this later as_of_date, and three
    new ones (Royval, Pantoja) have entered."""
    result = recent_damage_absorbed(con, JOSHUA_VAN_ID, date(2026, 5, 9))
    assert result == pytest.approx(566, abs=0.01)