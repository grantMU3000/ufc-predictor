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

from features.build_lgbm_matrix import _load_labels_and_elo
from features.tier3 import build_fighter_bout_history, strength_of_schedule

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


def test_debut_sos_is_nan(khabib_history):
    """Shalorus, fight 1: no prior fights means no schedule to be
    strong or weak — must be NaN, not 0 or some other default that
    would quietly tell the model "average schedule" about a debut."""
    sos = strength_of_schedule(khabib_history, n=5)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 6976)]
    assert row["sos_last_5"].isna().all()


def test_second_fight_sos_equals_lone_prior_opponent(khabib_history):
    """Tibau, fight 2: with exactly 1 prior fight, min_periods=1
    means SoS is that single opponent's Elo, not an average diluted
    by phantom zeros or held back until 5 fights exist."""
    sos = strength_of_schedule(khabib_history, n=5)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 6809)]
    assert row["sos_last_5"].iloc[0] == pytest.approx(1475.2773, abs=0.01)


def test_ninth_fight_matches_hand_computed_five_fight_window(khabib_history):
    """Barboza, fight 9: real 5-fight rolling window — Trujillo,
    Healy, Dos Anjos, Horcher, M. Johnson. This is the core rolling-
    mean logic, exercised for real once history is long enough that
    the window is actually full."""
    sos = strength_of_schedule(khabib_history, n=5)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 4339)]
    assert row["sos_last_5"].iloc[0] == pytest.approx(1527.6754, abs=0.01)


def test_current_opponent_never_enters_own_sos_window(khabib_history):
    """McGregor, fight 11: the leakage guard. McGregor's own Elo
    (1681.6164, inflated by PPV drawing power relative to his actual
    UFC lightweight resume) must NOT appear in the window used to
    predict THIS fight. Expected value uses only fights 6-10 (Dos
    Anjos through Iaquinta). If .shift(1) is ever dropped from
    strength_of_schedule, McGregor's rating leaks into his own row
    and this number rises — the test would fail loudly instead of
    silently handing the model a peek at the outcome it's rating."""
    sos = strength_of_schedule(khabib_history, n=5)
    row = sos[(sos["fighter_id"] == KHABIB_ID) & (sos["bout_id"] == 3988)]
    assert row["sos_last_5"].iloc[0] == pytest.approx(1578.0218, abs=0.01)