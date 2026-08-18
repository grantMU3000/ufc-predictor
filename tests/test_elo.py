"""
Hand-computed tests for features/elo.py — three fighters, three
fights, worked out by hand BEFORE looking at the implementation,
same discipline as test_symmetrize.py / test_metrics.py.
"""

import pandas as pd
import pytest

from features.elo import compute_elo_ratings, expected_score


def test_expected_score_equal_ratings_is_fifty_fifty():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_symmetric():
    a = expected_score(1600, 1400)
    b = expected_score(1400, 1600)
    assert a + b == pytest.approx(1.0)


def _synthetic_bouts() -> pd.DataFrame:
    """
    Fighter A=1, B=2, C=3, all starting at the default 1500.

    Fight 1 (bout_id=101): A beats B.
    Fight 2 (bout_id=102): B beats C.
    Fight 3 (bout_id=103): C beats A (upset).

    Hand-computed pre-fight ratings, K=32, initial=1500 (verified
    against a standalone Python calc, not just the code under test):
        Fight 1: A=1500.0,          B=1500.0
        Fight 2: B=1484.0,          C=1500.0
        Fight 3: A=1516.0,          C=1483.263693206478
    """
    return pd.DataFrame(
        [
            {"bout_id": 101, "event_date": pd.Timestamp("2024-01-01"),
             "fighter_red_id": 1, "fighter_blue_id": 2, "winner_id": 1},
            {"bout_id": 102, "event_date": pd.Timestamp("2024-02-01"),
             "fighter_red_id": 2, "fighter_blue_id": 3, "winner_id": 2},
            {"bout_id": 103, "event_date": pd.Timestamp("2024-03-01"),
             "fighter_red_id": 1, "fighter_blue_id": 3, "winner_id": 3},
        ]
    )


def test_first_fight_both_fighters_at_initial_rating():
    result = compute_elo_ratings(_synthetic_bouts(), k_factor=32.0)
    row = result[result["bout_id"] == 101].iloc[0]
    assert row["red_elo_pre"] == pytest.approx(1500.0)
    assert row["blue_elo_pre"] == pytest.approx(1500.0)


def test_second_fight_reflects_first_fights_result():
    result = compute_elo_ratings(_synthetic_bouts(), k_factor=32.0)
    row = result[result["bout_id"] == 102].iloc[0]
    # B lost fight 1 as a 50/50 favorite, so B drops exactly K*0.5=16:
    # 1500 - 16 = 1484.
    assert row["red_elo_pre"] == pytest.approx(1484.0)
    # C is a debutant here — never seen before, so default 1500.
    assert row["blue_elo_pre"] == pytest.approx(1500.0)


def test_third_fight_reflects_both_prior_results():
    result = compute_elo_ratings(_synthetic_bouts(), k_factor=32.0)
    row = result[result["bout_id"] == 103].iloc[0]
    assert row["red_elo_pre"] == pytest.approx(1516.0)
    assert row["blue_elo_pre"] == pytest.approx(1483.263693, abs=1e-4)


def test_raises_on_out_of_order_input():
    shuffled = _synthetic_bouts().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        compute_elo_ratings(shuffled)


def test_custom_initial_rating_applied_to_debutants():
    result = compute_elo_ratings(
        _synthetic_bouts(), k_factor=32.0, initial_rating=1000.0
    )
    row = result[result["bout_id"] == 101].iloc[0]
    assert row["red_elo_pre"] == pytest.approx(1000.0)
    assert row["blue_elo_pre"] == pytest.approx(1000.0)