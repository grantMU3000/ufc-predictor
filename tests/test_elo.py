"""
Hand-computed tests for features/elo.py — three fighters, three
fights, worked out by hand BEFORE looking at the implementation,
same discipline as test_symmetrize.py / test_metrics.py.
"""

import pandas as pd
import pytest

from features.elo import compute_elo_ratings, expected_score, k_factor_by_experience


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

def test_k_factor_by_experience_starts_at_k_new():
    assert k_factor_by_experience(0, k_new=64.0, k_veteran=24.0, decay_scale=10.0) == pytest.approx(64.0)


def test_k_factor_by_experience_decays_toward_k_veteran():
    result = k_factor_by_experience(10, k_new=64.0, k_veteran=24.0, decay_scale=10.0)
    assert result == pytest.approx(38.715178, abs=1e-4)


def test_k_factor_by_experience_approaches_floor_for_veteran():
    result = k_factor_by_experience(1000, k_new=64.0, k_veteran=24.0, decay_scale=10.0)
    assert result == pytest.approx(24.0, abs=1e-3)


def test_k_factor_by_experience_monotonically_non_increasing():
    values = [k_factor_by_experience(n) for n in range(100)]
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_compute_elo_ratings_constant_k_unchanged():
    # Backward-compat guard: same assertion as
    # test_first_fight_both_fighters_at_initial_rating, re-run to
    # prove the new callable-dispatch code didn't quietly change the
    # old constant-K path.
    result = compute_elo_ratings(_synthetic_bouts(), k_factor=32.0)
    row = result[result["bout_id"] == 101].iloc[0]
    assert row["red_elo_pre"] == pytest.approx(1500.0)
    assert row["blue_elo_pre"] == pytest.approx(1500.0)


def test_compute_elo_ratings_with_experience_based_k():
    """
    Same 3-fighter, 3-fight sequence as before, but k_factor is now
    k_factor_by_experience (k_new=64, k_veteran=24, decay_scale=10).
    Hand-computed via a standalone script mirroring the real update
    loop (not derived from the implementation under test):

        bout 101 (A beats B, both debuts, both K=64):
            red_pre=1500.0   blue_pre=1500.0
        bout 102 (B beats C; B has 1 prior fight, C is a debut):
            red_pre=1468.0   blue_pre=1500.0
        bout 103 (C beats A; A has 1 prior fight, C has 1 prior fight):
            red_pre=1532.0   blue_pre=1465.060997
    """
    def k_fn(n):
        return k_factor_by_experience(n, k_new=64.0, k_veteran=24.0, decay_scale=10.0)

    result = compute_elo_ratings(_synthetic_bouts(), k_factor=k_fn)

    row1 = result[result["bout_id"] == 101].iloc[0]
    assert row1["red_elo_pre"] == pytest.approx(1500.0)
    assert row1["blue_elo_pre"] == pytest.approx(1500.0)

    row2 = result[result["bout_id"] == 102].iloc[0]
    assert row2["red_elo_pre"] == pytest.approx(1468.0)
    assert row2["blue_elo_pre"] == pytest.approx(1500.0)

    row3 = result[result["bout_id"] == 103].iloc[0]
    assert row3["red_elo_pre"] == pytest.approx(1532.0)
    assert row3["blue_elo_pre"] == pytest.approx(1465.060997, abs=1e-4)