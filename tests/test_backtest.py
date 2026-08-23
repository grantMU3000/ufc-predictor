"""
Hand-computable checks on models/backtest.py. Every expected value
here was worked out on paper first — if one of these fails, the
arithmetic is wrong, not the data.

Exists because Week 3 Friday's test-set read is one-time (ADR-020).
Discovering a sign error in the ROI calculation after the unlock
would mean the money numbers are wrong and cannot be re-run.
"""

import pandas as pd
import pytest

from features.odds import american_to_decimal
from models.backtest import (
    bootstrap_roi_ci,
    flat_stake_backtest,
    kelly_fraction,
    kelly_simulation,
    select_bets,
)


def _bet_frame(rows: list[dict]) -> pd.DataFrame:
    """Minimal frame with the columns select_bets requires."""
    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def test_american_to_decimal_favorite():
    # -150: risk 150 to win 100 -> 1 + 100/150
    result = american_to_decimal(pd.Series([-150.0]))
    assert result.iloc[0] == pytest.approx(1.666667, abs=1e-6)


def test_american_to_decimal_underdog():
    # +130: risk 100 to win 130 -> 1 + 130/100
    result = american_to_decimal(pd.Series([130.0]))
    assert result.iloc[0] == pytest.approx(2.30, abs=1e-9)


def test_american_to_decimal_matches_devig_direction():
    # A shorter price must always imply a higher probability, i.e. a
    # lower decimal payout. Guards against a flipped branch.
    favorite = american_to_decimal(pd.Series([-200.0])).iloc[0]
    underdog = american_to_decimal(pd.Series([200.0])).iloc[0]
    assert favorite < underdog


def test_kelly_even_money_sixty_percent():
    # Textbook case: p=0.60 at decimal 2.0 -> f* = 0.20
    f = kelly_fraction(pd.Series([0.60]).to_numpy(), pd.Series([2.0]).to_numpy())
    assert f[0] == pytest.approx(0.20, abs=1e-9)


def test_kelly_no_edge_is_zero():
    # p exactly equal to the price's implied probability -> no bet.
    f = kelly_fraction(pd.Series([0.50]).to_numpy(), pd.Series([2.0]).to_numpy())
    assert f[0] == pytest.approx(0.0, abs=1e-9)


def test_kelly_negative_edge_clipped_to_zero():
    f = kelly_fraction(pd.Series([0.40]).to_numpy(), pd.Series([2.0]).to_numpy())
    assert f[0] == 0.0


def test_flat_stake_roi_hand_computed():
    # 3 bets at even money, 2 wins 1 loss:
    #   profit = +1 +1 -1 = 1 unit on 3 staked -> ROI = 0.3333
    bets = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.6, "edge": 0.1},
            {"bout_id": "b", "event_date": "2025-01-02", "self_won": True,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.6, "edge": 0.1},
            {"bout_id": "c", "event_date": "2025-01-03", "self_won": False,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.6, "edge": 0.1},
        ]
    )
    result = flat_stake_backtest(bets)
    assert result["n_bets"] == 3
    assert result["total_profit"] == pytest.approx(1.0)
    assert result["roi"] == pytest.approx(1 / 3)
    assert result["hit_rate"] == pytest.approx(2 / 3)


def test_losing_bet_costs_exactly_one_unit_regardless_of_price():
    # A loss at 5.0 must cost the same as a loss at 1.2 — one unit.
    bets = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": False,
             "market_prob": 0.2, "decimal_odds": 5.0, "model_prob": 0.3, "edge": 0.1},
            {"bout_id": "b", "event_date": "2025-01-02", "self_won": False,
             "market_prob": 0.8, "decimal_odds": 1.2, "model_prob": 0.9, "edge": 0.1},
        ]
    )
    assert flat_stake_backtest(bets)["total_profit"] == pytest.approx(-2.0)


def test_select_bets_requires_positive_edge_at_zero_threshold():
    # Model LESS confident than market is never a bet, even at 0.00.
    df = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.58, "decimal_odds": 1.7, "model_prob": 0.56},
        ]
    )
    assert len(select_bets(df, 0.00)) == 0


def test_select_bets_threshold_filters():
    df = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.50, "decimal_odds": 2.0, "model_prob": 0.53},
        ]
    )
    assert len(select_bets(df, 0.02)) == 1   # edge 0.03 clears 0.02
    assert len(select_bets(df, 0.05)) == 0   # edge 0.03 misses 0.05


def test_select_bets_one_per_bout_when_both_sides_show_edge():
    # The asymmetry case: LightGBM's pair sums aren't exactly 1.0
    # (docs/RESULTS.md records max_pair_dev 0.0702), so both corners
    # of one bout can show positive edge. Only the larger survives.
    df = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.55, "decimal_odds": 1.8, "model_prob": 0.60},
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": False,
             "market_prob": 0.45, "decimal_odds": 2.2, "model_prob": 0.47},
        ]
    )
    bets = select_bets(df, 0.00)
    assert len(bets) == 1
    assert bets.iloc[0]["edge"] == pytest.approx(0.05)


def test_kelly_simulation_compounds_and_orders_chronologically():
    # Two even-money wins at quarter-Kelly on p=0.60 (f_full=0.20,
    # f_used=0.05, under the 5% cap):
    #   100 -> 105 -> 110.25
    bets = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.60, "edge": 0.1},
            {"bout_id": "b", "event_date": "2025-01-02", "self_won": True,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.60, "edge": 0.1},
        ]
    )
    summary, ledger = kelly_simulation(bets)
    assert summary["final_bankroll"] == pytest.approx(110.25, abs=1e-6)
    assert ledger["bankroll_before"].tolist() == pytest.approx([100.0, 105.0])


def test_kelly_per_bet_cap_binds():
    # p=0.90 at 2.0 -> f_full = 0.80, quarter-Kelly = 0.20, capped at 0.05.
    bets = _bet_frame(
        [
            {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
             "market_prob": 0.5, "decimal_odds": 2.0, "model_prob": 0.90, "edge": 0.4},
        ]
    )
    _, ledger = kelly_simulation(bets)
    assert ledger.iloc[0]["kelly_used"] == pytest.approx(0.05)
    assert ledger.iloc[0]["stake"] == pytest.approx(5.0)


def test_empty_selection_returns_clean_zeros_not_an_error():
    empty = select_bets(
        _bet_frame(
            [
                {"bout_id": "a", "event_date": "2025-01-01", "self_won": True,
                 "market_prob": 0.60, "decimal_odds": 1.7, "model_prob": 0.55},
            ]
        ),
        0.05,
    )
    assert flat_stake_backtest(empty)["n_bets"] == 0
    assert kelly_simulation(empty)[0]["final_bankroll"] == 100.0


def test_bootstrap_ci_brackets_point_estimate():
    bets = _bet_frame(
        [
            {"bout_id": str(i), "event_date": f"2025-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
             "self_won": i % 2 == 0, "market_prob": 0.5, "decimal_odds": 2.1,
             "model_prob": 0.55, "edge": 0.05}
            for i in range(40)
        ]
    )
    point = flat_stake_backtest(bets)["roi"]
    lo, hi = bootstrap_roi_ci(bets, n_boot=500)
    assert lo < point < hi