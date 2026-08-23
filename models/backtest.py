"""
Turns predicted probabilities into simulated betting results — the
ROI backtest and Kelly simulation from docs/PLAN.md Section 3's Week 3
Friday entry, with the rules pre-registered in ADR-020 Decision 4.

Deliberately knows NOTHING about models, splits, or Parquet files. It
takes a DataFrame of (bout_id, event_date, self_won, model_prob,
market_prob, decimal_odds) and returns money numbers. That separation
is what lets the whole file be validated against hand-computed
examples in tests/test_backtest.py BEFORE it ever touches the test
set — the same "debug the harness on data you already understand"
discipline that Step 2 applies to the pipeline as a whole.

The core distinction running through this file, from ADR-020:
  - market_prob  = de-vigged belief -> used to compute EDGE
  - decimal_odds = raw vigged price -> used to compute MONEY
Mixing these up manufactures profit out of the sportsbook's margin.
"""

import numpy as np
import pandas as pd

# ADR-020 Decision 4: all three registered up front as a sensitivity
# sweep, not searched over after results are seen.
EDGE_THRESHOLDS = (0.00, 0.02, 0.05)

# ADR-020: paper-only, per docs/PLAN.md Section 0.3 (real-money
# betting is cut until 3+ months of logged out-of-sample results).
DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MAX_BET_FRACTION = 0.05


def select_bets(
    df: pd.DataFrame, edge_threshold: float, prob_col: str = "model_prob"
) -> pd.DataFrame:
    """
    Decides which fights get a bet, at one edge threshold.

    Simple version: edge = how much more confident the model is than
    the market. Bet only when the model likes a fighter MORE than the
    market does, by at least `edge_threshold`. A model that's less
    confident than the market on a fighter isn't a bet — it's just
    the model deferring.

    ONE BET PER BOUT, enforced here rather than assumed. Symmetrized
    data means every bout appears twice (once per fighter's
    perspective), and LightGBM has no structural symmetry guarantee —
    docs/RESULTS.md records a max pair-sum deviation of 0.0702, so a
    bout's two model probabilities can sum to ~1.07 rather than 1.00.
    When that happens, BOTH sides can show a positive edge at once,
    and betting both is betting a fight to end in two different
    results. The larger edge wins; the other row is dropped.

    Note that at threshold 0.00 the rule is still strictly `edge > 0`
    — a zero edge is the model agreeing with the market exactly,
    which is a reason not to bet, not a reason to bet at random.

    Parameters
    ----------
    df : pd.DataFrame
        Needs bout_id, event_date, self_won, market_prob,
        decimal_odds, and whatever column `prob_col` names.
    edge_threshold : float
        Minimum model_prob - market_prob required to place a bet.
    prob_col : str
        Which probability column is the model's opinion. Parameterized
        so the same function can score the train-only diagnostic
        artifact (ADR-020 Decision 1) without duplication.

    Returns
    -------
    pd.DataFrame — one row per bet, sorted chronologically (which
    kelly_simulation depends on), with an added `edge` column.
    """
    required = {"bout_id", "event_date", "self_won", "market_prob", "decimal_odds"}
    missing = required - set(df.columns)
    assert not missing, f"select_bets missing columns: {sorted(missing)}"

    candidates = df.copy()
    candidates["edge"] = candidates[prob_col] - candidates["market_prob"]

    qualifying = candidates[
        (candidates["edge"] > 0) & (candidates["edge"] >= edge_threshold)
    ]

    # One bet per bout: keep the side with the larger edge.
    qualifying = (
        qualifying.sort_values("edge", ascending=False)
        .drop_duplicates(subset="bout_id", keep="first")
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    return qualifying


def flat_stake_backtest(bets: pd.DataFrame) -> dict:
    """
    Flat 1-unit stake on every selected bet — the plan's specified
    sizing, and the honest way to measure whether the EDGE is real
    before letting bet sizing flatter or flatten the result.

    Profit per bet: (decimal_odds - 1) on a win, -1 on a loss. So a
    winning bet at 1.67 nets +0.67 units, and every loss costs
    exactly 1 unit regardless of price.

    Simple version: this answers "if I'd bet one unit on every fight
    my model liked, would I have more money at the end than I
    started with?" ROI is profit divided by total staked — +0.05
    means five cents of profit per dollar risked.

    Returns
    -------
    dict: n_bets, total_staked, total_profit, roi, hit_rate,
    avg_decimal_odds, avg_edge. Returns zeros/NaN cleanly at n_bets=0
    rather than raising — a threshold that selects no bets is a real,
    reportable outcome, not an error.
    """
    n_bets = len(bets)
    if n_bets == 0:
        return {
            "n_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": float("nan"),
            "hit_rate": float("nan"),
            "avg_decimal_odds": float("nan"),
            "avg_edge": float("nan"),
        }

    won = bets["self_won"].astype(bool).to_numpy()
    decimal = bets["decimal_odds"].to_numpy()
    profit = np.where(won, decimal - 1.0, -1.0)

    return {
        "n_bets": n_bets,
        "total_staked": float(n_bets),
        "total_profit": float(profit.sum()),
        "roi": float(profit.sum() / n_bets),
        "hit_rate": float(won.mean()),
        "avg_decimal_odds": float(decimal.mean()),
        "avg_edge": float(bets["edge"].mean()),
    }


def kelly_fraction(model_prob: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    """
    The Kelly criterion: what fraction of the bankroll maximizes
    long-run growth for a given edge and price.

        f* = (p * decimal_odds - 1) / (decimal_odds - 1)

    Simple version: Kelly says bet more when you have a bigger edge
    and when the price is better, and bet nothing when you have no
    edge. It's the mathematically optimal sizing IF your probability
    is exactly right — which is precisely the assumption this project
    has no business making, hence the fractional scaling in
    kelly_simulation below.

    Hand-checkable: a true 60% shot at even money (decimal 2.0) gives
    f* = (0.6*2 - 1) / 1 = 0.20 — bet 20% of bankroll. That number
    should feel uncomfortably large, and it is; full Kelly is
    notoriously volatile, which is why nobody sane bets it.

    Negative results are clipped to 0 — a negative Kelly fraction
    means "bet the other side," which select_bets has already ruled
    out by only ever returning positive-edge rows.
    """
    f = (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return np.clip(f, 0.0, 1.0)


def kelly_simulation(
    bets: pd.DataFrame,
    fraction: float = DEFAULT_KELLY_FRACTION,
    starting_bankroll: float = 100.0,
    max_bet_fraction: float = DEFAULT_MAX_BET_FRACTION,
    prob_col: str = "model_prob",
) -> tuple[dict, pd.DataFrame]:
    """
    Sequential, compounding bankroll simulation at fractional Kelly.

    Three deliberate guardrails, each guarding a different way this
    number could lie:
      1. FRACTIONAL (default 0.25). Full Kelly is optimal only if the
         model's probabilities are exactly correct. This project's own
         ECE is ~0.03 and ADR-017 found the model mildly
         underconfident — the probabilities are decent, not exact.
         Quarter-Kelly is the standard hedge against exactly that.
      2. PER-BET CAP (default 5% of bankroll). Stops one
         high-confidence call at a long price from becoming a
         bankroll-defining event.
      3. CHRONOLOGICAL ORDER, compounding. Bets are settled in the
         order they actually happened, so the bankroll path is a real
         sequence rather than a reshuffled one. Order matters enormously
         when stakes scale with bankroll — an early losing streak
         permanently shrinks every bet that follows.

    Bouts on the same card are settled sequentially rather than
    simultaneously. Slightly unrealistic (you'd size all of one card's
    bets off the same pre-card bankroll), but it's a small effect at
    quarter-Kelly with a 5% cap, and it errs toward smoother rather
    than flattering results.

    Returns
    -------
    (summary, ledger)
        summary : dict — n_bets, starting/final bankroll, growth
            multiple, max_drawdown, roi_on_turnover
        ledger : pd.DataFrame — the full per-bet path (bankroll before,
            stake, profit, bankroll after), for plotting the equity
            curve in the README.
    """
    if len(bets) == 0:
        return (
            {
                "n_bets": 0,
                "starting_bankroll": starting_bankroll,
                "final_bankroll": starting_bankroll,
                "growth_multiple": 1.0,
                "max_drawdown": 0.0,
                "roi_on_turnover": float("nan"),
            },
            pd.DataFrame(),
        )

    f_full = kelly_fraction(
        bets[prob_col].to_numpy(), bets["decimal_odds"].to_numpy()
    )
    f_used = np.minimum(f_full * fraction, max_bet_fraction)

    bankroll = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    total_staked = 0.0
    rows = []

    for i, bet in enumerate(bets.itertuples(index=False)):
        stake = bankroll * f_used[i]
        won = bool(bet.self_won)
        profit = stake * (bet.decimal_odds - 1.0) if won else -stake

        rows.append(
            {
                "event_date": bet.event_date,
                "bout_id": bet.bout_id,
                "decimal_odds": bet.decimal_odds,
                "kelly_full": f_full[i],
                "kelly_used": f_used[i],
                "bankroll_before": bankroll,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll + profit,
            }
        )

        bankroll += profit
        total_staked += stake
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

    summary = {
        "n_bets": len(bets),
        "starting_bankroll": starting_bankroll,
        "final_bankroll": float(bankroll),
        "growth_multiple": float(bankroll / starting_bankroll),
        "max_drawdown": float(max_drawdown),
        "roi_on_turnover": float((bankroll - starting_bankroll) / total_staked)
        if total_staked > 0
        else float("nan"),
    }
    return summary, pd.DataFrame(rows)


def bootstrap_roi_ci(
    bets: pd.DataFrame, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval on flat-stake ROI.

    Simple version: resample the same bets with replacement a few
    thousand times, recompute ROI each time, and report the middle
    95% of those answers. It asks "how different could this ROI have
    looked if the same strategy had run into a slightly different
    run of luck?"

    This exists because a point ROI on a few hundred bets is close to
    meaningless on its own — fight outcomes are high-variance, and a
    handful of underdog wins can swing the number by several points.
    Same instinct as ADR-015's permutation test: before believing a
    number, establish what it looks like under noise. If this
    interval spans zero — and on a few hundred bets it almost
    certainly will — the honest reading is "no demonstrated edge,"
    not "slightly profitable."

    Returns
    -------
    (lower, upper) at the given alpha. (nan, nan) if there are no bets.
    """
    if len(bets) == 0:
        return (float("nan"), float("nan"))

    won = bets["self_won"].astype(bool).to_numpy()
    decimal = bets["decimal_odds"].to_numpy()
    profit = np.where(won, decimal - 1.0, -1.0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    boot_rois = profit[idx].mean(axis=1)

    lower = float(np.percentile(boot_rois, 100 * alpha / 2))
    upper = float(np.percentile(boot_rois, 100 * (1 - alpha / 2)))
    return (lower, upper)


def run_backtest_sweep(
    df: pd.DataFrame,
    prob_col: str = "model_prob",
    thresholds: tuple = EDGE_THRESHOLDS,
    label: str = "",
) -> pd.DataFrame:
    """
    Runs the full ADR-020 Decision 4 sweep: flat-stake ROI plus
    bootstrap CI plus a Kelly simulation, at every pre-registered
    edge threshold, in one table.

    All three thresholds are always reported together, by design.
    Reporting only the best-performing one after the fact is exactly
    the post-hoc selection ADR-020 forecloses — the sweep is a
    sensitivity analysis, and a strategy that only works at one
    threshold is a strategy that probably doesn't work.

    Returns
    -------
    pd.DataFrame — one row per threshold, ready to paste into
    docs/RESULTS.md.
    """
    rows = []
    for threshold in thresholds:
        bets = select_bets(df, threshold, prob_col=prob_col)
        flat = flat_stake_backtest(bets)
        ci_lo, ci_hi = bootstrap_roi_ci(bets)
        kelly, _ = kelly_simulation(bets, prob_col=prob_col)

        rows.append(
            {
                "label": label,
                "edge_threshold": threshold,
                **flat,
                "roi_ci_low": ci_lo,
                "roi_ci_high": ci_hi,
                "kelly_final_bankroll": kelly["final_bankroll"],
                "kelly_growth_multiple": kelly["growth_multiple"],
                "kelly_max_drawdown": kelly["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)