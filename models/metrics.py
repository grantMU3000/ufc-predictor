"""
Scores a set of predicted probabilities against what actually happened.
This is the single most important file in the whole project — every
baseline and every future model (LR today, LightGBM in Week 3) gets
judged through these same four functions, computed the same way every
time. Write it once, trust it forever; get it wrong once, and every
number in docs/RESULTS.md is quietly wrong too.
"""

import numpy as np
import pandas as pd


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> dict:
    """
    Computes accuracy, log loss, Brier score, and ECE for one set of
    predictions, all in one call, so every baseline reports the exact
    same four numbers.

    Simple version: y_true is what actually happened (1 = this fighter
    won). y_prob is what was predicted beforehand. Accuracy just asks
    "did you call it right." Log loss and Brier also ask "how honest
    was your confidence" — calling a coinflip fight 95% and being wrong
    costs way more here than accuracy alone would ever show. ECE asks
    a third question: "when you said 70%, did that fighter actually
    win about 70% of the time?"

    Parameters
    ----------
    y_true : array-like of 0/1
        Whether self_fighter_id actually won.
    y_prob : array-like of float in [0, 1]
        Predicted probability self_fighter_id wins.
    name : str
        Label carried into the returned dict (e.g. "market", "logreg")
        so results from multiple calls stack cleanly into one
        DataFrame for docs/RESULTS.md.

    Returns
    -------
    dict: name, n, accuracy, log_loss, brier, ece
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    assert len(y_true) == len(y_prob), "y_true and y_prob must be the same length"
    assert len(y_true) > 0, "cannot evaluate on zero rows"

    return {
        "name": name,
        "n": len(y_true),
        "accuracy": _accuracy(y_true, y_prob),
        "log_loss": _log_loss(y_true, y_prob),
        "brier": _brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Fraction of rows where the side given >50% actually won. A
    prediction of exactly 0.5 counts as a miss either way — a true
    coin-flip call isn't "correct" no matter which way the tie breaks.
    """
    predicted_win = y_prob > 0.5
    return float((predicted_win == (y_true == 1)).mean())


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Average of -[y*log(p) + (1-y)*log(1-p)]. This is the plan's real
    headline metric (docs/PLAN.md Section 0.1), not accuracy — it
    punishes confident wrong calls far harder than accuracy does,
    which is exactly the behavior you want a betting-adjacent model
    graded on.

    eps keeps y_prob away from exactly 0 or 1 — log(0) is -infinity,
    and a single maximally-overconfident wrong prediction would
    otherwise blow the whole average up to infinity.
    """
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Average (y_prob - y_true)^2 — same spirit as log loss (punishes
    overconfidence) but with squared error instead of a log penalty,
    so one wild miscall can't dominate the average the way it can in
    log loss. Kept as a sanity check ALONGSIDE log loss per
    docs/PLAN.md Section 0.1: if the two ever rank baselines
    differently, that disagreement is worth a second look before
    trusting either one.
    """
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    Checks whether stated confidence means what it says. Buckets
    predictions by probability (e.g. everything called 70-80%), then
    for each bucket compares average predicted probability against
    actual win rate in that bucket. ECE is the average gap, weighted
    by bucket size.

    Simple version: if you say "70% chance" a hundred times across a
    season, a well-calibrated forecaster wins about 70 of those, not
    50 and not 95. This measures how far off that is. 0.0 is perfect;
    docs/PLAN.md's target is <= 0.05.

    Returns
    -------
    float — weighted average |predicted - actual| across non-empty bins.
    """
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    curve = curve[curve["count"] > 0]
    if curve.empty:
        return float("nan")
    weights = curve["count"] / curve["count"].sum()
    gaps = (curve["mean_predicted"] - curve["actual_rate"]).abs()
    return float((weights * gaps).sum())


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    The raw data behind a reliability diagram (docs/PLAN.md Week 3
    Wednesday) and behind expected_calibration_error above — its own
    function, not inlined into ECE, so the exact same numbers feed a
    plot later without recomputing anything.

    Parameters
    ----------
    n_bins : int
        Equal-width buckets over [0, 1]. 10 = deciles.

    Returns
    -------
    pd.DataFrame, one row per bin: bin_low, bin_high, mean_predicted,
    actual_rate, count. Empty bins still appear (NaN rate, count=0)
    rather than vanishing — easier to spot a confidence range with no
    predictions on a plot than to have it silently missing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0, 1, n_bins + 1)
    # right=True + clip: a prediction of exactly 1.0 lands in the
    # last bin instead of digitizing one bin past the end.
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        rows.append({
            "bin_low": edges[i],
            "bin_high": edges[i + 1],
            "mean_predicted": float(y_prob[mask].mean()) if count > 0 else float("nan"),
            "actual_rate": float(y_true[mask].mean()) if count > 0 else float("nan"),
            "count": count,
        })
    return pd.DataFrame(rows)