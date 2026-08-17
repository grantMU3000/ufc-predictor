"""
Hand-computed expected values, worked out on paper (or a calculator)
BEFORE looking at metrics.py's implementation — same discipline as
tests/test_symmetrize.py. If these ever fail, trust the hand math,
not the code.
"""

import numpy as np
import pytest

from models.metrics import (
    evaluate,
    expected_calibration_error,
    reliability_curve,
)

# Fixed 6-row example used across every test below, so the numbers
# can be verified once by hand and reused everywhere.
Y_TRUE = np.array([1, 0, 1, 1, 0, 0])
Y_PROB = np.array([0.9, 0.1, 0.6, 0.4, 0.5, 0.2])


def test_accuracy_hand_computed():
    # predicted winner (prob > 0.5): [1, 0, 1, 0, 0, 0]
    # actual winner:                 [1, 0, 1, 1, 0, 0]
    # row 3 (index 3) is the only miss -> 5/6 correct
    result = evaluate(Y_TRUE, Y_PROB, name="test")
    assert result["accuracy"] == pytest.approx(5 / 6)


def test_log_loss_hand_computed():
    # -log(0.9) - log(0.9) - log(0.6) - log(0.4) - log(0.5) - log(0.8), /6
    expected = -np.mean([
        np.log(0.9),  # y=1, p=0.9
        np.log(0.9),  # y=0, p=0.1 -> 1-p=0.9
        np.log(0.6),  # y=1, p=0.6
        np.log(0.4),  # y=1, p=0.4
        np.log(0.5),  # y=0, p=0.5 -> 1-p=0.5
        np.log(0.8),  # y=0, p=0.2 -> 1-p=0.8
    ])
    result = evaluate(Y_TRUE, Y_PROB, name="test")
    assert result["log_loss"] == pytest.approx(expected)


def test_brier_hand_computed():
    # (0.9-1)^2 + (0.1-0)^2 + (0.6-1)^2 + (0.4-1)^2 + (0.5-0)^2 + (0.2-0)^2, /6
    expected = np.mean([0.01, 0.01, 0.16, 0.36, 0.25, 0.04])
    result = evaluate(Y_TRUE, Y_PROB, name="test")
    assert result["brier"] == pytest.approx(expected)


def test_ece_hand_computed_two_bins():
    # n_bins=2 -> edges [0, 0.5, 1.0]. right=True means p<=0.5 -> bin0.
    # bin0 (p<=0.5): indices 1,3,4,5 -> y=[0,1,0,0] mean=0.25, p=[.1,.4,.5,.2] mean=0.3
    # bin1 (p>0.5):  indices 0,2     -> y=[1,1]     mean=1.0,  p=[.9,.6]      mean=0.75
    # ECE = (4/6)*|0.3-0.25| + (2/6)*|0.75-1.0| = (2/3)*0.05 + (1/3)*0.25
    expected = (2 / 3) * 0.05 + (1 / 3) * 0.25
    result = expected_calibration_error(Y_TRUE, Y_PROB, n_bins=2)
    assert result == pytest.approx(expected)


def test_reliability_curve_bin_contents_two_bins():
    curve = reliability_curve(Y_TRUE, Y_PROB, n_bins=2)
    assert len(curve) == 2

    bin0, bin1 = curve.iloc[0], curve.iloc[1]
    assert bin0["count"] == 4
    assert bin0["mean_predicted"] == pytest.approx(0.3)
    assert bin0["actual_rate"] == pytest.approx(0.25)

    assert bin1["count"] == 2
    assert bin1["mean_predicted"] == pytest.approx(0.75)
    assert bin1["actual_rate"] == pytest.approx(1.0)


def test_perfect_coin_flip_hits_reference_floor():
    # Every prediction says exactly 50%, and outcomes are an even
    # split -- this is the "beat a coin flip" floor from the plan.
    # A broken metric function is more likely to fail THIS test than
    # a real, subtle bug -- if this doesn't come out clean, stop and
    # check the function before trusting anything else.
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.full(8, 0.5)

    result = evaluate(y_true, y_prob, name="coinflip")
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["log_loss"] == pytest.approx(-np.log(0.5))  # ~0.6931
    assert result["brier"] == pytest.approx(0.25)


def test_mismatched_lengths_raises():
    with pytest.raises(AssertionError):
        evaluate(np.array([1, 0]), np.array([0.5]), name="bad")


def test_empty_input_raises():
    with pytest.raises(AssertionError):
        evaluate(np.array([]), np.array([]), name="empty")