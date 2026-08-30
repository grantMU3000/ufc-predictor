"""
Offline unit tests for the inference arithmetic — Week 4 Tuesday,
Step 8.

WHAT'S HERE AND WHY: these test the small pure functions that decide
what the project publicly claims — the symmetry average, the symmetry
gap, and the contribution ranking. No database, no snapshot, no model
file. They run in milliseconds on every push.

The parity test (tests/integration/api/test_inference_parity.py) proves
the FEATURES are right. This file proves the MATH on top of them is
right. Both can fail independently, and a bug in either one is
invisible in the output — a wrong average and a wrong feature both
produce a well-formed probability.

average_symmetric_probability in particular is three characters from
being wrong: averaging p_A and p_B directly (instead of flipping the
second) pins every prediction near 0.5, and every fight would still
"look fine" on a dashboard.
"""

import numpy as np
import pytest

from api.services.features import DataCoverage
from api.services.inference import (
    average_symmetric_probability,
    rank_contributions,
    symmetry_gap,
)


class TestAverageSymmetricProbability:
    """The corner-swap fold, ADR-024 Decision 2."""

    def test_perfectly_symmetric_model_returns_input_unchanged(self) -> None:
        """
        A model that already ignores corner order gives p and 1-p on
        the two passes. Averaging must return p exactly — the whole
        point is that symmetry averaging costs nothing when the model
        is already symmetric.
        """
        assert average_symmetric_probability(0.7, 0.3) == pytest.approx(0.7)

    def test_asymmetric_model_lands_between_the_two_passes(self) -> None:
        """
        Red pass says 0.70, blue pass says 0.34 (i.e. 0.66 for red).
        The average is 0.68 — strictly between, favoring neither
        reading.
        """
        assert average_symmetric_probability(0.70, 0.34) == pytest.approx(0.68)

    def test_does_not_naively_average(self) -> None:
        """
        THE REGRESSION GUARD. The bug this function exists to prevent
        is averaging p_A and p_B without flipping the second. With
        0.7 and 0.3 that naive version returns 0.5 — a plausible
        number that would silently flatten every prediction the
        project ever publishes toward a coin flip.
        """
        result = average_symmetric_probability(0.7, 0.3)
        assert result != pytest.approx(0.5)
        assert result == pytest.approx(0.7)

    def test_output_stays_a_valid_probability(self) -> None:
        """Any two valid inputs must produce a valid probability out."""
        for p_red in (0.01, 0.5, 0.99):
            for p_blue in (0.01, 0.5, 0.99):
                assert 0.0 <= average_symmetric_probability(p_red, p_blue) <= 1.0


class TestSymmetryGap:
    """The standing corner-agnosticism diagnostic."""

    def test_symmetric_model_has_zero_gap(self) -> None:
        assert symmetry_gap(0.7, 0.3) == pytest.approx(0.0)

    def test_gap_is_the_disagreement_in_probability_points(self) -> None:
        """Red pass 0.70 vs blue pass implying 0.66 -> gap of 0.04."""
        assert symmetry_gap(0.70, 0.34) == pytest.approx(0.04)

    def test_gap_is_always_non_negative(self) -> None:
        """
        Absolute value, deliberately. The DIRECTION of the asymmetry
        isn't the question — "how far apart are the two readings" is.
        A signed gap would let opposite-direction bouts average out to
        a reassuring zero across a card.
        """
        assert symmetry_gap(0.34, 0.70) == pytest.approx(symmetry_gap(0.70, 0.34))
        assert symmetry_gap(0.34, 0.70) >= 0


class TestRankContributions:
    """Top-N feature attribution, ADR-024 Decision 3."""

    FEATURES = ("diff_age", "diff_reach_cm", "diff_slpm", "diff_elo_pre")

    def test_ranks_by_absolute_value_not_signed(self) -> None:
        """
        THE IMPORTANT ONE. A large negative contribution answers "what
        mattered most" just as much as a large positive. Ranking by
        signed value would return the top N reasons red wins and
        silently hide every reason he doesn't — a "why" panel that
        only ever argues one side.
        """
        contributions = np.array([0.1, -0.9, 0.3, -0.05])
        result = rank_contributions(
            self.FEATURES, contributions, np.array([1.0, 2.0, 3.0, 4.0]), top_n=2
        )
        assert [c.feature for c in result] == ["diff_reach_cm", "diff_slpm"]

    def test_favors_follows_the_sign(self) -> None:
        """Positive log-odds push toward red, negative toward blue."""
        contributions = np.array([0.5, -0.5, 0.0, 0.0])
        result = rank_contributions(
            self.FEATURES, contributions, np.array([1.0, 2.0, 3.0, 4.0]), top_n=2
        )
        favors = {c.feature: c.favors for c in result}
        assert favors["diff_age"] == "red"
        assert favors["diff_reach_cm"] == "blue"

    def test_nan_feature_value_becomes_none_not_zero(self) -> None:
        """
        A missing stat must serialize as null, never 0.0. Coercing it
        would read as "this fighter's reach advantage is exactly zero"
        — a claim, where the truth is "we don't know." Thin-history
        fighters hit this constantly (RESULTS.md: ~65% of rows are NaN
        on submission_success_rate alone).
        """
        result = rank_contributions(
            self.FEATURES,
            np.array([0.5, 0.1, 0.1, 0.1]),
            np.array([np.nan, 2.0, 3.0, 4.0]),
            top_n=1,
        )
        assert result[0].feature_value is None

    def test_top_n_caps_the_output(self) -> None:
        result = rank_contributions(
            self.FEATURES,
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            top_n=2,
        )
        assert len(result) == 2

    def test_top_n_larger_than_feature_count_is_safe(self) -> None:
        """Asking for 10 of 4 returns 4, not an error."""
        result = rank_contributions(
            self.FEATURES,
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            top_n=10,
        )
        assert len(result) == 4


class TestDataCoverage:
    """ADR-024 Decision 4 — coverage labels a prediction, never gates one."""

    def test_fraction_is_present_over_total(self) -> None:
        coverage = DataCoverage(
            n_features_present=24,
            n_features_total=32,
            red_prior_bouts=9,
            blue_prior_bouts=17,
        )
        assert coverage.fraction == pytest.approx(0.75)

    def test_full_coverage_is_one(self) -> None:
        coverage = DataCoverage(
            n_features_present=32,
            n_features_total=32,
            red_prior_bouts=9,
            blue_prior_bouts=17,
        )
        assert coverage.fraction == 1.0

    def test_debutant_coverage_matches_observed_floor(self) -> None:
        """
        The ~19% floor seen across seven debutant bouts in the first
        real scoring run. Pinned as a regression check: if a future
        change makes a debutant's coverage jump, some feature started
        returning a value where it previously (correctly) returned
        None — worth noticing.
        """
        coverage = DataCoverage(
            n_features_present=6,
            n_features_total=32,
            red_prior_bouts=0,
            blue_prior_bouts=0,
        )
        assert coverage.fraction == pytest.approx(0.1875, abs=0.005)