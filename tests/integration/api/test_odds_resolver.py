"""
Point-in-time odds resolution — Week 4 Wednesday, ADR-025 Decision 2.

SYNTHETIC SNAPSHOTS, REAL FK TARGETS. odds_snapshots.bout_id and
.fighter_id are both NOT NULL foreign keys — a fake ID doesn't just
fail loudly, it fails for the wrong reason. Snapshot TIMING is what's
synthetic here (real odds_snapshots holds exactly one capture per
sportsbook per bout — confirmed n_snapshots == 2 * distinct_bouts for
every one of 23 books — so there is structurally never more than one
snapshot to choose from in real data). bout_id and fighter_id are real
throwaway rows from tests/integration/conftest.py's sample_bout /
sample_fighters, cleaned up the same way every other integration test
here cleans up.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from api.services.odds import (
    THIN_MARKET_BOOKS,
    implied_prob_to_moneyline,
    moneyline_to_implied_prob,
    resolve_odds_at,
)

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def odds_fixture(db_engine, sample_bout, sample_fighters):
    """
    Yields (conn, bout_id, fighter_id) plus an insert() helper, and
    deletes every snapshot row this test wrote afterward. No trigger
    to disable here — odds_snapshots isn't append-only, unlike
    predictions.
    """
    red_id, _blue_id, _swap_id = sample_fighters
    inserted_ids: list[int] = []

    def _insert(conn, sportsbook, moneyline, collected_at, implied_prob=None):
        row_id = conn.execute(
            text(
                """
                INSERT INTO odds_snapshots
                    (bout_id, fighter_id, sportsbook, moneyline,
                     implied_prob, collected_at)
                VALUES
                    (:bout_id, :fighter_id, :sportsbook, :moneyline,
                     :implied_prob, :collected_at)
                RETURNING id
                """
            ),
            {
                "bout_id": sample_bout,
                "fighter_id": red_id,
                "sportsbook": sportsbook,
                "moneyline": moneyline,
                "implied_prob": implied_prob,
                "collected_at": collected_at,
            },
        ).scalar_one()
        inserted_ids.append(row_id)

    with db_engine.begin() as conn:
        yield conn, sample_bout, red_id, _insert

    with db_engine.begin() as conn:
        conn.execute(
            text("DELETE FROM odds_snapshots WHERE bout_id = :bout_id"),
            {"bout_id": sample_bout},
        )


class TestMoneylineConversion:
    def test_negative_line_round_trips(self):
        prob = moneyline_to_implied_prob(-150)
        assert prob == pytest.approx(0.6)
        assert implied_prob_to_moneyline(prob) == -150

    def test_positive_line_round_trips(self):
        prob = moneyline_to_implied_prob(130)
        assert prob == pytest.approx(0.4347826, abs=1e-6)
        assert implied_prob_to_moneyline(prob) == 130

    def test_extreme_probability_raises(self):
        with pytest.raises(ValueError):
            implied_prob_to_moneyline(0.0)
        with pytest.raises(ValueError):
            implied_prob_to_moneyline(1.0)


class TestResolveOddsAt:
    def test_future_snapshot_excluded(self, odds_fixture):
        """
        The trap this file exists to catch: a snapshot collected AFTER
        the cutoff must never leak into the result. Same class of bug
        as training-time feature leakage, one layer over.
        """
        conn, bout_id, fighter_id, insert = odds_fixture
        insert(conn, "DraftKings", -150, T0 - timedelta(hours=2))
        insert(conn, "DraftKings", -200, T0 + timedelta(hours=2))

        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)

        assert result is not None
        assert result.moneyline == -150  # NOT -200, the future line
        assert result.n_books == 1

    def test_latest_before_cutoff_wins_per_book(self, odds_fixture):
        """One book, three snapshots — must resolve to the newest one before cutoff."""
        conn, bout_id, fighter_id, insert = odds_fixture
        insert(conn, "FanDuel", -110, T0 - timedelta(days=2))
        insert(conn, "FanDuel", -130, T0 - timedelta(hours=6))
        insert(conn, "FanDuel", -180, T0 - timedelta(hours=1))

        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)

        assert result.moneyline == -180
        assert result.n_books == 1

    def test_median_over_probability_not_moneyline(self, odds_fixture):
        """
        -150 -> 0.6000, +130 -> 0.4348. Averaging the RAW moneylines
        (-150, +130) lands on -10 — a price no book could post, since
        no American line exists strictly between -100 and +100.
        Averaging the PROBABILITIES avoids that; the result must
        convert back to a legal line.
        """
        conn, bout_id, fighter_id, insert = odds_fixture
        insert(conn, "BookA", -150, T0 - timedelta(hours=1))
        insert(conn, "BookB", 130, T0 - timedelta(hours=1))

        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)

        assert result.n_books == 2
        assert not (-100 < result.moneyline < 100)

    def test_no_coverage_returns_none(self, odds_fixture):
        conn, bout_id, fighter_id, _insert = odds_fixture
        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)
        assert result is None

    def test_thin_market_flag(self, odds_fixture):
        """Below THIN_MARKET_BOOKS, is_thin is True but the value is still returned."""
        assert THIN_MARKET_BOOKS == 3
        conn, bout_id, fighter_id, insert = odds_fixture
        insert(conn, "BookA", -150, T0 - timedelta(hours=1))
        insert(conn, "BookB", -140, T0 - timedelta(hours=1))

        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)

        assert result.n_books == 2
        assert result.is_thin is True

    def test_implied_prob_column_preferred_over_derived(self, odds_fixture):
        """
        When implied_prob is already stored, use it directly rather
        than re-deriving from moneyline — the column exists precisely
        so a book's own rounding isn't silently replaced by ours.
        """
        conn, bout_id, fighter_id, insert = odds_fixture
        insert(conn, "DraftKings", -150, T0 - timedelta(hours=1), implied_prob=0.6123)

        result = resolve_odds_at(conn, bout_id, fighter_id, as_of=T0)

        assert result.implied_prob == pytest.approx(0.6123)