"""
tests/test_features.py

Hand-verified regression tests for Tier 1 / Tier 2 feature functions,
anchored on two real fighterz (Quillan Salkilld, fighter_id=394, and
Charles Oliveira, fighter_id=146) whose
numbers were independently hand-computed against raw SQL queries
before any assertion here was written. These tests exist to catch a
future refactor silently changing what a feature computes — the bugs
they might have caught were already found and fixed during manual
verification, not by these tests themselves.

Requires a local Parquet snapshot at data/processed/ (see
features/snapshot.py) — skipped automatically if it hasn't been
generated, the same pattern already used by the integration tests'
TEST_DATABASE_URL skip.
"""

from datetime import date
from pathlib import Path

import duckdb
import pytest

from features.tier1 import (
    age_at_fight,
    height_at_fight,
    is_title_fight_at_bout,
    reach_at_fight,
    reach_to_height_ratio,
    scheduled_rounds_at_bout,
    stance_at_fight,
    stance_matchup,
    weight_class_at_bout,
)
from features.tier2 import (
    average_fight_time_seconds,
    career_win_percentage,
    control_time_percentage,
    days_since_last_fight,
    decision_loss_percentage,
    decision_win_percentage,
    finish_rate,
    get_total_seconds_fought,
    knockdown_rate,
    ko_loss_rate,
    significant_strike_rate,
    strikes_absorbed_per_minute,
    strikes_landed_per_minute,
    striking_accuracy,
    striking_defense,
    striking_output_decay,
    sub_loss_rate,
    submission_success_rate,
    submission_win_count,
    submissions_attempted_per_15,
    takedown_accuracy,
    takedown_defense,
    takedown_output_decay,
    takedowns_landed_per_15,
    time_controlled_percentage,
    times_knocked_down,
    title_fight_experience,
    total_ufc_fights,
)

SNAPSHOT_DIR = Path("data/processed")
REQUIRED_TABLES = ["fighters", "events", "bouts", "bout_stats"]

# Quillan Salkild — the hand-verification anchor: a real, small
# (6-fight) UFC record, all wins, one 3-round fight, no draws/DQs/
# no-contests. Small enough to hand-count from raw SQL, varied enough
# to exercise the decided-bouts filter and the round-split logic —
# not just the trivial debutant (0 prior fights) case.
SALKILD_FIGHTER_ID = 394
AS_OF_DATE = date(2026, 8, 14)
SALKILD_LATEST_BOUT_ID = 34437
SALKILD_LATEST_OPPONENT_ID = 2049  # from bout 34437 — used for stance_matchup


def _snapshot_available() -> bool:
    """True only if every table this file needs has a local Parquet file."""
    return all(
        (SNAPSHOT_DIR / f"{table}.parquet").exists() for table in REQUIRED_TABLES
    )


@pytest.fixture(scope="module")
def con():
    """
    A DuckDB connection with fighters/events/bouts/bout_stats exposed
    as VIEWS over the local Parquet snapshot — the same table names
    every feature function's SQL expects to find. A view doesn't copy
    the data, it just tells DuckDB "read this Parquet file whenever
    something asks for this table name."

    module-scoped: every test in this file shares one connection
    instead of re-attaching the snapshot per test — the snapshot
    doesn't change mid-file, so there's no reason to redo the setup.
    """
    if not _snapshot_available():
        pytest.skip(
            "Local Parquet snapshot not found at data/processed/ — "
            "run features/snapshot.py first."
        )

    connection = duckdb.connect()
    for table in REQUIRED_TABLES:
        parquet_path = SNAPSHOT_DIR / f"{table}.parquet"
        connection.execute(f"CREATE VIEW {table} AS SELECT * FROM '{parquet_path}'")
    yield connection
    connection.close()


class TestSalkildHandVerified:
    """
    Each test's expected value came from a hand computation done
    independently, against raw SQL, BEFORE this assertion was
    written — not derived by calling other code. If one of these
    ever fails after a refactor, the bug is almost certainly in the
    new CODE, not in the expected number — re-verify by hand before
    changing anything in this file.
    """

    def test_age_at_fight(self, con):
        """
        dob = 1999-12-28, as_of_date = 2026-08-14 -> 9,726 days
        (26 full years + a 229-day remainder, hand-counted) ->
        9726 / 365.25 ≈ 26.628 years.
        """
        result = age_at_fight(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(26.628, abs=0.01)

    def test_striking_accuracy(self, con):
        """
        SUM(sig_strikes_landed)=126, SUM(sig_strikes_attempted)=218
        across all 6 prior completed bouts -> 126/218 ≈ 0.578.
        """
        result = striking_accuracy(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.578, abs=0.001)

    def test_career_win_percentage(self, con):
        """
        All 6 prior bouts have winner_id == 394 — no draws, DQs, or
        no-contests in this window. 6 decided bouts, 6 wins -> 1.0.
        Useful edge case: confirms _decided_bouts behaves correctly
        on a fully-decided, undefeated-so-far record, not just a
        record with a draw/DQ mixed in.
        """
        result = career_win_percentage(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1.0)

    def test_striking_output_decay(self, con):
        """
        Early rounds (1-2) sig strikes landed: 6, 19, 20, 14, 5, 12,
        10 -> avg 86/7 ≈ 12.286. Late rounds (3+): only 40 (his one
        3-round fight, 2025-06-07) -> avg 40.0. Decay = 40.0 - 12.286
        ≈ +27.714 — his output actually INCREASED in the one fight
        that reached round 3. Built from a single late-round data
        point; don't read this as a stable "doesn't fade" pattern for
        this fighter specifically, just as a correct computation.
        """
        result = striking_output_decay(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(27.714, abs=0.01)

    def test_days_since_last_fight(self, con):
        """Last prior bout 2026-08-08 -> (2026-08-14 - 2026-08-08).days = 6."""
        assert days_since_last_fight(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == 6

    def test_total_ufc_fights(self, con):
        """6 completed prior bouts, hand-counted from the bout list."""
        assert total_ufc_fights(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == 6

    def test_submission_win_count(self, con):
        """2 of 6 wins by Submission (2026-01-31, 2026-08-08)."""
        assert submission_win_count(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == 2

    def test_decision_win_percentage(self, con):
        """1 decision win / 6 decided bouts -> 1/6 ≈ 0.167."""
        result = decision_win_percentage(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.167, abs=0.001)

    def test_decision_loss_percentage(self, con):
        """0 decision losses / 6 decided bouts -> 0.0. Real zero, not None -- he HAS decided bouts, just no losses at all."""
        result = decision_loss_percentage(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.0)

    def test_finish_rate(self, con):
        """5 of 6 wins are non-decision, non-DQ finishes -> 5/6 ≈ 0.833."""
        result = finish_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.833, abs=0.001)

    def test_ko_loss_rate_none_when_undefeated(self, con):
        """
        Salkild has zero non-DQ losses in this window, so _non_dq_losses
        is empty and this correctly returns None instead of dividing by
        zero or fabricating a 0.0. Doesn't verify real KO-loss arithmetic
        -- see conversation note on testing this with a second fighter.
        """
        assert ko_loss_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE) is None

    def test_sub_loss_rate_none_when_undefeated(self, con):
        """Same reasoning as ko_loss_rate_none_when_undefeated."""
        assert sub_loss_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE) is None

    # --- Tier 1: physical/stance, straight DB lookups ---

    def test_height_at_fight(self, con):
        assert height_at_fight(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == pytest.approx(
            182.9
        )

    def test_reach_at_fight(self, con):
        assert reach_at_fight(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == pytest.approx(
            190.5
        )

    def test_reach_to_height_ratio(self, con):
        """190.5 / 182.9 ≈ 1.04155 (long division verified by hand)."""
        result = reach_to_height_ratio(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1.04155, abs=0.0005)

    def test_stance_at_fight(self, con):
        assert stance_at_fight(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == "orthodox"

    def test_stance_matchup(self, con):
        """
        Salkild (orthodox) vs his bout-34437 opponent (southpaw) ->
        a genuine open-stance pairing, per _OPEN_STANCE_PAIRS.
        """
        descriptive, is_open = stance_matchup(
            con, SALKILD_FIGHTER_ID, SALKILD_LATEST_OPPONENT_ID, AS_OF_DATE
        )
        assert descriptive == "orthodox vs southpaw"
        assert is_open is True

    # --- Tier 1: bout-level facts (bout_id only, no fighter/date) ---

    def test_weight_class_at_bout(self, con):
        assert weight_class_at_bout(con, SALKILD_LATEST_BOUT_ID) == "Lightweight Bout"

    def test_is_title_fight_at_bout(self, con):
        assert is_title_fight_at_bout(con, SALKILD_LATEST_BOUT_ID) is False

    def test_scheduled_rounds_at_bout(self, con):
        assert scheduled_rounds_at_bout(con, SALKILD_LATEST_BOUT_ID) == 5

    # --- Tier 2: total fight time (the shared denominator) ---

    def test_get_total_seconds_fought(self, con):
        """
        Per-bout durations: 19 + 900 + 150 + 182 + 209 + 265 = 1725s.
        Every rate-based Tier 2 test below divides by this same number
        -- if this one is wrong, most of the others would be too.
        """
        result = get_total_seconds_fought(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1725.0)

    def test_average_fight_time_seconds(self, con):
        assert average_fight_time_seconds(
            con, SALKILD_FIGHTER_ID, AS_OF_DATE
        ) == pytest.approx(287.5)

    # --- Tier 2: per-time-window rates ---

    def test_strikes_landed_per_minute(self, con):
        """126 sig strikes landed * 60 / 1725s ≈ 4.3826."""
        result = strikes_landed_per_minute(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(4.3826, abs=0.001)

    def test_strikes_absorbed_per_minute(self, con):
        """58 opponent sig strikes landed * 60 / 1725s ≈ 2.0174."""
        result = strikes_absorbed_per_minute(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(2.0174, abs=0.001)

    def test_takedowns_landed_per_15(self, con):
        """11 takedowns landed * 900 / 1725s ≈ 5.7391."""
        result = takedowns_landed_per_15(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(5.7391, abs=0.001)

    def test_submissions_attempted_per_15(self, con):
        """2 submission attempts * 900 / 1725s ≈ 1.0435."""
        result = submissions_attempted_per_15(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1.0435, abs=0.001)

    def test_knockdown_rate(self, con):
        """3 knockdowns scored * 900 / 1725s ≈ 1.5652."""
        result = knockdown_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1.5652, abs=0.001)

    # --- Tier 2: ratio/accuracy features ---

    def test_takedown_accuracy(self, con):
        """11 landed / 33 attempted = 0.3333."""
        result = takedown_accuracy(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.3333, abs=0.001)

    def test_significant_strike_rate(self, con):
        """126 sig strikes landed / 162 total strikes landed = 7/9 ≈ 0.7778."""
        result = significant_strike_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.7778, abs=0.001)

    def test_striking_defense(self, con):
        """1 - (58 opp landed / 115 opp attempted) ≈ 0.4957."""
        result = striking_defense(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.4957, abs=0.001)

    def test_takedown_defense(self, con):
        """1 - (3 opp landed / 13 opp attempted) ≈ 0.7692."""
        result = takedown_defense(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.7692, abs=0.001)

    def test_submission_success_rate(self, con):
        """2 submission wins / 2 submission attempts = 1.0."""
        result = submission_success_rate(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(1.0)

    # --- Tier 2: percentages ---

    def test_control_time_percentage(self, con):
        """637s controlling / 1725s total ≈ 0.3693."""
        result = control_time_percentage(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.3693, abs=0.001)

    def test_time_controlled_percentage(self, con):
        """374s controlled / 1725s total ≈ 0.2168."""
        result = time_controlled_percentage(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.2168, abs=0.001)

    # --- Tier 2: career counts, including two documented "true zero" cases ---

    def test_title_fight_experience_is_zero(self, con):
        """All 6 prior bouts have is_title_fight=false -- a real 0, not a gap."""
        assert title_fight_experience(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == 0

    def test_times_knocked_down_is_zero(self, con):
        """opp_knockdowns sums to 0 across all 6 prior bouts -- never been dropped, per this data."""
        assert times_knocked_down(con, SALKILD_FIGHTER_ID, AS_OF_DATE) == 0

    # --- Tier 2: takedown output decay ---

    def test_takedown_output_decay(self, con):
        """
        Early-round (1-2) takedowns landed: 0,4,3,0,2,0,1 -> avg 10/7
        ≈ 1.4286. Late-round (3+): just 1 (his one 3-round fight) ->
        avg 1.0. Decay ≈ -0.4286 -- fewer takedowns late, the OPPOSITE
        sign from his striking_output_decay (+27.71) on that same
        fight. A genuinely interesting real pattern, not a bug: he
        seems to shift toward striking and away from grappling as
        that particular fight wore on.
        """
        result = takedown_output_decay(con, SALKILD_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(-0.4286, abs=0.001)


OLIVEIRA_FIGHTER_ID = 146


class TestOliveiraHandVerified:
    """
    Second hand-verification anchor, specifically to get REAL numeric
    coverage on ko_loss_rate/sub_loss_rate — Salkild is undefeated in
    his tracked window, so those two could only be proven not to
    crash on him, never proven arithmetically correct.

    One bout in his history (bout_id 7163, method="Overturned")
    initially appeared to have a non-null winner_id in a pasted query
    result, which would have contradicted _decided_bouts' assumption
    that Overturned bouts null out winner_id like a draw. A direct
    COUNT(winner_id) check confirmed this was a formatting artifact
    in how the result was copied, not real data — Overturned bouts do
    null out winner_id as originally assumed, and this bout is
    correctly excluded from the counts below.
    """

    def test_ko_loss_rate(self, con):
        """
        36 decided bouts (Overturned bout_id=7163 correctly excluded
        -- winner_id is null for it). 11 non-DQ losses, 5 by KO/TKO
        -> 5/11 ≈ 0.4545.
        """
        result = ko_loss_rate(con, OLIVEIRA_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.4545, abs=0.001)

    def test_sub_loss_rate(self, con):
        """4 of 11 non-DQ losses by Submission -> 4/11 ≈ 0.3636."""
        result = sub_loss_rate(con, OLIVEIRA_FIGHTER_ID, AS_OF_DATE)
        assert result == pytest.approx(0.3636, abs=0.001)
