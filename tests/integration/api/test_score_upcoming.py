"""
Integration tests for the batch scorer — Week 4 Tuesday, Step 8.

Skips (does not fail) without a snapshot or an active model, matching
the conftest pattern from Week 4 Monday. These exercise the parts of
scripts/score_upcoming.py that unit tests can't reach: the Elo-per-date
grouping, per-bout failure isolation, and JSON serializability.
"""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from api.dependencies import load_model_bundle
from api.services.features import snapshot_connection
from scripts.score_upcoming import fetch_upcoming_bout_ids, score_card, to_record


@pytest.fixture(scope="module")
def model(api_settings):
    """The active model bundle, or skip if none is registered."""
    engine = create_engine(api_settings.database_url)
    try:
        return load_model_bundle(engine)
    except RuntimeError as exc:
        pytest.skip(f"No active model: {exc}")


@pytest.fixture(scope="module")
def snapshot_available() -> None:
    if not Path("data/processed/bouts.parquet").exists():
        pytest.skip("No snapshot — run `uv run python -m features.snapshot`")


def test_scores_upcoming_card_without_failures(model, snapshot_available) -> None:
    """
    The end-to-end smoke test: every scheduled bout in the window
    scores cleanly.

    Asserts on SHAPE, never on specific fighters or probabilities —
    the card changes every week, and a test that hardcodes an expected
    probability becomes a test of the calendar rather than the code.
    """
    with snapshot_connection() as con:
        bouts = fetch_upcoming_bout_ids(con, weeks=4)
        if not bouts:
            pytest.skip("No scheduled bouts in the next 4 weeks")
        predictions, failures = score_card(con, model, bouts)

    assert not failures, f"bouts failed to score: {failures}"
    assert len(predictions) == len(bouts)

    for p in predictions:
        assert 0.0 <= p.probability_red <= 1.0
        assert p.predicted_winner_id in (p.fighter_red_id, p.fighter_blue_id)
        assert p.model_version == model.version
        # Uncalibrated is the CURRENT truth (ADR-017), read from the
        # registry. If a calibrated v2 ships, this assertion should be
        # updated deliberately, not silently pass either way.
        assert p.is_calibrated is False


def test_winner_matches_probability_direction(model, snapshot_available) -> None:
    """
    predicted_winner_id must agree with probability_red. These are
    computed in the same function, so they can only disagree via a
    tie-break bug — but a ledger row whose pick contradicts its own
    probability would be indefensible, so it gets asserted.
    """
    with snapshot_connection() as con:
        bouts = fetch_upcoming_bout_ids(con, weeks=4)
        if not bouts:
            pytest.skip("No scheduled bouts in the next 4 weeks")
        predictions, _ = score_card(con, model, bouts)

    for p in predictions:
        expected = (
            p.fighter_red_id if p.probability_red >= 0.5 else p.fighter_blue_id
        )
        assert p.predicted_winner_id == expected


def test_records_are_json_serializable(model, snapshot_available) -> None:
    """
    to_record output must survive json.dumps.

    Not busywork: BoutPrediction holds numpy floats and nested
    dataclasses, and numpy types are a classic silent JSON failure.
    Wednesday's ledger writer reuses this exact function, so a
    serialization bug here surfaces tomorrow as a failed write
    mid-card rather than a clean error.
    """
    with snapshot_connection() as con:
        bouts = fetch_upcoming_bout_ids(con, weeks=4)
        if not bouts:
            pytest.skip("No scheduled bouts in the next 4 weeks")
        predictions, _ = score_card(con, model, bouts[:3])

    for p in predictions:
        record = to_record(p)
        restored = json.loads(json.dumps(record))
        assert restored["bout_id"] == p.bout_id
        assert restored["coverage"]["fraction"] == pytest.approx(
            p.coverage.fraction
        )


def test_bad_bout_id_is_isolated_not_fatal(model, snapshot_available) -> None:
    """
    THE RESILIENCE GUARANTEE. One unscoreable bout must not take down
    the card.

    A nonexistent bout_id stands in for the real-world case: a
    Wikipedia stub with no resolvable record, or a bout row missing a
    corner. If one bad prelim aborted the run, the main event would
    never get scored — and Friday's cron would report total failure
    for a card that was 95% fine.
    """
    with snapshot_connection() as con:
        real = fetch_upcoming_bout_ids(con, weeks=4)
        if not real:
            pytest.skip("No scheduled bouts in the next 4 weeks")
        mixed = [real[0], (-999, real[0][1], "fake event")]
        predictions, failures = score_card(con, model, mixed)

    assert len(predictions) == 1
    assert len(failures) == 1
    assert failures[0][0] == -999