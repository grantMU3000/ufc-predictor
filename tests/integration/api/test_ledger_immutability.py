"""
Proves the append-only trigger (migration c3f7a9d15b42, ADR-025
Decision 3) actually blocks UPDATE and DELETE at the database level.

WHY THIS EXISTS AS A TEST. "The ledger is immutable" is the load-
bearing claim of this entire project — it's what separates a real
track record from a spreadsheet someone could have edited after the
fact. An untested trigger is a claim, not a guarantee. The migration
could be reverted, a future migration could drop it, a restore could
recreate the table without it; every one of those is silent.

Asserts on SQLSTATE 23001 (restrict_violation), the ERRCODE the
trigger raises with — not on message text, which is free to be
reworded later. The SQLSTATE is the actual contract.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

RESTRICT_VIOLATION = "23001"


def test_update_is_rejected(db_engine, written_prediction):
    """Editing a published probability after the fact must be impossible."""
    entry, _ = written_prediction

    with db_engine.connect() as conn:
        with pytest.raises(DBAPIError) as exc_info:
            conn.execute(
                text(
                    "UPDATE predictions SET predicted_prob_red = 0.5000 "
                    "WHERE id = :id"
                ),
                {"id": entry.prediction_id},
            )
        assert exc_info.value.orig.pgcode == RESTRICT_VIOLATION
        conn.rollback()


def test_delete_is_rejected(db_engine, written_prediction):
    """
    Deleting a losing prediction is the more tempting failure mode —
    it silently improves every accuracy number downstream and leaves
    no trace.
    """
    entry, _ = written_prediction

    with db_engine.connect() as conn:
        with pytest.raises(DBAPIError) as exc_info:
            conn.execute(
                text("DELETE FROM predictions WHERE id = :id"),
                {"id": entry.prediction_id},
            )
        assert exc_info.value.orig.pgcode == RESTRICT_VIOLATION
        conn.rollback()


def test_rejected_update_leaves_the_row_intact(db_engine, written_prediction):
    """
    The trigger raises AND changes nothing.

    Not redundant with the raise test: a BEFORE trigger that raised
    after a partial write, or one attached with the wrong timing,
    could throw and still leave the row modified. This confirms the
    value on disk is untouched.
    """
    entry, original = written_prediction

    with db_engine.connect() as conn:
        try:
            conn.execute(
                text(
                    "UPDATE predictions SET predicted_prob_red = 0.5000 "
                    "WHERE id = :id"
                ),
                {"id": entry.prediction_id},
            )
        except DBAPIError:
            conn.rollback()

    with db_engine.connect() as conn:
        stored = conn.execute(
            text("SELECT predicted_prob_red FROM predictions WHERE id = :id"),
            {"id": entry.prediction_id},
        ).scalar_one()

    assert float(stored) == pytest.approx(original.probability_red, abs=1e-4)


def test_re_predicting_appends_rather_than_replaces(
    db_engine, written_prediction, sample_bout_prediction
):
    """
    Append-only means a second prediction for the same bout is a NEW
    row, and get_latest_prediction() resolves "current" by created_at
    at read time.

    This is the behavior that makes the immutability workable rather
    than merely restrictive — you can re-score a card after a data
    fix without ever destroying what you originally published.
    """
    from api.services.ledger import write_prediction
    from api.services.queries import fetch_latest_prediction_for_bout

    entry, _ = written_prediction

    with db_engine.connect() as conn:
        second = write_prediction(conn, sample_bout_prediction)
        conn.commit()

    try:
        with db_engine.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM predictions WHERE bout_id = :bout_id"),
                {"bout_id": entry.bout_id},
            ).scalar_one()
        latest = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

        assert count == 2
        assert latest["prediction_id"] == second.prediction_id
    finally:
        # Same force-delete the written_prediction fixture uses; this
        # second row has no fixture of its own to clean it up.
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE predictions DISABLE TRIGGER "
                    "trg_predictions_append_only"
                )
            )
            conn.execute(
                text("DELETE FROM predictions WHERE id = :id"),
                {"id": second.prediction_id},
            )
            conn.execute(
                text(
                    "ALTER TABLE predictions ENABLE TRIGGER "
                    "trg_predictions_append_only"
                )
            )


def test_cancelled_bout_is_refused(db_engine, sample_bout, sample_bout_prediction):
    """
    ADR-025 Decision 5, checked against live Postgres.

    The scorer reads status from the Parquet snapshot, which can be
    days stale — and fight-week withdrawals are exactly when that
    staleness bites. write_prediction() re-checks Postgres, so a bout
    cancelled since the last snapshot refresh is refused rather than
    logged as a live prediction.
    """
    from api.services.ledger import BoutNotScheduledError, write_prediction

    with db_engine.begin() as conn:
        conn.execute(
            text("UPDATE bouts SET status = 'cancelled' WHERE id = :id"),
            {"id": sample_bout},
        )

    with db_engine.connect() as conn:
        with pytest.raises(BoutNotScheduledError):
            write_prediction(conn, sample_bout_prediction)
        conn.rollback()