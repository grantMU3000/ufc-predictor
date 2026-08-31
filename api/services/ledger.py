"""
Writes predictions to the immutable ledger — Week 4 Wednesday, ADR-025.

Two guards live here and nowhere else:

  1. STATUS CHECKED AGAINST POSTGRES, NOT THE PARQUET SNAPSHOT.
     scripts/score_upcoming.py correctly filters by the snapshot's
     status column for SCORING — a bout has to exist in Parquet to be
     scored at all. But a bout can be cancelled AFTER the last
     snapshot refresh, and fight-week withdrawals are exactly when
     that's likely. write_prediction() re-checks bouts.status live,
     at write time, so a stale snapshot can never log a cancelled
     bout as a real prediction.

  2. THIS FILE NEVER ISSUES UPDATE OR DELETE. The migration's trigger
     (ADR-025 Decision 3) is the real guarantee; every write here is a
     plain INSERT, and re-predicting a bout adds a new row.

Reads live in api/services/queries.py, not here. That module already
owns every read path the API uses and already joins the fighter names
and settlement rows the responses need — a second "latest prediction
for a bout" implementation would be exactly the drift ADR-024
Decision 1 committed against. This file is the WRITE half only.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import Connection, text

from api.services.inference import BoutPrediction
from api.services.odds import resolve_odds_at
from api.services.records import to_record

# predicted_prob_red is Numeric(5,4) in Postgres. Full float precision
# still lives inside feature_snapshot's JSONB — this only bounds the
# typed column's storage contract.
_PROB_QUANTIZE = Decimal("0.0001")


class BoutNotScheduledError(Exception):
    """Raised when write_prediction is called on a non-scheduled bout."""


@dataclass(frozen=True)
class LedgerEntry:
    """The row identity returned after a successful write."""

    prediction_id: int
    bout_id: int
    model_version: str
    created_at: datetime


def _get_bout_status(conn: Connection, bout_id: int) -> str | None:
    """Live status from Postgres. None if the bout_id doesn't exist."""
    row = conn.execute(
        text("SELECT status FROM bouts WHERE id = :bout_id"),
        {"bout_id": bout_id},
    ).one_or_none()
    return None if row is None else str(row.status)


def write_prediction(
    conn: Connection,
    prediction: BoutPrediction,
    as_of: datetime | None = None,
) -> LedgerEntry:
    """
    Write one prediction to the immutable ledger.

    Simple version: confirm the fight is still happening, look up what
    the market thought right now, package everything the model saw
    into one JSON blob, insert one row. Never touches an existing row.

    Parameters
    ----------
    conn : Connection
        An open connection the caller commits or rolls back. Left to
        the caller so Step 6's per-bout loop can isolate one bad write
        without rolling back the whole card — same "one bad bout does
        not kill the card" principle as score_card() itself.
    prediction : BoutPrediction
        Output of api.services.inference.predict_bout.
    as_of : datetime, optional
        The moment this prediction is being made — used as the odds
        cutoff. Defaults to now(UTC). Exposed so tests can pin it.

    Raises
    ------
    BoutNotScheduledError
        If bouts.status (live, in Postgres) is not 'scheduled' —
        whether it was already cancelled before scoring or cancelled
        between scoring and writing (ADR-025 Decision 5).
    """
    as_of = as_of or datetime.now(UTC)

    status = _get_bout_status(conn, prediction.bout_id)
    if status is None:
        raise BoutNotScheduledError(f"bout {prediction.bout_id} does not exist")
    if status != "scheduled":
        raise BoutNotScheduledError(
            f"bout {prediction.bout_id} has status '{status}', refusing to "
            f"write a prediction (ADR-025 Decision 5)"
        )

    odds = resolve_odds_at(
        conn, prediction.bout_id, prediction.predicted_winner_id, as_of
    )

    envelope = to_record(prediction)
    envelope["predicted_at"] = as_of.isoformat()

    prob_rounded = Decimal(str(prediction.probability_red)).quantize(
        _PROB_QUANTIZE, rounding=ROUND_HALF_UP
    )

    row = conn.execute(
        text(
            """
            INSERT INTO predictions (
                bout_id, model_version, predicted_prob_red,
                predicted_winner_id, feature_snapshot, symmetry_gap,
                odds_at_prediction_time, odds_fighter_id,
                odds_collected_at, odds_n_books
            ) VALUES (
                :bout_id, :model_version, :predicted_prob_red,
                :predicted_winner_id, CAST(:feature_snapshot AS jsonb), :symmetry_gap,
                :odds_at_prediction_time, :odds_fighter_id,
                :odds_collected_at, :odds_n_books
            )
            RETURNING id, created_at
            """
        ),
        {
            "bout_id": prediction.bout_id,
            "model_version": prediction.model_version,
            "predicted_prob_red": prob_rounded,
            "predicted_winner_id": prediction.predicted_winner_id,
            "feature_snapshot": json.dumps(envelope),
            "symmetry_gap": round(prediction.symmetry_gap, 5),
            "odds_at_prediction_time": odds.moneyline if odds else None,
            "odds_fighter_id": odds.fighter_id if odds else None,
            "odds_collected_at": odds.collected_at if odds else None,
            "odds_n_books": odds.n_books if odds else None,
        },
    ).one()

    return LedgerEntry(
        prediction_id=int(row.id),
        bout_id=prediction.bout_id,
        model_version=prediction.model_version,
        created_at=row.created_at,
    )



if __name__ == "__main__":
    # SMOKE TEST — WRITES NOTHING PERMANENT. The ADR-025 trigger blocks
    # DELETE as well as UPDATE, so any row this leaves committed can
    # NEVER be removed by application code. conn.rollback() is called
    # deliberately instead of commit — running this file must leave the
    # ledger exactly as it found it. If a future edit ever swaps that
    # rollback for a commit, this script starts permanently polluting
    # the real ledger with test rows nothing can delete.
    import os

    from api.dependencies import build_engine, load_model_bundle
    from api.services.features import build_bout_features, snapshot_connection
    from api.services.inference import predict_bout
    from api.services.queries import fetch_latest_prediction_for_bout

    pg_engine = build_engine(os.environ["DATABASE_URL"])
    bundle = load_model_bundle(pg_engine)

    with snapshot_connection() as duck_con:
        scheduled = duck_con.execute(
            "SELECT b.id FROM bouts b JOIN events e ON e.id = b.event_id "
            "WHERE b.status = 'scheduled' ORDER BY e.event_date LIMIT 1"
        ).fetchone()
        if scheduled is None:
            raise SystemExit("No scheduled bouts in the snapshot.")
        features = build_bout_features(duck_con, int(scheduled[0]), bundle.feature_order)

    prediction = predict_bout(bundle, features)
    bout_id = int(scheduled[0])

    # What the ledger held BEFORE this run — so the check at the end is
    # "did we leave it as we found it," not "is it empty." After Step 8
    # this bout almost certainly already has a committed prediction.
    before = fetch_latest_prediction_for_bout(pg_engine, bout_id)

    with pg_engine.connect() as conn:
        entry = write_prediction(conn, prediction)
        print(f"would write: prediction_id={entry.prediction_id} "
              f"bout_id={entry.bout_id} model={entry.model_version} "
              f"created_at={entry.created_at}")
        conn.rollback()  # discard — see warning above

    after = fetch_latest_prediction_for_bout(pg_engine, bout_id)

    before_id = before["prediction_id"] if before else None
    after_id = after["prediction_id"] if after else None
    print(f"latest prediction_id before: {before_id}, after: {after_id}")
    print(f"row discarded after rollback: {before_id == after_id}")