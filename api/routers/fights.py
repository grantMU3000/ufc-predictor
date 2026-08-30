"""
GET /fights/{bout_id}/prediction — Week 4 Monday (STUB).

Today this endpoint only SERVES what is already in the prediction
ledger. It does not compute anything: live inference is Week 4 Tuesday's
job, and the prediction-writing path is Wednesday's.

The three-way behaviour below is deliberate:

  bout does not exist            -> 404
  a ledger entry exists          -> 200, return it verbatim
  bout exists, no prediction yet -> 501 Not Implemented

The 501 is deliberate and permanent, not a placeholder. The tempting
shortcut is returning something like 0.5 so the frontend has a number
to render — do not. This endpoint serves the ledger and only the
ledger: a published probability must always be the one recorded before
the fight, never one recomputed afterward from data that has since
changed. Computing on demand is what would break that guarantee, so
this route never does it (ADR-024 Decision 1). Predictions arrive via
scripts/score_upcoming.py.
"""

from fastapi import APIRouter, HTTPException, Path

from api.dependencies import EngineDep
from api.schemas import ErrorDetail, PredictionResponse
from api.services.queries import fetch_bout, fetch_latest_prediction_for_bout

router = APIRouter(prefix="/fights", tags=["predictions"])


@router.get(
    "/{bout_id}/prediction",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorDetail, "description": "No such bout"},
        501: {"model": ErrorDetail, "description": "Live inference not yet wired"},
    },
)
def fight_prediction(
    engine: EngineDep,
    bout_id: int = Path(ge=1, description="bouts.id"),
) -> PredictionResponse:
    """Return the most recent logged prediction for a bout."""
    bout = fetch_bout(engine, bout_id)
    if bout is None:
        raise HTTPException(status_code=404, detail=f"No bout with id {bout_id}")

    row = fetch_latest_prediction_for_bout(engine, bout_id)
    if row is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "No prediction logged for this bout yet. Predictions are "
                "computed in batch and recorded to the ledger before the "
                "event; this endpoint serves recorded predictions only "
                "and never computes one on demand."
            ),
        )

    return PredictionResponse(
        bout_id=row["bout_id"],
        model_version=row["model_version"],
        predicted_prob_red=float(row["predicted_prob_red"]),
        predicted_winner_id=row["predicted_winner_id"],
        predicted_winner_name=row["predicted_winner_name"],
        odds_at_prediction_time=row["odds_at_prediction_time"],
        created_at=row["created_at"],
    )