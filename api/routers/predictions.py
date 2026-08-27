"""
GET /predictions/history — Week 4 Monday.

The public track record. Every prediction ever written, newest first,
with its settlement outcome attached where one exists.

This is project goal #6 and the thing that makes the whole project
credible: anyone can see that a prediction was published with a
timestamp before the fight, and whether it was right. Predictions that
have not been settled yet are INCLUDED, not filtered out — hiding open
positions is how a track record becomes marketing.
"""

from fastapi import APIRouter, Query

from api.dependencies import EngineDep
from api.schemas import PredictionHistoryItem, PredictionHistoryResponse
from api.services.queries import fetch_prediction_history

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/history", response_model=PredictionHistoryResponse)
def prediction_history(
    engine: EngineDep,
    limit: int = Query(default=50, ge=1, le=200, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Rows to skip."),
) -> PredictionHistoryResponse:
    """
    Return one page of the prediction ledger.

    Offset pagination is used rather than a cursor because the ledger is
    small (hundreds of rows this year) and offset is trivial for a
    frontend to drive. If it ever grows to the point where deep offsets
    get slow, a keyset cursor on (created_at, id) is the upgrade — the
    ORDER BY is already written to support it.
    """
    rows = fetch_prediction_history(engine, limit=limit, offset=offset)

    items = [
        PredictionHistoryItem(
            prediction_id=row["prediction_id"],
            bout_id=row["bout_id"],
            model_version=row["model_version"],
            predicted_prob_red=float(row["predicted_prob_red"]),
            predicted_winner_id=row["predicted_winner_id"],
            predicted_winner_name=row["predicted_winner_name"],
            odds_at_prediction_time=row["odds_at_prediction_time"],
            created_at=row["created_at"],
            event_name=row["event_name"],
            event_date=row["event_date"],
            fighter_red_name=row["fighter_red_name"],
            fighter_blue_name=row["fighter_blue_name"],
            # result_id is NULL for an unsettled prediction — that NULL is
            # the signal, not an error.
            settled=row["result_id"] is not None,
            correct=row["correct"],
            actual_winner_id=row["actual_winner_id"],
            settled_at=row["settled_at"],
        )
        for row in rows
    ]

    return PredictionHistoryResponse(
        limit=limit, offset=offset, returned=len(items), items=items
    )