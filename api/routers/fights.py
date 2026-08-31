"""
GET /fights/{bout_id}/prediction — Week 4 Monday, completed Wednesday
(ADR-025).

Serves the immutable prediction ledger, and only the ledger. This route
never computes a prediction on demand, and that is permanent design,
not a limitation (ADR-024 Decision 1): a published probability must
always be the one recorded before the fight, never one recomputed
afterward from data that has since changed. Recomputing per request
would let a published number drift silently as new results land, which
destroys the entire audit trail. Predictions arrive via
`uv run python -m scripts.score_upcoming --write`.

Behaviour:

  bout does not exist            -> 404
  bout exists, no prediction yet -> 404
  a ledger entry exists          -> 200, the recorded row

THE SECOND 404 USED TO BE A 501, and that was correct while the ledger
was empty — 501 means the server lacks the capability, which was true
through Tuesday. The ledger exists and is populated now, so a bout with
no prediction row is a MISSING RESOURCE (cancelled bout, or one outside
the scoring window), not a missing feature. 404 is the honest code
(ADR-025 Decision 6).

The tempting shortcut is still to return 0.5 so the frontend has
something to render. Don't. A fabricated probability in a ledger-backed
endpoint is indistinguishable from a real one downstream.
"""

from fastapi import APIRouter, HTTPException, Path, Query

from api.dependencies import EngineDep
from api.schemas import (
    CoverageDetail,
    ErrorDetail,
    FeatureContributionDetail,
    PredictionDetail,
)
from api.services.queries import fetch_bout, fetch_latest_prediction_for_bout

router = APIRouter(prefix="/fights", tags=["predictions"])

# Enough to explain a fight without burying the reader. The ledger
# stores all 32 (ADR-025 Decision 1) and rank_contributions() already
# sorted them by |impact|, so slicing the front is free and correct.
DEFAULT_CONTRIBUTIONS = 5


@router.get(
    "/{bout_id}/prediction",
    response_model=PredictionDetail,
    responses={
        404: {"model": ErrorDetail, "description": "No such bout, or no prediction"},
    },
)
def fight_prediction(
    engine: EngineDep,
    bout_id: int = Path(ge=1, description="bouts.id"),
    contributions: int = Query(
        default=DEFAULT_CONTRIBUTIONS,
        ge=0,
        le=32,
        description="Top-N feature contributions to return, by absolute impact.",
    ),
) -> PredictionDetail:
    """Return the most recent logged prediction for a bout, with its explanation."""
    bout = fetch_bout(engine, bout_id)
    if bout is None:
        raise HTTPException(status_code=404, detail=f"No bout with id {bout_id}")

    row = fetch_latest_prediction_for_bout(engine, bout_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No prediction logged for bout {bout_id}. Predictions are "
                "computed in batch and recorded to the ledger before the "
                "event; this endpoint serves recorded predictions only and "
                "never computes one on demand."
            ),
        )

    # Everything below the typed columns lives in the feature_snapshot
    # envelope — the self-sufficient record written at prediction time.
    snapshot = row["feature_snapshot"] or {}

    coverage_raw = snapshot.get("coverage")
    coverage = CoverageDetail(**coverage_raw) if coverage_raw else None

    # Already sorted by |log_odds| when written, so the head of the
    # list is the top-N by impact without re-sorting here.
    ranked = [
        FeatureContributionDetail(**c)
        for c in snapshot.get("contributions", [])[:contributions]
    ]

    return PredictionDetail(
        bout_id=row["bout_id"],
        model_version=row["model_version"],
        predicted_prob_red=float(row["predicted_prob_red"]),
        predicted_winner_id=row["predicted_winner_id"],
        predicted_winner_name=row["predicted_winner_name"],
        odds_at_prediction_time=row["odds_at_prediction_time"],
        odds_fighter_id=row["odds_fighter_id"],
        odds_collected_at=row["odds_collected_at"],
        odds_n_books=row["odds_n_books"],
        symmetry_gap=(
            float(row["symmetry_gap"]) if row["symmetry_gap"] is not None else None
        ),
        bout_status=row["bout_status"],
        created_at=row["created_at"],
        is_calibrated=bool(snapshot.get("is_calibrated", False)),
        coverage=coverage,
        contributions=ranked,
    )