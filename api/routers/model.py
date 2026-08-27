"""
GET /model/performance — Week 4 Monday.

Serves the frozen test-set numbers recorded in model_registry at freeze
time (ADR-022), filtered to the shipping artifact only.

Nothing is recomputed here. The test set was unlocked exactly once
(ADR-020) and those numbers are final. This endpoint reads them out of
the registry row; it never touches a parquet file or a model.

Week 4 Thursday's settlement job will add a second, LIVE block to this
response, computed from prediction_results. The two must stay clearly
labelled and separate — "how it scored on the 2025 holdout" and "how it
is doing on fights since deployment" are different claims.
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import ModelDep
from api.schemas import ErrorDetail, MetricRow, ModelPerformanceResponse
from api.services.metrics import SHIPPING_ARTIFACT, select_shipping_metrics

router = APIRouter(prefix="/model", tags=["meta"])


@router.get(
    "/performance",
    response_model=ModelPerformanceResponse,
    responses={503: {"model": ErrorDetail, "description": "No active model"}},
)
def model_performance(model: ModelDep) -> ModelPerformanceResponse:
    """Return metadata and frozen test metrics for the active model."""
    row = model.registry_row
    if not row:
        raise HTTPException(status_code=503, detail="No active model registered")

    metrics = [
        MetricRow.model_validate(m) for m in select_shipping_metrics(row.get("metrics"))
    ]

    return ModelPerformanceResponse(
        version=row["version"],
        model_type=row["model_type"],
        training_cutoff=row["training_cutoff"],
        train_row_count=row["train_row_count"],
        train_bout_count=row["train_bout_count"],
        feature_count=len(model.feature_order),
        is_calibrated=row["is_calibrated"],
        git_sha=row.get("git_sha"),
        trained_at=row.get("trained_at"),
        shipping_artifact=SHIPPING_ARTIFACT,
        test_metrics=metrics,
    )