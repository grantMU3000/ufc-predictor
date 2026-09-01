"""
GET /model/performance — Week 4 Monday, extended Week 4 Thursday (ADR-026).

Serves TWO separate claims in one response:

  test_metrics — the frozen test-set numbers recorded in model_registry at
                 freeze time (ADR-022), filtered to the shipping artifact.
                 Nothing is recomputed. The test set was unlocked exactly
                 once (ADR-020) and those numbers are final.

  live         — the since-deployment track record, aggregated from
                 prediction_results. Recomputed per request, because it
                 legitimately changes every time a card settles.

They stay structurally separate and are never combined into a headline
number. "How it scored on the 2025 holdout" and "how it is doing on fights
since deployment" have different sample sizes, different populations, and
different amounts of trust attached — averaging them would be meaningless.

The live block is two aggregate queries over a table that will hold
thousands of rows at most, so it is cheap enough to compute inline. If that
ever stops being true, it caches at the router, not by freezing numbers
into a column — a stored aggregate is a number that can drift from its
own source.
"""

from fastapi import APIRouter, HTTPException

from api.dependencies import EngineDep, ModelDep
from api.schemas import (
    ErrorDetail,
    LiveMetricRow,
    LivePerformanceBlock,
    MetricRow,
    ModelPerformanceResponse,
    PaperRoi,
    SettlementCounts,
)
from api.services.live_metrics import compute_live_metrics
from api.services.metrics import SHIPPING_ARTIFACT, select_shipping_metrics

router = APIRouter(prefix="/model", tags=["meta"])


@router.get(
    "/performance",
    response_model=ModelPerformanceResponse,
    responses={503: {"model": ErrorDetail, "description": "No active model"}},
)
def model_performance(model: ModelDep, engine: EngineDep) -> ModelPerformanceResponse:
    """Frozen test metrics and the live track record for the active model."""
    row = model.registry_row
    if not row:
        raise HTTPException(status_code=503, detail="No active model registered")

    metrics = [
        MetricRow.model_validate(m) for m in select_shipping_metrics(row.get("metrics"))
    ]

    live = compute_live_metrics(engine)
    live_block = LivePerformanceBlock(
        counts=SettlementCounts(**vars(live.counts)),
        full=[LiveMetricRow(**vars(r)) for r in live.full],
        odds_covered=[LiveMetricRow(**vars(r)) for r in live.odds_covered],
        paper_roi=PaperRoi(**vars(live.paper_roi)),
    )

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
        live=live_block,
    )