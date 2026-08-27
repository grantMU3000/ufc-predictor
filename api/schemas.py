"""
Pydantic response models — Week 4 Monday (ADR-023).

These describe the API's public CONTRACT, and they are deliberately
decoupled from the database tables. A migration that renames a column
should force a conscious decision here, not silently change what the
frontend receives. They also generate the OpenAPI schema at /docs for
free.

Note the `protected_namespaces=()` config on models carrying a
`model_version` field: pydantic reserves the `model_` prefix for its own
methods and warns otherwise. Here `model_` means the ML model, so the
reservation is switched off for those specific classes.
"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness + readiness payload. Consumed by Fly.io's health check."""

    status: str = Field(description="'ok' or 'degraded'")
    app_env: str
    api_version: str
    database_reachable: bool
    model_loaded: bool
    active_model_version: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class FighterSummary(BaseModel):
    """Minimal fighter identity for display on a fight card."""

    id: int
    name: str
    stance: str | None = None
    height_cm: float | None = None
    reach_cm: float | None = None


class BoutSummary(BaseModel):
    """A single scheduled bout within an upcoming event."""

    id: int
    weight_class: str
    is_title_fight: bool
    scheduled_rounds: int
    card_position: str | None = None
    status: str
    fighter_red: FighterSummary
    fighter_blue: FighterSummary


class UpcomingEvent(BaseModel):
    """One upcoming card, with its scheduled bouts nested underneath."""

    id: int
    name: str
    event_date: date
    venue: str | None = None
    location: str | None = None
    bouts: list[BoutSummary]


class UpcomingEventsResponse(BaseModel):
    """Envelope for GET /events/upcoming."""

    weeks: int
    window_start: date
    window_end: date
    event_count: int
    events: list[UpcomingEvent]


class PredictionResponse(BaseModel):
    """
    One row of the immutable prediction ledger.

    predicted_winner_id is returned alongside predicted_prob_red on
    purpose. ADR-013 established that a late-replacement swap can flip
    which fighter sits in the red corner between the pre-fight Wikipedia
    row and the post-fight Greco row — so corner position is not a safe
    way to recover which fighter was actually picked. The explicit
    fighter id is.
    """

    bout_id: int
    model_version: str
    predicted_prob_red: float
    predicted_winner_id: int
    predicted_winner_name: str | None = None
    odds_at_prediction_time: int | None = None
    created_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class PredictionHistoryItem(PredictionResponse):
    """A ledger row plus its settlement outcome, if it has been settled."""

    prediction_id: int
    event_name: str | None = None
    event_date: date | None = None
    fighter_red_name: str | None = None
    fighter_blue_name: str | None = None
    settled: bool
    correct: bool | None = None
    actual_winner_id: int | None = None
    settled_at: datetime | None = None


class PredictionHistoryResponse(BaseModel):
    """Paginated envelope for GET /predictions/history."""

    limit: int
    offset: int
    returned: int
    items: list[PredictionHistoryItem]


class MetricRow(BaseModel):
    """
    One evaluation row from the frozen test-set unlock.

    `slice` is which subset of fights ('full', 'odds_covered', 'close')
    and `who` is whose prediction is being scored ('model' or 'market').
    Both are kept so the market comparison travels with the model's own
    numbers — a model number without its baseline is not a claim, it is
    a decoration.
    """

    name: str
    slice_name: str = Field(alias="slice")
    who: str
    n: int
    n_bouts: int | None = None
    accuracy: float
    log_loss: float
    brier: float
    ece: float

    model_config = ConfigDict(populate_by_name=True)


class ModelPerformanceResponse(BaseModel):
    """
    Envelope for GET /model/performance.

    Today this serves the frozen test-set metrics recorded at freeze
    time. Week 4 Thursday's settlement job adds live rolling metrics
    from prediction_results; this response grows a second block then.
    """

    version: str
    model_type: str
    training_cutoff: date
    train_row_count: int
    train_bout_count: int
    feature_count: int
    is_calibrated: bool
    git_sha: str | None = None
    trained_at: datetime | None = None
    shipping_artifact: str
    test_metrics: list[MetricRow]

    model_config = ConfigDict(protected_namespaces=())


class ErrorDetail(BaseModel):
    """Uniform error body, so the frontend has one shape to handle."""

    detail: str