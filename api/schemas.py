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

class CoverageDetail(BaseModel):
    """
    How much fighter history backed a prediction — ADR-024 Decision 4.

    Surfaced in the API rather than kept internal because a 6-of-32
    prediction and a 32-of-32 prediction look identical once they are
    both a probability. The frontend needs this to caption one
    honestly. It never gates anything.
    """

    n_features_present: int
    n_features_total: int
    fraction: float
    red_prior_bouts: int
    blue_prior_bouts: int


class FeatureContributionDetail(BaseModel):
    """
    One feature's push on this prediction, from the RED corner's view.

    log_odds is a RAW TreeSHAP value, not percentage points (ADR-024
    Decision 3). The same +0.4 moves a coinflip a long way and a 90/10
    barely at all, so it must not be rendered as "added 8% to his
    chances." `favors` plus relative magnitude is enough to phrase it
    directionally without lying.
    """

    feature: str
    log_odds: float
    feature_value: float | None = None
    favors: str

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
    odds_fighter_id: int | None = Field(
        default=None,
        description=(
            "Which fighter odds_at_prediction_time refers to. Never assume "
            "the red corner — a late replacement can flip corners (ADR-013)."
        ),
    )
    odds_collected_at: datetime | None = None
    odds_n_books: int | None = Field(
        default=None,
        description="Sportsbooks behind the consensus. Under 3 is a thin market.",
    )
    symmetry_gap: float | None = Field(
        default=None,
        description=(
            "|p_A - (1 - p_B)| across both corner orderings. Near zero means "
            "the model ignored corner position, as symmetrized training intended."
        ),
    )
    bout_status: str | None = Field(
        default=None,
        description=(
            "Live bouts.status, joined at read time. 'cancelled' means the "
            "prediction stands but will never be settled (ADR-025 Decision 5)."
        ),
    )
    created_at: datetime

    model_config = ConfigDict(protected_namespaces=())


class PredictionDetail(PredictionResponse):
    """
    A single ledger row with its full explanation attached.

    Separate from PredictionResponse so the history endpoint stays
    light: contributions is 32 entries per prediction, which is right
    for one bout and wasteful for a 50-row page.
    """

    is_calibrated: bool = Field(
        description=(
            "v1 ships uncalibrated — both calibrators were rejected against "
            "pre-registered gates (ADR-017). Read from the registry, so a "
            "calibrated v2 flips this with no code change."
        )
    )
    coverage: CoverageDetail | None = None
    contributions: list[FeatureContributionDetail] = []
    

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