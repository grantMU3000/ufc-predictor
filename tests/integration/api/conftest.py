"""
tests/integration/api/conftest.py

Fixtures for the FastAPI integration tests.

ELI5: tests/test_api_metrics.py checks "does the filing clerk sort the
paperwork correctly." These tests check "does the actual front desk open
in the morning" -- they boot the real app, against a real (disposable)
database, with the real frozen model file loaded off disk.

That cannot be faked usefully. The whole point of Week 4 Monday's
lifespan wiring is that a connection pool opens and a model
deserializes at startup; a mock would test the mock.

Setup required (same as tests/integration/conftest.py one level up):
    TEST_DATABASE_URL -> a disposable Postgres database, ideally a Neon
    branch. NEVER point this at production.

Every fixture here SKIPS rather than fails when prerequisites are
missing, so `uv run pytest` stays green on a fresh clone.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text
import pytest
from fastapi.testclient import TestClient
from datetime import date

from api.config import Settings
from api.main import create_app
from models.registry import get_active_model
from api.dependencies import load_model_bundle
from api.services.features import BoutFeatureMatrix, DataCoverage
from api.services.inference import predict_bout
from api.services.ledger import write_prediction

# A feature that's genuinely NaN in real data often enough to matter —
# ~65% of training rows per docs/RESULTS.md, and the exact column that
# caused the Step 5 dtype bug. Deliberately NaN in the fixture so the
# None round-trip through JSONB gets exercised, not assumed.
NAN_FEATURE = "diff_submission_success_rate"

@pytest.fixture(scope="session")
def api_settings():
    """Settings pointed at the throwaway test database, not .env's."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set -- skipping API integration tests")
    return Settings(database_url=url)


@pytest.fixture(scope="session")
def api_client(api_settings, db_engine):
    """
    A TestClient wrapping the real app.

    Used as a CONTEXT MANAGER on purpose: `with TestClient(app)` is what
    triggers Starlette's lifespan, which is exactly the startup path
    under test (engine built, registry queried, Booster loaded). A bare
    TestClient(app) skips lifespan entirely and would leave app.state
    empty.

    Skips if the test database has no active model row, or if the
    artifact that row points at is not on disk -- both are environment
    setup problems, not code failures.
    """
    row = get_active_model(db_engine)
    if row is None:
        pytest.skip("no active model_registry row in the test database")
    if not Path(row["artifact_path"]).exists():
        pytest.skip(f"model artifact missing at {row['artifact_path']}")

    app = create_app(api_settings)
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def append_only_trigger(db_engine):
    """
    Skip anything touching the ledger if migration c3f7a9d15b42 hasn't
    been applied to the TEST database.

    Easy to forget: `alembic upgrade head` runs against DATABASE_URL,
    not TEST_DATABASE_URL. Without this, the immutability tests would
    fail with "expected an exception, got none" — which reads like the
    trigger is broken rather than absent.
    """
    with db_engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT 1 FROM pg_trigger "
                "WHERE tgname = 'trg_predictions_append_only'"
            )
        ).first()
    if exists is None:
        pytest.skip(
            "trg_predictions_append_only missing -- run `alembic upgrade head` "
            "against TEST_DATABASE_URL"
        )


@pytest.fixture(scope="session")
def model_bundle(db_engine):
    """
    The real frozen model, resolved through the test database's
    registry — same path api/main.py's lifespan uses at startup.

    Session-scoped: deserializing the Booster is the expensive part,
    and nothing in these tests mutates it (ModelBundle is frozen).
    """
    from pathlib import Path

    from models.registry import get_active_model

    row = get_active_model(db_engine)
    if row is None:
        pytest.skip("no active model_registry row in the test database")
    if not Path(row["artifact_path"]).exists():
        pytest.skip(f"model artifact missing at {row['artifact_path']}")
    return load_model_bundle(db_engine)

@pytest.fixture
def synthetic_feature_matrix(model_bundle, sample_bout, sample_fighters):
    """
    A BoutFeatureMatrix built by hand, not read from the Parquet
    snapshot.

    WHY SYNTHETIC. The obvious alternative is scoring a real scheduled
    bout through build_bout_features(). That works today and breaks
    later: the snapshot is built from PRODUCTION Postgres, while these
    tests run against a disposable Neon branch. Whether any given real
    bout_id exists in the branch depends on when it was forked, so the
    test would start failing for a reason that has nothing to do with
    the code under test.

    IDs are still real throwaway rows (sample_bout, sample_fighters),
    because predictions.bout_id and .predicted_winner_id are NOT NULL
    foreign keys. Only the feature VALUES are invented.

    Row 1 is the exact negation of row 0 — that's algebra, not a model
    property (diff = self - opp, and the rows swap self/opp), and it's
    the same invariant test_inference_parity.py asserts on real data.
    """
    red_id, blue_id, _swap_id = sample_fighters
    order = model_bundle.feature_order

    # Deterministic, modest, non-degenerate values. Exact magnitudes
    # don't matter — the replay test checks that a stored vector
    # reproduces its own probability, not that the probability is
    # correct in any domain sense.
    values = np.linspace(-1.5, 1.5, len(order))
    if NAN_FEATURE in order:
        values[order.index(NAN_FEATURE)] = np.nan

    matrix = pd.DataFrame(
        [values, -values], columns=list(order), dtype="float64"
    )

    return BoutFeatureMatrix(
        bout_id=sample_bout,
        fighter_red_id=red_id,
        fighter_blue_id=blue_id,
        as_of_date=date(2099, 1, 1),  # matches sample_event's date
        matrix=matrix,
        coverage=DataCoverage(
            n_features_present=int(matrix.iloc[0].notna().sum()),
            n_features_total=len(order),
            red_prior_bouts=12,
            blue_prior_bouts=7,
        ),
    )


@pytest.fixture
def sample_bout_prediction(model_bundle, synthetic_feature_matrix):
    """
    A real BoutPrediction — real Booster, real TreeSHAP, real
    reconstruction invariant — over synthetic inputs.

    top_n = every feature, matching what scripts/score_upcoming.py
    does in write mode (ADR-025 Decision 1: the ledger stores all 32,
    not the display top-5).
    """
    return predict_bout(
        model_bundle,
        synthetic_feature_matrix,
        top_n=len(model_bundle.feature_order),
    )

@pytest.fixture
def written_prediction(db_engine, append_only_trigger, sample_bout_prediction):
    """
    Commit one prediction to the ledger, yield it, then force-delete.

    THE TRIGGER MUST BE DISABLED TO CLEAN UP. That's not a workaround
    for a design flaw — it's the design working. Application code can
    never delete a prediction; only a table owner doing explicit DDL
    can, which is exactly the bar ADR-025 Decision 3 wanted. This is
    the one place in the repo allowed to do it, and only for test
    isolation.
    """
    with db_engine.connect() as conn:
        entry = write_prediction(conn, sample_bout_prediction)
        conn.commit()

    yield entry, sample_bout_prediction

    with db_engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE predictions DISABLE TRIGGER trg_predictions_append_only")
        )
        conn.execute(
            text("DELETE FROM predictions WHERE id = :id"),
            {"id": entry.prediction_id},
        )
        conn.execute(
            text("ALTER TABLE predictions ENABLE TRIGGER trg_predictions_append_only")
        )