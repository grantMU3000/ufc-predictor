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

import pytest
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import create_app
from models.registry import get_active_model


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