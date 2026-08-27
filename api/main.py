"""
FastAPI application — Week 4 Monday (docs/PLAN.md Section 3, ADR-023).

Two things worth understanding here.

1. LIFESPAN. The database engine and the LightGBM model are created ONCE
   when the process starts, not per request. Opening a connection pool
   and deserializing a model on every request would add latency to every
   call for no benefit. `lifespan` is FastAPI's hook for "do this on
   startup, undo it on shutdown"; the objects live on `app.state` and
   reach routers through Depends() (see api/dependencies.py).

2. SYNC ROUTES. Every endpoint in this service is a plain `def`, never
   `async def`. The whole data layer (models/registry.py,
   data/ingestion/loaders.py) is synchronous SQLAlchemy. Calling a
   blocking function inside an `async def` route stalls the event loop
   and freezes EVERY concurrent request, not just the slow one. With
   plain `def`, FastAPI runs the handler in a worker thread and the loop
   stays free. Full reasoning: ADR-023.

Run locally:
    uv run uvicorn api.main:app --reload
Then open http://localhost:8000/docs
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import Settings, get_settings
from api.dependencies import build_engine, load_model_bundle
from api.routers import events, fights, health, model, predictions


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Build the FastAPI application.

    A factory rather than a bare module-level app so tests can construct
    an instance pointed at a throwaway database (by passing their own
    Settings) without mutating global state or the .env file.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """
        Startup: open the pool, resolve the active model, load it.
        Shutdown: return every pooled connection to Neon.

        Deliberately NOT wrapped in try/except. If the database is
        unreachable or no model is registered, the process should fail
        to start. A service that boots "successfully" and then 500s on
        every request is worse than one that refuses to boot — Fly.io
        automatically rolls back a deploy whose health check never comes
        up, and that safety net only works if failure is loud.
        """
        engine = build_engine(settings.database_url)
        app.state.engine = engine
        app.state.model = load_model_bundle(engine)
        app.state.settings = settings
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
    )

    # get_settings is @lru_cache'd and reads the environment, which would
    # ignore any Settings passed into create_app(). Overriding it here
    # keeps a test-constructed app fully self-consistent.
    app.dependency_overrides[get_settings] = lambda: settings

    app.include_router(health.router)
    app.include_router(events.router)
    app.include_router(fights.router)
    app.include_router(predictions.router)
    app.include_router(model.router)

    return app


app = create_app()