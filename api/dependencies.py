"""
Shared FastAPI dependencies — Week 4 Monday (ADR-023).

Two expensive things exist exactly once per process, created in
api/main.py's lifespan block and stashed on app.state:

  1. the SQLAlchemy Engine (its connection pool)
  2. the loaded LightGBM Booster, plus the feature order it expects

Routers reach them through Depends(get_engine) / Depends(get_model)
rather than importing them directly. That indirection is what makes the
endpoints testable: a test can swap either one out via
app.dependency_overrides without the router knowing.
"""

from dataclasses import dataclass
from typing import Annotated, TypeAlias

from fastapi import Depends, Request
from lightgbm import Booster
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from api.config import Settings, get_settings
from models.registry import get_active_model


@dataclass(frozen=True)
class ModelBundle:
    """
    Everything the inference path (Week 4 Tuesday) needs, resolved once
    at startup.

    feature_order is a tuple, not a list, and frozen=True makes the whole
    bundle immutable — deliberately. LightGBM matches features by
    POSITION, not by name. If anything downstream ever reordered this
    list, the model would keep returning confident, plausible-looking,
    completely wrong probabilities. Making it un-mutable-by-accident is
    cheap insurance against a silent failure.
    """

    version: str
    booster: Booster
    feature_order: tuple[str, ...]
    registry_row: dict


def build_engine(database_url: str) -> Engine:
    """
    Create the one Engine this process will use.

    pool_pre_ping=True is not optional on Neon: Neon suspends idle
    compute, which silently kills pooled connections. Without pre-ping,
    the first request after a quiet period fails on a dead socket.
    Pre-ping costs one trivial round-trip and makes that class of error
    disappear.

    pool_size stays small because Neon's free tier has a modest
    connection ceiling and this service will never need more.
    """
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=300,
    )


def load_model_bundle(engine: Engine) -> ModelBundle:
    """
    Resolve the active model through the registry and load it from disk.

    Note the indirection: the API never hardcodes 'models/v1/model.txt'.
    It asks model_registry which version is active, and loads whatever
    artifact_path that row points at. Promoting v2 later becomes a
    database update plus a restart — no code change, no redeploy of a
    different image.

    Raises if no active model exists or the artifact is missing, so a
    misconfigured deploy fails at boot rather than 500-ing on every
    prediction request.
    """
    row = get_active_model(engine)
    if row is None:
        raise RuntimeError(
            "No active row in model_registry. Run `uv run python -m "
            "models.registry` to register a frozen artifact."
        )

    booster = Booster(model_file=row["artifact_path"])
    return ModelBundle(
        version=row["version"],
        booster=booster,
        feature_order=tuple(row["feature_list"]),
        registry_row=row,
    )


def get_engine(request: Request) -> Engine:
    """Dependency: the app-scoped Engine created during startup."""
    engine: Engine = request.app.state.engine
    return engine


def get_model(request: Request) -> ModelBundle:
    """Dependency: the app-scoped ModelBundle loaded during startup."""
    bundle: ModelBundle = request.app.state.model
    return bundle


# Annotated aliases so routers read as `engine: EngineDep` instead of
# repeating Depends(...) in every signature. This is the current
# FastAPI-recommended style; it also keeps ruff's bugbear rule happy,
# since Depends() lives in the annotation rather than in a mutable
# default argument.
EngineDep: TypeAlias = Annotated[Engine, Depends(get_engine)]
ModelDep: TypeAlias = Annotated[ModelBundle, Depends(get_model)]
SettingsDep: TypeAlias = Annotated[Settings, Depends(get_settings)]