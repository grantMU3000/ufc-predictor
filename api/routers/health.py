"""
GET /health — Week 4 Monday.

This is the endpoint Fly.io will poll on Week 4 Saturday to decide
whether a deploy is healthy. A hardcoded {"ok": true} would happily
report health while the database was on fire, so this actually checks
both dependencies and returns 503 if either is down. A load balancer
can only route around a broken instance if the instance is honest about
being broken.
"""

from fastapi import APIRouter, Response

from api.dependencies import EngineDep, ModelDep, SettingsDep
from api.schemas import HealthResponse
from api.services.queries import check_database

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    engine: EngineDep,
    model: ModelDep,
    settings: SettingsDep,
) -> HealthResponse:
    """Report database reachability and the active model version."""
    db_ok = check_database(engine)
    model_ok = model is not None and model.booster is not None

    if not (db_ok and model_ok):
        response.status_code = 503

    return HealthResponse(
        status="ok" if (db_ok and model_ok) else "degraded",
        app_env=settings.app_env,
        api_version=settings.api_version,
        database_reachable=db_ok,
        model_loaded=model_ok,
        active_model_version=model.version if model_ok else None,
    )