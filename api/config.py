"""
API configuration — Week 4 Monday (docs/PLAN.md Section 3, ADR-023).

12-factor config: every environment-specific value arrives through the
environment (or a local .env file), never a hardcoded literal. This is
the same DATABASE_URL that data/ingestion/loaders.py and
models/registry.py already read via os.environ — the difference is that
here it is read once, validated once, and cached, instead of being
re-read on every call.

pydantic-settings does the reading and the type-checking. If
DATABASE_URL is missing the app raises at import time with a clear
error, rather than failing later on the first database call.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the FastAPI service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    app_env: str = "development"

    api_title: str = "UFC Fight Predictor API"
    api_description: str = (
        "Read-only API over the UFC prediction pipeline: upcoming cards, "
        "the immutable prediction ledger, and frozen model metrics. "
        "Not betting advice."
    )
    api_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide Settings singleton.

    @lru_cache means the .env file is parsed exactly once per process,
    no matter how many times this is called. Tests that need different
    settings construct Settings(...) directly and pass it to
    create_app(), bypassing the cache entirely.
    """
    return Settings()