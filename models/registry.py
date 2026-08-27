"""
Model registry — Week 3 Saturday (docs/PLAN.md Section 3, ADR-022).

Thin data-access layer over the model_registry table. Kept separate
from models/freeze.py on purpose: freeze.py is a training script (it
imports lightgbm, features/); this is a plain database reader/writer
that Week 4's FastAPI service will import directly on every prediction
request via get_active_model(). An API process has no business
importing a training pipeline just to find out which model file to
load.

Same DB-access pattern as data/ingestion/loaders.py: raw SQLAlchemy
Core (create_engine + Table reflection via autoload_with), not the
ORM — reusing what the ingestion layer already established, not
introducing a second pattern.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, select, update
from sqlalchemy.engine import Engine

load_dotenv()


def _get_engine() -> Engine:
    """Identical construction to data/ingestion/loaders.py's
    _get_engine — intentionally, not a coincidence."""
    return create_engine(os.environ["DATABASE_URL"])


def _reflect(engine: Engine) -> Table:
    metadata = MetaData()
    return Table("model_registry", metadata, autoload_with=engine)


def register_model(engine: Engine, metadata: dict, activate: bool = True) -> int:
    """
    Inserts one row into model_registry from a models/v1/metadata.json
    -shaped dict.

    Deliberately a plain INSERT, not an upsert like loaders.py's
    fighters/events/bouts — those tables legitimately see the same
    real-world row again as ingestion reruns. A model version being
    frozen twice under the same version string is not that situation:
    it almost always means a mistake (re-running freeze.py without
    bumping VERSION), so this fails loudly on a duplicate `version`
    rather than silently overwriting a frozen artifact's row.

    If activate=True, deactivating the current active row and
    activating the new one happens in the SAME transaction as the
    insert — the partial unique index on is_active makes "two active
    rows" impossible at the DB level, but only a single transaction
    makes "zero active rows" (a crash between two separate statements)
    impossible too.

    Returns
    -------
    int — the new row's id.
    """
    tbl = _reflect(engine)

    record = {
        "version": metadata["version"],
        "model_type": metadata["model_type"],
        "artifact_path": metadata["artifact_path"],
        "artifact_sha256": metadata["artifact_sha256"],
        "feature_list": metadata["feature_list"],
        "training_cutoff": date.fromisoformat(metadata["training_cutoff"]),
        "train_row_count": metadata["train_row_count"],
        "train_bout_count": metadata["train_bout_count"],
        "hyperparameters": metadata.get("full_model_params")
        or metadata.get("tuned_hyperparameters"),
        "is_calibrated": metadata.get("is_calibrated", False),
        "git_sha": metadata["git_sha"],
        "metrics": metadata.get("metrics"),
        "is_active": activate,
        "trained_at": datetime.fromisoformat(metadata["trained_at"]),
        "notes": metadata.get("notes"),
    }

    with engine.begin() as conn:
        if activate:
            conn.execute(update(tbl).where(tbl.c.is_active).values(is_active=False))
        result = conn.execute(tbl.insert().values(**record).returning(tbl.c.id))
        return result.scalar_one()


def get_active_model(engine: Engine) -> dict | None:
    """
    Returns the currently active model_registry row as a dict, or
    None if nothing is active yet. This is what Week 4's FastAPI
    inference path calls on every request that needs to know which
    model file to load: get the row, read artifact_path,
    Booster.load_model() it.
    """
    tbl = _reflect(engine)
    with engine.connect() as conn:
        row = conn.execute(select(tbl).where(tbl.c.is_active)).mappings().first()
    return dict(row) if row else None


def get_model(engine: Engine, version: str) -> dict | None:
    """
    Returns a specific model_registry row by version string,
    regardless of active status — used by the prediction ledger to
    look up metadata for a historical prediction's model_version, and
    by anything auditing a past prediction later.
    """
    tbl = _reflect(engine)
    with engine.connect() as conn:
        row = conn.execute(select(tbl).where(tbl.c.version == version)).mappings().first()
    return dict(row) if row else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register a frozen model artifact")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("models/v1/metadata.json"),
        help="path to the metadata.json written by models/freeze.py",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="register without making this the active model",
    )
    args = parser.parse_args()

    meta = json.loads(args.metadata.read_text())
    engine = _get_engine()
    row_id = register_model(engine, meta, activate=not args.no_activate)
    print(
        f"registered {meta['version']} as model_registry.id={row_id} "
        f"(active={not args.no_activate})"
    )