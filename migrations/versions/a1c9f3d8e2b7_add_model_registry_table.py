"""add model_registry table and single-active-version invariant

Adds model_registry — the frozen record of every serialized model
artifact this project ships, starting with Week 3 Saturday's v1
freeze (docs/PLAN.md Section 3, ADR-021).

Two enforcement points at the DB level, same instinct as every prior
migration in this project (ADR-005's CHECK constraints, ADR-009's
wikipedia_pageid):

  1. `version` is UNIQUE — a model version string can be registered
     exactly once, ever. Re-freezing the same version is a mistake,
     not a legitimate update (see models/registry.py's
     register_model, which INSERTs rather than upserts, on purpose).

  2. A PARTIAL UNIQUE INDEX on is_active (WHERE is_active) makes "two
     active models at once" impossible for Postgres to accept,
     regardless of what application code does. Flipping the active
     model is one transaction: deactivate the old row, activate the
     new one (models/registry.py's register_model).

Also adds a foreign key from predictions.model_version to
model_registry.version. predictions is empty as of this migration
(Week 4 hasn't started), so this costs nothing today and guarantees
going forward that no prediction can ever be logged against a model
version that was never registered.

Revision ID: a1c9f3d8e2b7
Revises: bf7cdbcd66ed
Create Date: 2026-08-25 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1c9f3d8e2b7'
down_revision: str | Sequence[str] | None = 'bf7cdbcd66ed'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "model_registry",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("version", sa.Text(), nullable=False, unique=True),
        sa.Column("model_type", sa.Text(), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("artifact_sha256", sa.Text(), nullable=False),
        sa.Column("feature_list", postgresql.JSONB(), nullable=False),
        sa.Column("training_cutoff", sa.Date(), nullable=False),
        sa.Column("train_row_count", sa.Integer(), nullable=False),
        sa.Column("train_bout_count", sa.Integer(), nullable=False),
        sa.Column("hyperparameters", postgresql.JSONB(), nullable=False),
        sa.Column(
            "is_calibrated", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("git_sha", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
    )

    # Singleton invariant: at most one row can have is_active = true.
    # A plain unique index would reject ANY duplicate value, including
    # duplicate `false`s — the postgresql_where filter is what narrows
    # this to only active rows. Same pattern as ix_bouts_status_scheduled.
    op.create_index(
        "ix_model_registry_one_active",
        "model_registry",
        ["is_active"],
        unique=True,
        postgresql_where="is_active",
    )

    op.create_foreign_key(
        "fk_predictions_model_version",
        "predictions",
        "model_registry",
        ["model_version"],
        ["version"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_predictions_model_version", "predictions", type_="foreignkey"
    )
    op.drop_index("ix_model_registry_one_active", table_name="model_registry")
    op.drop_table("model_registry")