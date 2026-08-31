"""add ledger columns and predictions immutability

Revision ID: c3f7a9d15b42
Revises: a1c9f3d8e2b7
Create Date: 2026-08-30

Week 4 Wednesday, ADR-025. Three changes to `predictions`:

1. Odds provenance columns. The table shipped with a bare
   `odds_at_prediction_time` integer that names no fighter, no
   timestamp, and no book count. A moneyline without a fighter is
   ambiguous, and defaulting it to "red" would key the ledger on
   corner position — which ADR-013 forbids, because a late
   replacement can flip corners between the pre-fight Wikipedia row
   and the post-fight Greco row.

2. `symmetry_gap`, typed rather than buried in the JSONB envelope.
   It is a standing drift diagnostic (ADR-024 Decision 2) that gets
   aggregated across rows, and unpacking JSONB for every
   `max(symmetry_gap)` is friction.

3. An append-only trigger. Immutability enforced by the database, not
   by everyone remembering not to write an UPDATE. Same reasoning as
   ADR-005's CHECK constraints and ADR-022's partial unique index:
   make the invalid state impossible rather than merely discouraged.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3f7a9d15b42"
down_revision: str | Sequence[str] | None = "a1c9f3d8e2b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add odds provenance, symmetry_gap, read indexes, and the append-only trigger."""
    # --- Odds provenance -------------------------------------------------
    # All nullable: odds_snapshots currently holds nothing past
    # 2026-08-01, so every row written today has NULL here. The
    # resolver ships correct and dormant until Friday's refresh job.
    op.add_column(
        "predictions",
        sa.Column(
            "odds_fighter_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=True,
        ),
    )
    op.add_column(
        "predictions",
        sa.Column("odds_collected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("odds_n_books", sa.SmallInteger(), nullable=True),
    )

    # A moneyline that names no fighter is unusable. Either all three
    # odds fields are present or none are — enforced, not assumed.
    op.create_check_constraint(
        "ck_predictions_odds_complete",
        "predictions",
        "(odds_at_prediction_time IS NULL AND odds_fighter_id IS NULL "
        " AND odds_collected_at IS NULL AND odds_n_books IS NULL) "
        "OR (odds_at_prediction_time IS NOT NULL AND odds_fighter_id IS NOT NULL "
        "    AND odds_collected_at IS NOT NULL AND odds_n_books IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_predictions_odds_n_books_positive",
        "predictions",
        "odds_n_books IS NULL OR odds_n_books > 0",
    )

    # --- Symmetry diagnostic ---------------------------------------------
    # |p_A - (1 - p_B)| is a magnitude, so it cannot be negative, and
    # it cannot exceed 1. Numeric(6,5) holds the full range with room.
    op.add_column(
        "predictions",
        sa.Column("symmetry_gap", sa.Numeric(6, 5), nullable=True),
    )
    op.create_check_constraint(
        "ck_predictions_symmetry_gap_range",
        "predictions",
        "symmetry_gap IS NULL OR symmetry_gap BETWEEN 0 AND 1",
    )

    # --- Read indexes ----------------------------------------------------
    # "Latest prediction for this bout" is every read the API does.
    op.create_index(
        "ix_predictions_bout_created",
        "predictions",
        ["bout_id", sa.text("created_at DESC")],
    )
    # Thursday's rolling metrics scan one model version over a window.
    op.create_index(
        "ix_predictions_model_created",
        "predictions",
        ["model_version", "created_at"],
    )

    # --- Append-only enforcement -----------------------------------------
    # Re-predicting a bout inserts a new row; nothing ever edits an
    # old one. restrict_violation (23001) is a distinct SQLSTATE, so
    # the test asserts on the error class rather than string-matching
    # the message.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION predictions_reject_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION
                'predictions is append-only (ADR-025): % denied on prediction id %',
                TG_OP, OLD.id
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_predictions_append_only
        BEFORE UPDATE OR DELETE ON predictions
        FOR EACH ROW EXECUTE FUNCTION predictions_reject_mutation();
        """
    )


def downgrade() -> None:
    """Reverse in strict creation order — trigger first, or the drops are blocked."""
    op.execute("DROP TRIGGER IF EXISTS trg_predictions_append_only ON predictions;")
    op.execute("DROP FUNCTION IF EXISTS predictions_reject_mutation();")

    op.drop_index("ix_predictions_model_created", table_name="predictions")
    op.drop_index("ix_predictions_bout_created", table_name="predictions")

    op.drop_constraint(
        "ck_predictions_symmetry_gap_range", "predictions", type_="check"
    )
    op.drop_column("predictions", "symmetry_gap")

    op.drop_constraint(
        "ck_predictions_odds_n_books_positive", "predictions", type_="check"
    )
    op.drop_constraint("ck_predictions_odds_complete", "predictions", type_="check")
    op.drop_column("predictions", "odds_n_books")
    op.drop_column("predictions", "odds_collected_at")
    op.drop_column("predictions", "odds_fighter_id")