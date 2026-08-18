"""add unique constraint to odds_snapshots

Revision ID: 873526c2bf73
Revises: 2d5034864737
Create Date: 2026-08-07 14:57:45.273421

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "873526c2bf73"
down_revision: str | Sequence[str] | None = "2d5034864737"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_odds_snapshots_natural_key",
        "odds_snapshots",
        ["bout_id", "fighter_id", "sportsbook", "collected_at"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_odds_snapshots_natural_key", "odds_snapshots", type_="unique"
    )
