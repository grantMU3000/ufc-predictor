"""add reversals to bout_stats

Revision ID: e849068ec219
Revises: e14a345445b9
Create Date: 2026-08-05 15:58:07.345781

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e849068ec219'
down_revision: str | Sequence[str] | None = "e14a345445b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "bout_stats",
        sa.Column(
            "reversals", sa.SmallInteger(), nullable=False, server_default="0"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("bout_stats", "reversals")
