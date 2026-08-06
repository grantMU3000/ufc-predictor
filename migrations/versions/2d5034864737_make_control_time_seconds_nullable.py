"""make control_time_seconds nullable

Revision ID: 2d5034864737
Revises: 5442556c70dd
Create Date: 2026-08-06 16:12:28.623714

"""
from collections.abc import Sequence

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '2d5034864737'
down_revision: str | Sequence[str] | None  = '5442556c70dd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "bout_stats", "control_time_seconds",
        nullable=True, server_default=None,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "bout_stats", "control_time_seconds",
        nullable=False, server_default="0",
    )
