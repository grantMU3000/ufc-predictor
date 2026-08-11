"""Adding rounds confirmed column

Revision ID: f29818945860
Revises: 9cd265588036
Create Date: 2026-08-11 11:42:30.841268

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f29818945860'
down_revision: str | Sequence[str] | None = '9cd265588036'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "bouts", sa.Column("rounds_confirmed", sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "bouts", sa.Column("rounds_confirmed", sa.Boolean(), nullable=False, server_default=sa.false())
    )
