"""add unique constraint to events source_url

Revision ID: 5442556c70dd
Revises: e849068ec219
Create Date: 2026-08-06 15:47:44.095638

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5442556c70dd'
down_revision: str | Sequence[str] | None = "e849068ec219"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_unique_constraint(
        "uq_events_source_url", "events", ["source_url"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_events_source_url", "events", type_="unique")