"""add wikipedia_pageid to events

Revision ID: 9cd265588036
Revises: 873526c2bf73
Create Date: 2026-08-10 13:35:04.052122

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9cd265588036'
down_revision: str | Sequence[str] | None = '873526c2bf73'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "events", sa.Column("wikipedia_pageid", sa.BigInteger(), nullable=True)
    )
    op.create_unique_constraint(
        "uq_events_wikipedia_pageid", "events", ["wikipedia_pageid"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_events_wikipedia_pageid", "events", type_="unique")
    op.drop_column("events", "wikipedia_pageid")
