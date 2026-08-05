"""add source_url to fighters and bouts

Revision ID: e14a345445b9
Revises: 38e3db2c80a5
Create Date: 2026-08-05 15:20:44.441399

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e14a345445b9'
down_revision: str | Sequence[str] | None = "38e3db2c80a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "fighters", sa.Column("source_url", sa.Text(), nullable=True)
    )
    op.create_unique_constraint(
        "uq_fighters_source_url", "fighters", ["source_url"]
    )

    op.add_column(
        "bouts", sa.Column("source_url", sa.Text(), nullable=True)
    )
    op.create_unique_constraint(
        "uq_bouts_source_url", "bouts", ["source_url"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_bouts_source_url", "bouts", type_="unique")
    op.drop_column("bouts", "source_url")

    op.drop_constraint("uq_fighters_source_url", "fighters", type_="unique")
    op.drop_column("fighters", "source_url")
