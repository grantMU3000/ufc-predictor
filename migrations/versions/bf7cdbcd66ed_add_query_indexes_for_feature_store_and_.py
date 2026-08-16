"""add query indexes for feature store and api

Adds btree indexes on the columns that the point-in-time feature store and the
(future) FastAPI inference path filter and join on.

Postgres automatically indexes primary keys and UNIQUE constraints, but *not*
foreign keys — so every FK in this schema was unindexed before this migration.

Naming convention: ix_<table>_<columns>, matching Alembic/SQLAlchemy defaults.

Revision ID: bf7cdbcd66ed
Revises: f29818945860
Create Date: 2026-08-14 18:35:20.837187

"""
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'bf7cdbcd66ed'
down_revision: str | Sequence[str] | None = 'f29818945860'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add indexes supporting point-in-time feature queries and API reads."""
    # The temporal cutoff column. Every leakage-safe query filters on
    # `event_date < as_of_date`; the API also range-scans it for "next 4 weeks".
    op.create_index("ix_events_event_date", "events", ["event_date"])

    # Corner FKs indexed SEPARATELY, not as a composite. get_prior_bouts() uses
    # `fighter_red_id = :fid OR fighter_blue_id = :fid`, and Postgres can only
    # optimize that via BitmapOr across two independent indexes.
    op.create_index("ix_bouts_fighter_red_id", "bouts", ["fighter_red_id"])
    op.create_index("ix_bouts_fighter_blue_id", "bouts", ["fighter_blue_id"])

    # Join key for bouts -> events, used by every point-in-time query.
    op.create_index("ix_bouts_event_id", "bouts", ["event_id"])

    # Outcome lookups: win rate, finish rate, streaks.
    op.create_index("ix_bouts_winner_id", "bouts", ["winner_id"])

    # Partial index: only ~100 scheduled bouts exist at any time, so this stays
    # tiny and makes the "upcoming fights" API path near-instant. Completed
    # bouts drop out of the index automatically when status changes.
    op.create_index(
        "ix_bouts_status_scheduled",
        "bouts",
        ["status"],
        postgresql_where="status = 'scheduled'",
    )

    # get_prior_bout_stats() filters by fighter. The existing unique constraint
    # (bout_id, fighter_id, round_number) leads with bout_id, so it cannot serve
    # a fighter-first lookup.
    op.create_index("ix_bout_stats_fighter_id", "bout_stats", ["fighter_id"])


def downgrade() -> None:
    """Drop the indexes added in upgrade()."""
    op.drop_index("ix_bout_stats_fighter_id", table_name="bout_stats")
    op.drop_index("ix_bouts_status_scheduled", table_name="bouts")
    op.drop_index("ix_bouts_winner_id", table_name="bouts")
    op.drop_index("ix_bouts_event_id", table_name="bouts")
    op.drop_index("ix_bouts_fighter_blue_id", table_name="bouts")
    op.drop_index("ix_bouts_fighter_red_id", table_name="bouts")
    op.drop_index("ix_events_event_date", table_name="events")
