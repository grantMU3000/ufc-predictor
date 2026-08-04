"""create initial schema

Revision ID: 38e3db2c80a5
Revises: 
Create Date: 2026-08-04 14:43:00.121102

"""
from typing import Sequence, Union
from sqlalchemy.dialects import postgresql
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '38e3db2c80a5'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fighters",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("real_name", sa.Text(), nullable=False),
        sa.Column("dob", sa.Date(), nullable=True),
        sa.Column("height_cm", sa.Numeric(4, 1), nullable=True),
        sa.Column("reach_cm", sa.Numeric(4, 1), nullable=True),
        sa.Column("stance", sa.Text(), nullable=True),
        sa.Column("nationality", sa.Text(), nullable=True),
        sa.Column("ufc_debut_date", sa.Date(), nullable=True),
    )

    op.create_table(
        "fighter_aliases",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "fighter_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=False,
        ),
        sa.Column("alias_name", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "fighter_id", "alias_name", name="uq_fighter_aliases_fighter_alias"
        ),
    )

    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("event_date", sa.Date(), nullable=False),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
    )

    op.create_table(
        "bouts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "event_id", sa.BigInteger(), sa.ForeignKey("events.id"), nullable=False
        ),
        sa.Column(
            "fighter_red_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=False,
        ),
        sa.Column(
            "fighter_blue_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=False,
        ),
        sa.Column("weight_class", sa.Text(), nullable=False),
        sa.Column(
            "is_title_fight", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("scheduled_rounds", sa.SmallInteger(), nullable=False),
        sa.Column("card_position", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default="scheduled"
        ),
        sa.Column(
            "winner_id", sa.BigInteger(), sa.ForeignKey("fighters.id"), nullable=True
        ),
        sa.Column("method", sa.Text(), nullable=True),
        sa.Column("method_detail", sa.Text(), nullable=True),
        sa.Column("ending_round", sa.SmallInteger(), nullable=True),
        sa.Column("ending_time_seconds", sa.SmallInteger(), nullable=True),
        sa.Column("fighter_red_weigh_in_lbs", sa.Numeric(4, 1), nullable=True),
        sa.Column("fighter_blue_weigh_in_lbs", sa.Numeric(4, 1), nullable=True),
        sa.CheckConstraint(
            "scheduled_rounds IN (3, 5)", name="ck_bouts_scheduled_rounds"
        ),
        sa.CheckConstraint(
            "status IN ('scheduled', 'completed', 'cancelled')",
            name="ck_bouts_status",
        ),
        sa.CheckConstraint(
            "fighter_red_id <> fighter_blue_id", name="ck_bouts_distinct_fighters"
        ),
        sa.CheckConstraint(
            "winner_id IS NULL OR winner_id = fighter_red_id OR winner_id = fighter_blue_id",
            name="ck_bouts_valid_winner"
        )
    )

    op.create_table(
        "bout_stats",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "bout_id", sa.BigInteger(), sa.ForeignKey("bouts.id"), nullable=False
        ),
        sa.Column(
            "fighter_id", sa.BigInteger(), sa.ForeignKey("fighters.id"), nullable=False
        ),
        sa.Column("round_number", sa.SmallInteger(), nullable=False),
        sa.Column("sig_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("sig_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("total_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("total_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("takedowns_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("takedowns_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("sub_attempts", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("knockdowns", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("control_time_seconds", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("head_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("head_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("body_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("body_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("leg_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("leg_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("distance_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("distance_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("clinch_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("clinch_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("ground_strikes_landed", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("ground_strikes_attempted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.UniqueConstraint(
            "bout_id", "fighter_id", "round_number", name="uq_bout_stats_bout_fighter_round"
        ),
    )

    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "bout_id", sa.BigInteger(), sa.ForeignKey("bouts.id"), nullable=False
        ),
        sa.Column("sportsbook", sa.Text(), nullable=False),
        sa.Column(
            "fighter_id", sa.BigInteger(), sa.ForeignKey("fighters.id"), nullable=False
        ),
        sa.Column("moneyline", sa.Integer(), nullable=False),
        sa.Column("implied_prob", sa.Numeric(5, 4), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "bout_id", sa.BigInteger(), sa.ForeignKey("bouts.id"), nullable=False
        ),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("predicted_prob_red", sa.Numeric(5, 4), nullable=False),
        sa.Column(
            "predicted_winner_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=False,
        ),
        sa.Column("feature_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("odds_at_prediction_time", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "predicted_prob_red BETWEEN 0 AND 1", name="ck_predictions_prob_range"
        ),
    )

    op.create_table(
        "prediction_results",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id",
            sa.BigInteger(),
            sa.ForeignKey("predictions.id"),
            nullable=False,
        ),
        sa.Column(
            "actual_winner_id",
            sa.BigInteger(),
            sa.ForeignKey("fighters.id"),
            nullable=False,
        ),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("log_loss_contribution", sa.Numeric(8, 6), nullable=True),
        sa.Column("brier_contribution", sa.Numeric(8, 6), nullable=True),
        sa.Column(
            "settled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("prediction_id", name="uq_prediction_results_prediction_id"),
    )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("prediction_results")
    op.drop_table("predictions")
    op.drop_table("odds_snapshots")
    op.drop_table("bout_stats")
    op.drop_table("bouts")
    op.drop_table("events")
    op.drop_table("fighter_aliases")
    op.drop_table("fighters")
