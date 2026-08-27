"""
Read-only SQL for the API — Week 4 Monday (ADR-023).

Every function here takes an Engine and returns plain dicts. No Pydantic,
no FastAPI imports: keeping the SQL layer framework-free means these can
be exercised from a plain script or a test without spinning up an app.

Raw text() SQL rather than SQLAlchemy Core expression building, matching
what tests/integration/ already does. These are multi-table joins whose
shape matters for the query planner; written out, it is obvious which
indexes (added in migration bf7cdbcd66ed) each one is relying on.

Date windows are computed in PYTHON and passed as bind parameters rather
than using Postgres' CURRENT_DATE. That keeps "what does 'upcoming' mean"
in one place, and makes the queries trivially testable at any date.
"""

from collections.abc import Sequence
from datetime import date

from sqlalchemy import text
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import SQLAlchemyError


def check_database(engine: Engine) -> bool:
    """
    Cheapest possible round-trip, used by GET /health.

    Returns False instead of raising: a health endpoint's job is to
    REPORT that the database is unreachable, not to crash while trying
    to say so. SQLAlchemyError (not bare Exception) so a genuine bug in
    this function still surfaces as a 500 instead of being swallowed as
    "database down".
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


def fetch_upcoming_bouts(
    engine: Engine, window_start: date, window_end: date
) -> Sequence[RowMapping]:
    """
    Every scheduled bout on an event dated within [window_start,
    window_end], one row per bout, with both fighters joined in.

    Returns a flat row list; grouping into events happens in the router.
    Doing the grouping in Python rather than with a Postgres JSON
    aggregate keeps the SQL readable and the row count here is tiny
    (roughly 12 bouts x a handful of cards).

    Index path: ix_events_event_date for the date range, then
    ix_bouts_event_id for the join, and the partial index
    ix_bouts_status_scheduled keeps the status filter nearly free.
    """
    stmt = text("""
        SELECT
            e.id            AS event_id,
            e.name          AS event_name,
            e.event_date    AS event_date,
            e.venue         AS venue,
            e.location      AS location,
            b.id            AS bout_id,
            b.weight_class  AS weight_class,
            b.is_title_fight AS is_title_fight,
            b.scheduled_rounds AS scheduled_rounds,
            b.card_position AS card_position,
            b.status        AS status,
            fr.id           AS red_id,
            fr.real_name    AS red_name,
            fr.stance       AS red_stance,
            fr.height_cm    AS red_height_cm,
            fr.reach_cm     AS red_reach_cm,
            fb.id           AS blue_id,
            fb.real_name    AS blue_name,
            fb.stance       AS blue_stance,
            fb.height_cm    AS blue_height_cm,
            fb.reach_cm     AS blue_reach_cm
        FROM events e
        JOIN bouts b       ON b.event_id = e.id
        JOIN fighters fr   ON fr.id = b.fighter_red_id
        JOIN fighters fb   ON fb.id = b.fighter_blue_id
        WHERE e.event_date BETWEEN :window_start AND :window_end
          AND b.status = 'scheduled'
        ORDER BY e.event_date ASC, e.id ASC, b.id ASC
    """)
    with engine.connect() as conn:
        return (
            conn.execute(stmt, {"window_start": window_start, "window_end": window_end})
            .mappings()
            .all()
        )


def fetch_bout(engine: Engine, bout_id: int) -> RowMapping | None:
    """One bout by id, or None if it does not exist."""
    stmt = text("""
        SELECT b.id, b.event_id, b.status,
               b.fighter_red_id, b.fighter_blue_id
        FROM bouts b
        WHERE b.id = :bout_id
    """)
    with engine.connect() as conn:
        return conn.execute(stmt, {"bout_id": bout_id}).mappings().first()


def fetch_latest_prediction_for_bout(engine: Engine, bout_id: int) -> RowMapping | None:
    """
    The most recent ledger entry for a bout, or None if none exists yet.

    ORDER BY created_at DESC, id DESC: the id tiebreak matters because
    two predictions written inside the same transaction can share a
    timestamp, and "most recent" must be deterministic.

    The ledger is append-only — nothing here updates a prediction row.
    Re-predicting a bout writes a NEW row; the old one stays as the
    historical record of what was believed at that moment.
    """
    stmt = text("""
        SELECT
            p.id                       AS prediction_id,
            p.bout_id                  AS bout_id,
            p.model_version            AS model_version,
            p.predicted_prob_red       AS predicted_prob_red,
            p.predicted_winner_id      AS predicted_winner_id,
            p.odds_at_prediction_time  AS odds_at_prediction_time,
            p.created_at               AS created_at,
            w.real_name                AS predicted_winner_name
        FROM predictions p
        LEFT JOIN fighters w ON w.id = p.predicted_winner_id
        WHERE p.bout_id = :bout_id
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        return conn.execute(stmt, {"bout_id": bout_id}).mappings().first()


def fetch_prediction_history(
    engine: Engine, limit: int, offset: int
) -> Sequence[RowMapping]:
    """
    Newest-first page of the prediction ledger, with settlement attached
    where it exists.

    LEFT JOIN on prediction_results, not an inner join — that is the
    whole point. An unsettled prediction (fight has not happened yet) is
    exactly the row a track-record page most needs to show, because it
    proves the prediction was published BEFORE the result was known.
    An inner join would quietly hide every open prediction and make the
    ledger look retrospective.
    """
    stmt = text("""
        SELECT
            p.id                       AS prediction_id,
            p.bout_id                  AS bout_id,
            p.model_version            AS model_version,
            p.predicted_prob_red       AS predicted_prob_red,
            p.predicted_winner_id      AS predicted_winner_id,
            p.odds_at_prediction_time  AS odds_at_prediction_time,
            p.created_at               AS created_at,
            w.real_name                AS predicted_winner_name,
            fr.real_name               AS fighter_red_name,
            fb.real_name               AS fighter_blue_name,
            e.name                     AS event_name,
            e.event_date               AS event_date,
            r.id                       AS result_id,
            r.actual_winner_id         AS actual_winner_id,
            r.correct                  AS correct,
            r.settled_at               AS settled_at
        FROM predictions p
        JOIN bouts b            ON b.id = p.bout_id
        JOIN events e           ON e.id = b.event_id
        JOIN fighters fr        ON fr.id = b.fighter_red_id
        JOIN fighters fb        ON fb.id = b.fighter_blue_id
        LEFT JOIN fighters w    ON w.id = p.predicted_winner_id
        LEFT JOIN prediction_results r ON r.prediction_id = p.id
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        return conn.execute(stmt, {"limit": limit, "offset": offset}).mappings().all()