"""
GET /events/upcoming — Week 4 Monday.

The endpoint behind the frontend's main page: the next N weeks of UFC
cards with their scheduled bouts. Read-only, no model involvement — this
is pure database.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query

from api.dependencies import EngineDep
from api.schemas import (
    BoutSummary,
    FighterSummary,
    UpcomingEvent,
    UpcomingEventsResponse,
)
from api.services.queries import fetch_upcoming_bouts

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/upcoming", response_model=UpcomingEventsResponse)
def upcoming_events(
    engine: EngineDep,
    weeks: int = Query(
        default=4,
        ge=1,
        le=12,
        description="Size of the forward-looking window, in weeks.",
    ),
) -> UpcomingEventsResponse:
    """
    Return every scheduled bout on a card in the next `weeks` weeks,
    grouped by event and ordered by date.

    `weeks` is bounded (ge/le) rather than free-form so no single request
    can ask Postgres to scan an arbitrary date range. FastAPI enforces
    this before the handler body runs and returns a 422 with a clear
    message — validation is declarative, not a pile of if-statements.

    Grouping happens here in Python rather than in SQL: the result set is
    a few dozen rows at most, and a flat SELECT with a Python groupby is
    far easier to read (and to EXPLAIN) than a nested JSON aggregate.
    """
    # datetime.now(UTC).date() rather than date.today(): the server's
    # local timezone is whatever Fly.io happens to set, and "today" must
    # not shift depending on which region the container runs in.
    window_start = datetime.now(UTC).date()
    window_end = window_start + timedelta(weeks=weeks)

    rows = fetch_upcoming_bouts(engine, window_start, window_end)

    # dict preserves insertion order, and the SQL is already ordered by
    # event_date, so events come out chronologically without re-sorting.
    events: dict[int, UpcomingEvent] = {}
    for row in rows:
        event_id = row["event_id"]
        if event_id not in events:
            events[event_id] = UpcomingEvent(
                id=event_id,
                name=row["event_name"],
                event_date=row["event_date"],
                venue=row["venue"],
                location=row["location"],
                bouts=[],
            )
        events[event_id].bouts.append(
            BoutSummary(
                id=row["bout_id"],
                weight_class=row["weight_class"],
                is_title_fight=row["is_title_fight"],
                scheduled_rounds=row["scheduled_rounds"],
                card_position=row["card_position"],
                status=row["status"],
                fighter_red=FighterSummary(
                    id=row["red_id"],
                    name=row["red_name"],
                    stance=row["red_stance"],
                    height_cm=row["red_height_cm"],
                    reach_cm=row["red_reach_cm"],
                ),
                fighter_blue=FighterSummary(
                    id=row["blue_id"],
                    name=row["blue_name"],
                    stance=row["blue_stance"],
                    height_cm=row["blue_height_cm"],
                    reach_cm=row["blue_reach_cm"],
                ),
            )
        )

    return UpcomingEventsResponse(
        weeks=weeks,
        window_start=window_start,
        window_end=window_end,
        event_count=len(events),
        events=list(events.values()),
    )