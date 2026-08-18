"""
Loads parsed Wikipedia event/bout data into events/bouts.

Events use a bulk upsert (same shape as loaders.py's _upsert_by_source_url,
generalized to key on wikipedia_pageid). Bouts stay row-level — the
cancel-and-reinsert swap logic is conditional, not a blind upsert.
"""

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from data.ingestion.loaders import _clean_for_insert  # reused, not duplicated


def _upsert_by_key(
    engine, table: Table, df: pd.DataFrame, insert_cols: list[str], key_col: str
) -> dict:
    """Generalized version of loaders.py's _upsert_by_source_url, keyed on any unique column."""
    records = _clean_for_insert(df, insert_cols)
    if not records:
        return {}

    stmt = pg_insert(table).values(records)
    update_cols = {c: stmt.excluded[c] for c in insert_cols if c != key_col}
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[key_col], set_=update_cols
    ).returning(table.c.id, getattr(table.c, key_col))

    with engine.begin() as conn:
        result = conn.execute(upsert_stmt)
        return {getattr(row, key_col): row.id for row in result}


def load_upcoming_events(engine, events_df: pd.DataFrame) -> dict[int, int]:
    """
    Bulk upsert keyed on wikipedia_pageid. events_df columns:
    name, event_date, venue, location, wikipedia_pageid.
    Returns {wikipedia_pageid: events.id}.
    """
    metadata = MetaData()
    events_tbl = Table("events", metadata, autoload_with=engine)

    insert_cols = ["name", "event_date", "venue", "location", "wikipedia_pageid"]
    return _upsert_by_key(
        engine, events_tbl, events_df, insert_cols, key_col="wikipedia_pageid"
    )


TITLE_KEYWORDS = ("championship", "title")


def infer_bout_details(bout: dict, is_main_event: bool = False) -> dict:
    """
    Wikipedia's {{MMAevent bout}} template doesn't give us is_title_fight or
    scheduled_rounds directly. Title fights are 5 rounds; so are main
    events, belt or not — hence is_main_event, derived from the bout's
    position in document order by the caller.
    """
    is_title_fight = (
        bout["fighter_red_is_champion"]
        or bout["fighter_blue_is_champion"]
        or any(kw in bout["notes"].lower() for kw in TITLE_KEYWORDS)
    )
    return {
        **bout,
        "is_title_fight": is_title_fight,
        "scheduled_rounds": 5 if (is_title_fight or is_main_event) else 3,
    }


def load_bout(
    engine,
    event_id: int,
    bout: dict,
    fighter_red_id: int,
    fighter_blue_id: int,
    is_main_event: bool = False,
) -> None:
    """
    One transaction per bout — the check, any cancellation, and the insert
    happen atomically, so a fighter swap can't be left half-applied.
    """
    inferred = infer_bout_details(bout, is_main_event=is_main_event)

    with engine.begin() as conn:
        # Exact pairing already exists and is still active — nothing to do
        # beyond refreshing details that could've changed (weight class, card position).
        existing_exact = conn.execute(
            text("""
                SELECT id FROM bouts
                WHERE event_id = :event_id AND status = 'scheduled'
                  AND fighter_red_id = :red AND fighter_blue_id = :blue
            """),
            {"event_id": event_id, "red": fighter_red_id, "blue": fighter_blue_id},
        ).fetchone()

        if existing_exact:
            conn.execute(
                text("""
                    UPDATE bouts
                    SET weight_class = :wc, card_position = :cp,
                        scheduled_rounds = CASE WHEN rounds_confirmed THEN scheduled_rounds ELSE :rounds END,
                        is_title_fight = CASE WHEN rounds_confirmed THEN is_title_fight ELSE :title END
                    WHERE id = :id
                """),
                {
                    "wc": inferred["weight_class"],
                    "cp": inferred["card_tier"],
                    "rounds": inferred["scheduled_rounds"],
                    "title": inferred["is_title_fight"],
                    "id": existing_exact.id,
                },
            )
            return

        # Check for a fighter swap: an active bout on this event involving
        # either fighter, but not this exact pairing.
        stale = conn.execute(
            text("""
                SELECT id FROM bouts
                WHERE event_id = :event_id AND status = 'scheduled'
                  AND (fighter_red_id IN (:red, :blue) OR fighter_blue_id IN (:red, :blue))
            """),
            {"event_id": event_id, "red": fighter_red_id, "blue": fighter_blue_id},
        ).fetchall()

        for row in stale:
            conn.execute(
                text("UPDATE bouts SET status = 'cancelled' WHERE id = :id"),
                {"id": row.id},
            )

        conn.execute(
            text("""
                INSERT INTO bouts (
                    event_id, fighter_red_id, fighter_blue_id, weight_class,
                    is_title_fight, scheduled_rounds, card_position, status
                ) VALUES (
                    :event_id, :red, :blue, :wc, :title, :rounds, :cp, 'scheduled'
                )
            """),
            {
                "event_id": event_id,
                "red": fighter_red_id,
                "blue": fighter_blue_id,
                "wc": inferred["weight_class"],
                "title": inferred["is_title_fight"],
                "rounds": inferred["scheduled_rounds"],
                "cp": inferred["card_tier"],
            },
        )
