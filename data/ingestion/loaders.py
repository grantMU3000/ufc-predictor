"""
Loader module inserts all fight data into Postgres.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import MetaData, Table, create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

load_dotenv()

def _get_engine():
    return create_engine(os.environ["DATABASE_URL"])

def _clean_for_insert(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """
    Convert a DataFrame slice to insert-ready records, turning every NaN
    (pandas float NaN, nullable Int64's <NA>) into a real Python None.
    Without this, NaNs can slip through as-is and confuse the Postgres
    driver on integer/date columns expecting either a value or NULL.
    """
    clean = df[cols].astype(object).where(pd.notnull(df[cols]), None)
    return clean.to_dict(orient="records")

def _upsert_by_source_url(
        engine, table: Table, df: pd.DataFrame, insert_cols: list[str], 
        preserve_on_conflict: list[str] | None = None,
    ) -> dict[str, int]:
    """
    Upsert rows keyed on source_url. Returns {source_url: db_id} for
    every row — whether newly inserted or already present — which is
    exactly what's needed to translate other tables' FK references.

    preserve_on_conflict: columns to set on INSERT but never overwrite
    on conflict — for values populated by a separate process after the
    initial load (e.g. ufc_debut_date, backfilled from bout data).
    """
    preserve_on_conflict = preserve_on_conflict or []
    records = _clean_for_insert(df, insert_cols)
    if not records:
        return {}

    stmt = pg_insert(table).values(records)
    update_cols = {
        c: stmt.excluded[c] for c in insert_cols 
        if c != "source_url" and c not in preserve_on_conflict
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_url"], set_=update_cols
    ).returning(table.c.id, table.c.source_url)

    with engine.begin() as conn:
        result = conn.execute(stmt)
        return {row.source_url: row.id for row in result}

def _remap_ids(series: pd.Series, pandas_id_to_url: dict, url_to_db_id: dict) -> pd.Series:
    """Translate a column of pandas-local ids to real database ids, via
    each id's source_url. NaN/unmapped ids stay NaN."""
    return series.map(pandas_id_to_url).map(url_to_db_id)

from sqlalchemy import text

def backfill_ufc_debut_dates(engine) -> int:
    """
    Sets fighters.ufc_debut_date to the earliest event_date among all
    bouts that fighter appears in (either corner). Recomputes fresh every
    run — safe to call repeatedly, and correctly updates fighters whose
    known bout history has grown since the last run.
    """
    stmt = text("""
        UPDATE fighters f
        SET ufc_debut_date = sub.min_date
        FROM (
            SELECT fighter_id, MIN(event_date) AS min_date
            FROM (
                SELECT b.fighter_red_id AS fighter_id, e.event_date
                FROM bouts b JOIN events e ON b.event_id = e.id
                UNION ALL
                SELECT b.fighter_blue_id AS fighter_id, e.event_date
                FROM bouts b JOIN events e ON b.event_id = e.id
            ) all_appearances
            GROUP BY fighter_id
        ) sub
        WHERE f.id = sub.fighter_id
    """)
    with engine.begin() as conn:
        result = conn.execute(stmt)
        return result.rowcount

def load_fighters(engine, fighters_df: pd.DataFrame) -> dict[str, int]:
    metadata = MetaData()
    fighters_tbl = Table("fighters", metadata, autoload_with=engine)

    df = fighters_df.copy()
    df["nationality"] = None    
    df["ufc_debut_date"] = None     # correct default for a brand-new fighter

    insert_cols = [
        "real_name", "dob", "height_cm", "reach_cm", "stance",
        "nationality", "ufc_debut_date", "source_url",
    ]
    return _upsert_by_source_url(
        engine, fighters_tbl, df, insert_cols,
        preserve_on_conflict=["nationality", "ufc_debut_date"],
    )

def load_events(engine, events_df: pd.DataFrame) -> dict[str, int]:
    metadata = MetaData()
    events_tbl = Table("events", metadata, autoload_with=engine)

    df = events_df.copy()
    df["venue"] = None  # not provided by Greco's ufc_events.csv

    insert_cols = ["name", "event_date", "location", "venue", "source_url"]
    return _upsert_by_source_url(engine, events_tbl, df, insert_cols)

def load_bouts(
    engine, bouts_df: pd.DataFrame,
    fighter_pandas_id_to_url: dict, fighter_url_to_db_id: dict,
    event_pandas_id_to_url: dict, event_url_to_db_id: dict,
) -> dict[str, int]:
    """
    Remaps fighter_red_id/fighter_blue_id/winner_id/event_id through
    source_url before upserting
    """
    df = bouts_df.copy()
    df["fighter_red_id"] = _remap_ids(df["fighter_red_id"], fighter_pandas_id_to_url, fighter_url_to_db_id)
    df["fighter_blue_id"] = _remap_ids(df["fighter_blue_id"], fighter_pandas_id_to_url, fighter_url_to_db_id)
    df["winner_id"] = _remap_ids(df["winner_id"], fighter_pandas_id_to_url, fighter_url_to_db_id)
    df["event_id"] = _remap_ids(df["event_id"], event_pandas_id_to_url, event_url_to_db_id)

    # card_position and weigh-in columns aren't provided by Greco at all -
    # left unset here, NULL in the DB (both nullable)

    metadata = MetaData()
    bouts_tbl = Table("bouts", metadata, autoload_with=engine)

    insert_cols = [
        "event_id", "fighter_red_id", "fighter_blue_id", "weight_class",
        "is_title_fight", "scheduled_rounds", "status", "winner_id",
        "method", "method_detail", "ending_round", "ending_time_seconds",
        "source_url",
    ]
    return _upsert_by_source_url(engine, bouts_tbl, df, insert_cols)

def load_bout_stats(
    engine, bout_stats_df: pd.DataFrame,
    bout_pandas_id_to_url: dict, bout_url_to_db_id: dict,
    fighter_pandas_id_to_url: dict, fighter_url_to_db_id: dict,
) -> int:
    df = bout_stats_df.copy()
    df["bout_id"] = _remap_ids(df["bout_id"], bout_pandas_id_to_url, bout_url_to_db_id)
    df["fighter_id"] = _remap_ids(df["fighter_id"], fighter_pandas_id_to_url, fighter_url_to_db_id)

    before = len(df)
    df = df.dropna(subset = ["bout_id", "fighter_id"])
    if len(df) < before:
        print(f"load_bout_stats: dropped {before - len(df)} rows — "
              f"unmappable bout_id/fighter_id (should not happen if "
              f"fighters/bouts loaded cleanly first).")

    metadata = MetaData()
    stats_tbl = Table("bout_stats", metadata, autoload_with=engine)

    insert_cols = [c for c in df.columns if c in stats_tbl.c.keys()]
    records = _clean_for_insert(df, insert_cols)
    if not records:
        return

    stmt = pg_insert(stats_tbl).values(records)
    update_cols = {
        c: stmt.excluded[c] for c in insert_cols
        if c not in ("bout_id", "fighter_id", "round_number")
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["bout_id", "fighter_id", "round_number"], set_=update_cols
    ).returning(stats_tbl.c.id)

    with engine.begin() as conn:
        result = conn.execute(stmt)
        return len(result.fetchall())

def load_all(fighters_df, events_df, bouts_df, bout_stats_df) -> None:
    engine = _get_engine()

    fighter_pandas_id_to_url = dict(zip(fighters_df["fighter_id"], fighters_df["source_url"]))
    event_pandas_id_to_url = dict(zip(events_df["event_id"], events_df["source_url"]))
    bout_pandas_id_to_url = dict(zip(bouts_df["bout_id"], bouts_df["source_url"]))

    print("Loading fighters...")
    fighter_url_to_db_id = load_fighters(engine, fighters_df)
    print(f"  {len(fighter_url_to_db_id)} fighters upserted")

    print("Loading events...")
    event_url_to_db_id = load_events(engine, events_df)
    print(f"  {len(event_url_to_db_id)} events upserted")

    print("Loading bouts...")
    bout_url_to_db_id = load_bouts(
        engine, bouts_df,
        fighter_pandas_id_to_url, fighter_url_to_db_id,
        event_pandas_id_to_url, event_url_to_db_id,
    )
    print(f"  {len(bout_url_to_db_id)} bouts upserted")

    print("Backfilling ufc_debut_date...")
    debut_count = backfill_ufc_debut_dates(engine)
    print(f"  {debut_count} fighters updated")

    print("Loading bout_stats...")
    stats_count = load_bout_stats(
        engine, bout_stats_df,
        bout_pandas_id_to_url, bout_url_to_db_id,
        fighter_pandas_id_to_url, fighter_url_to_db_id,
    )
    print(f"  {stats_count} bout_stats upserted")