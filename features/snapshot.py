"""
Pulls the current, cleaned data out of Postgres (the source of truth)
and saves it as local Parquet files, so features can be built and
tested fast without hitting the real database every time.

Simple version: Postgres is a library across town. Parquet files are
photocopies on your desk. You still trust the library's copy is the
real one, but for a day of writing and re-running 50 feature
experiments, working off the desk copy is way faster.

This is a POINT-IN-TIME COPY, not a live connection. Re-run this any
time the underlying Postgres data changes (new results ingested, a
quality-check fix applied like the rounds_confirmed update) so the
local copy doesn't go stale and you don't debug against old data.
"""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("data/processed")
TABLES = ["fighters", "events", "bouts", "bout_stats", "fighter_aliases", "odds_snapshots"]


def export_postgres_to_parquet(database_url: str, output_dir: Path = OUTPUT_DIR) -> None:
    """
    Copy each table in TABLES from Postgres into its own Parquet file.

    Parameters
    ----------
    database_url : str
        SQLAlchemy-style connection string for Neon Postgres.
    output_dir : Path
        Where to write the .parquet files. Created if missing.

    Why DuckDB does the copying instead of pandas: DuckDB can talk
    directly to Postgres and stream rows straight into a Parquet file.
    That's faster and uses far less memory than pandas' "load the
    whole table into RAM, then write it back out" approach — which
    will matter once bout_stats (per-round data) grows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{database_url}' AS pg (TYPE postgres, READ_ONLY);")

    for table in TABLES:
        dest = output_dir / f"{table}.parquet"
        con.execute(f"COPY (SELECT * FROM pg.{table}) TO '{dest}' (FORMAT PARQUET);")
        print(f"  wrote {dest}")

    con.close()


if __name__ == "__main__":
    export_postgres_to_parquet(os.environ["DATABASE_URL"])