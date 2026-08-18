"""
tests/test_symmetrize.py

Structural checks for features/symmetrize.py's build_symmetrized_dataset:
does it produce exactly two rows per eligible bout, is self_won balanced
50/50, and has every red_/blue_ column been reshaped into self_/opp_.
These are shape/invariant checks, not hand-verified values (see
test_features.py for that pattern) — a violation of any of them means
the self_/opp_ reshape itself is broken, independent of what any single
feature computes.

Requires a local Parquet snapshot at data/processed/ (see
features/snapshot.py) — skipped automatically if it hasn't been
generated, the same pattern already used by test_features.py.
"""

from pathlib import Path

import duckdb
import pytest

from features.labels import get_completed_decided_bouts
from features.symmetrize import build_symmetrized_dataset

SNAPSHOT_DIR = Path("data/processed")
REQUIRED_TABLES = ["fighters", "events", "bouts", "bout_stats", "fighter_aliases"]


def _snapshot_available() -> bool:
    """True only if every table this file needs has a local Parquet file."""
    return all(
        (SNAPSHOT_DIR / f"{table}.parquet").exists() for table in REQUIRED_TABLES
    )


@pytest.fixture(scope="module")
def con():
    """
    A DuckDB connection with the snapshot tables exposed as VIEWs —
    module-scoped since the snapshot doesn't change mid-file.
    """
    if not _snapshot_available():
        pytest.skip(
            "Local Parquet snapshot not found at data/processed/ — "
            "run features/snapshot.py first."
        )

    connection = duckdb.connect()
    for table in REQUIRED_TABLES:
        parquet_path = SNAPSHOT_DIR / f"{table}.parquet"
        connection.execute(f"CREATE VIEW {table} AS SELECT * FROM '{parquet_path}'")
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def labels(con):
    return get_completed_decided_bouts(con)


@pytest.fixture(scope="module")
def df(con):
    """
    Built once per test session, module-scoped — build_symmetrized_dataset
    calls build_feature_row twice per bout (real querying), so every test
    in this file shares one build instead of redoing it per assertion.
    """
    return build_symmetrized_dataset(con)


def test_row_count_is_double_labels(df, labels):
    assert len(df) == 2 * len(labels), (
        f"expected {2 * len(labels)} rows (2 per bout), got {len(df)}"
    )


def test_self_won_is_balanced(df, labels):
    won_counts = df["self_won"].value_counts()
    assert won_counts[True] == won_counts[False] == len(labels), (
        f"self_won should split 50/50 across {len(labels)} bouts, "
        f"got {won_counts.to_dict()}"
    )


def test_no_red_blue_columns_leak_through(df):
    leaked_columns = [c for c in df.columns if c.startswith(("red_", "blue_"))]
    assert not leaked_columns, f"red_/blue_ columns leaked through: {leaked_columns}"
