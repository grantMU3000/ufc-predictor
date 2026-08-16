"""
Structural checks for features/split.py's temporal_split(): do train/
val/test sum to the whole dataset, are the date boundaries clean with
no overlap, does every bout's pair of rows land in the SAME split,
and does self_won stay 50/50 within each split. These are the same
four checks validate_split() already runs at build time — formalized
here so they run as part of the normal test suite, not just once
during the original build.

Builds a FRESH symmetrized dataset rather than reading the stored
parquet files in data/processed/ or data/test_locked/. Two reasons:
  1. data/test_locked/test.parquet is deliberately chmod 000 — a
     test that reads it would have to silently unlock it to run,
     which defeats the entire point of locking it in the first
     place (see features/split.py's save_split docstring).
  2. These tests exist to verify the SPLIT LOGIC, not "are these
     specific files on disk correct today." If the boundary dates
     ever change, a test built on stored files would keep passing
     against a stale split until someone remembers to rerun
     features/split.py — a test built on a fresh build always
     reflects what the code currently does.

Requires a local Parquet snapshot at data/processed/ (see
features/snapshot.py) — skipped automatically if it hasn't been
generated, same pattern as test_features.py and test_symmetrize.py.
"""

from pathlib import Path

import pandas as pd
import duckdb
import pytest

from features.symmetrize import build_symmetrized_dataset
from features.split import VAL_START, TEST_START, temporal_split

SNAPSHOT_DIR = Path("data/processed")
REQUIRED_TABLES = ["fighters", "events", "bouts", "bout_stats", "fighter_aliases"]


def _snapshot_available() -> bool:
    """True only if every table this file needs has a local Parquet file."""
    return all((SNAPSHOT_DIR / f"{table}.parquet").exists() for table in REQUIRED_TABLES)


@pytest.fixture(scope="module")
def con():
    """DuckDB connection with the snapshot tables exposed as VIEWs."""
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
def df(con):
    """
    Full symmetrized dataset, built once per test session and shared
    by every test below — build_symmetrized_dataset does real
    querying across ~8,456 bouts, so re-running it per test would be
    slow for no benefit (nothing in this file mutates df).
    """
    return build_symmetrized_dataset(con)


@pytest.fixture(scope="module")
def split(df):
    """(train, val, test), built once and shared — same reasoning as df."""
    return temporal_split(df)


def test_counts_sum_to_whole(df, split):
    """Nothing silently dropped or duplicated across the three splits."""
    train, val, test = split
    assert len(train) + len(val) + len(test) == len(df), (
        f"{len(train)} + {len(val)} + {len(test)} != {len(df)}"
    )


def test_boundaries_are_clean(split):
    """
    No split's dates spill into another's territory. Belt-and-
    suspenders on top of temporal_split's own filtering logic — this
    SHOULD be impossible to fail by construction, which is exactly
    why it's worth checking directly rather than just trusting the
    reasoning.
    """
    train, val, test = split
    assert pd.to_datetime(train["event_date"]).max() < VAL_START
    assert pd.to_datetime(val["event_date"]).min() >= VAL_START
    assert pd.to_datetime(val["event_date"]).max() < TEST_START
    assert pd.to_datetime(test["event_date"]).min() >= TEST_START


def test_bout_pairs_stay_together(split):
    """
    Both self_/opp_ rows of a single bout must land in the SAME
    split. This is the check that actually matters most here — if a
    bout's two rows ever ended up split across train and test, the
    model would effectively see that fight's outcome (via the
    opponent's row) during training and then get "tested" on the
    same fight, which would quietly inflate test accuracy. Since
    both rows share the same event_date, this should be impossible
    by construction — but "should" isn't "does."
    """
    train, val, test = split
    bout_to_splits = {}
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        for bout_id in part["bout_id"]:
            bout_to_splits.setdefault(bout_id, set()).add(split_name)

    split_bouts = {b: s for b, s in bout_to_splits.items() if len(s) > 1}
    assert not split_bouts, (
        f"{len(split_bouts)} bouts have rows landing in different splits: "
        f"{list(split_bouts.items())[:5]}"
    )


def test_self_won_balanced_within_each_split(split):
    """
    self_won stays 50/50 WITHIN each split, not just across the whole
    dataset. Guaranteed per-bout by symmetrize.py's construction (and
    by test_bout_pairs_stay_together above, since a balanced pair
    that can't be split apart is automatically balanced wherever it
    lands) — this is a cheap confirmation that holds, not a new risk.
    """
    train, val, test = split
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        counts = part["self_won"].value_counts()
        assert counts.get(True, 0) == counts.get(False, 0), (
            f"{split_name} not balanced: {counts.to_dict()}"
        )