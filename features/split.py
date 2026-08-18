"""
Splits the symmetrized training matrix into train/val/test by
event_date — the fix for leakage bug #3 in docs/PLAN.md Section 0.2:
a random split lets the model see a fighter's 2025 form while
predicting their 2021 fight. A strict TIME split means train only
ever sees dates older than val, and val only ever sees dates older
than test — no peeking into the future, ever.

Runs AFTER symmetrization (features/symmetrize.py), never before —
splitting needs both self_/opp_ rows of a bout to already exist side
by side, so a bout's two rows can be checked into the same split
(see validate_split's pairing check).

Reminder for future you: a fighter appearing in multiple splits is
NOT leakage. Only a FIGHT seeing its own future data is. Fighter 394
having bouts in train, val, AND test is expected and fine — each
row's features are frozen as of that specific bout's own date.
"""

import os
from pathlib import Path

import pandas as pd

# Half-open boundaries. Only two constants exist — train's upper
# edge and val's lower edge are the SAME number (VAL_START), so
# there's no way for them to independently drift into an off-by-one
# against each other. Same idea for val/test at TEST_START.
VAL_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2025-01-01")


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a symmetrized dataframe into (train, val, test) by
    event_date:
        train: event_date <  2023-01-01   (everything through 2022)
        val:   2023-01-01 <= event_date < 2025-01-01
        test:  event_date >= 2025-01-01

    pd.to_datetime() on the comparison column, not the raw column
    itself, is deliberate — event_date started life as a plain
    datetime.date (from DuckDB's fetchone() in store.py), and
    comparing a bare date directly against a pd.Timestamp is exactly
    the type mismatch that broke days_since_last_fight in tier2.py
    (TypeError: date vs Timestamp). Forcing to_datetime() here means
    this comparison is safe regardless of which flavor event_date
    happens to be by the time it reaches this function.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_symmetrized_dataset — must have event_date
        and bout_id columns.

    Returns
    -------
    (train, val, test) — three DataFrames, each a real copy (not a
    view into df), so downstream code can freely add/modify columns
    on one split without silently touching the others.
    """
    event_date = pd.to_datetime(df["event_date"])

    train = df[event_date < VAL_START].copy()
    val = df[(event_date >= VAL_START) & (event_date < TEST_START)].copy()
    test = df[event_date >= TEST_START].copy()

    return train, val, test


def validate_split(
    df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """
    The four checks from the Week 2 Wednesday roadmap. Raises
    AssertionError on the first violation — run this immediately
    after temporal_split(), before anything gets written to disk.
    """
    # 1. Counts sum to the whole — nothing silently dropped or duplicated.
    assert len(train) + len(val) + len(test) == len(df), (
        f"{len(train)} + {len(val)} + {len(test)} != {len(df)}"
    )

    # 2. Clean boundaries, no overlap — belt-and-suspenders on top of
    # the fact that temporal_split's own filtering logic guarantees
    # this by construction. Worth checking anyway: this is exactly
    # the kind of thing you check because it SHOULD be impossible to
    # fail, and "should be impossible" is a famous last sentence.
    assert pd.to_datetime(train["event_date"]).max() < VAL_START
    assert pd.to_datetime(val["event_date"]).min() >= VAL_START
    assert pd.to_datetime(val["event_date"]).max() < TEST_START
    assert pd.to_datetime(test["event_date"]).min() >= TEST_START

    # 3. Pairing integrity — the check that actually matters most.
    # Both self_/opp_ rows of a bout share the same event_date, so
    # they should mechanically always land in the same split — but
    # "should" isn't "does," so verify it directly rather than
    # trusting the reasoning.
    bout_to_splits: dict[str, set[str]] = {}
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        for bout_id in part["bout_id"]:
            bout_to_splits.setdefault(bout_id, set()).add(split_name)
    split_bouts = {b: s for b, s in bout_to_splits.items() if len(s) > 1}
    assert not split_bouts, (
        f"{len(split_bouts)} bouts have rows landing in different splits: "
        f"{list(split_bouts.items())[:5]}"
    )

    # 4. self_won stays ~50/50 within each split. Guaranteed per-bout
    # by construction in symmetrize.py, so this should hold
    # automatically — cheap to confirm it does.
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        counts = part["self_won"].value_counts()
        assert counts.get(True, 0) == counts.get(False, 0), (
            f"{split_name} not balanced: {counts.to_dict()}"
        )


def save_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: Path = Path("data/processed"),
    locked_dir: Path = Path("data/test_locked"),
) -> None:
    """
    Writes train/val to the normal working directory, and test to a
    SEPARATE, chmod-locked directory — the "sealed exam envelope"
    for Week 3 Friday's one-and-only test-set unlock.

    os.chmod(path, 0o000) turns "I promise not to open this file"
    into "I literally cannot open this file without a deliberate
    permissions change first." A real engineering control, not just
    a naming convention — the kind of detail that's easy to explain
    and hard to argue with in an interview.

    Note for future reruns: if you ever need to rebuild the test
    set, you'll need to chmod it back to something writable (e.g.
    0o644) BEFORE this function tries to overwrite it, or the write
    itself will fail. That friction is intentional — rebuilding the
    test set should never be a casual, accidental action.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    locked_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(processed_dir / "train.parquet")
    val.to_parquet(processed_dir / "val.parquet")

    test_path = locked_dir / "test.parquet"
    test.to_parquet(test_path)
    os.chmod(test_path, 0o000)
    print(f"test set locked at {test_path} (chmod 000)")


if __name__ == "__main__":
    import duckdb

    from features.symmetrize import build_symmetrized_dataset

    con = duckdb.connect()
    for table in [
        "fighters",
        "events",
        "bouts",
        "bout_stats",
        "fighter_aliases",
        "odds_snapshots",
    ]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    df = build_symmetrized_dataset(con)
    train, val, test = temporal_split(df)
    validate_split(df, train, val, test)
    save_split(train, val, test)

    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")
