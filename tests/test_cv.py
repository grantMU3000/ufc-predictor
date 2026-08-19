"""
Unit tests for models/cv.py's expanding_year_folds — Week 3 Monday
(docs/PLAN.md Section 3). Uses a small synthetic DataFrame rather than the real train.parquet —
fast, and it lets us construct exact, known edge cases (e.g. a
symmetrized bout pair, an empty year) instead of hoping real data
happens to cover them.
"""

import pandas as pd
import pytest

from models.cv import expanding_year_folds


def _make_bouts(year_bout_counts: dict[int, int]) -> pd.DataFrame:
    """
    Builds a synthetic, symmetrized-shaped DataFrame: `n` bouts in a
    given year each become 2 rows (self_/opp_ pair), same shape
    real train.parquet has. bout_id is unique per bout, shared across
    its two rows — mirrors symmetrize.py's actual output shape.

    Deliberately spreads each year's bouts across a few different
    dates (not all on Jan 1) so date-range assertions below are
    actually meaningful, not trivially true from every bout sharing
    one timestamp.
    """
    rows = []
    bout_id = 0
    for year, count in year_bout_counts.items():
        for i in range(count):
            month = (i % 12) + 1
            day = (i % 28) + 1
            event_date = pd.Timestamp(year=year, month=month, day=day)
            for source_corner in ("red", "blue"):
                rows.append(
                    {
                        "bout_id": bout_id,
                        "event_date": event_date,
                        "source_corner": source_corner,
                    }
                )
            bout_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Small but real enough to exercise a genuine expanding window:
    a thin warm-up block (2010-2011) plus three foldable years
    (2012-2014), each with a different bout count so fold sizes are
    distinguishable in assertions below.
    """
    return _make_bouts({2010: 3, 2011: 2, 2012: 4, 2013: 5, 2014: 6})


def test_yields_one_fold_per_year_after_warmup(sample_df):
    """warmup_end_year=2011 should fold 2012, 2013, 2014 — 3 folds, not 5."""
    folds = list(expanding_year_folds(sample_df, warmup_end_year=2011))
    assert len(folds) == 3


def test_val_fold_is_exactly_one_calendar_year(sample_df):
    """
    Each val_fold should contain rows from exactly one year, and that
    year should match what its position in the sequence implies
    (first fold -> 2012, since warmup absorbs everything through 2011).
    """
    folds = list(expanding_year_folds(sample_df, warmup_end_year=2011))
    expected_val_years = [2012, 2013, 2014]

    for (_, val_fold), expected_year in zip(folds, expected_val_years, strict=True):
        val_years = pd.to_datetime(val_fold["event_date"]).dt.year.unique()
        assert list(val_years) == [expected_year]


def test_val_fold_bout_counts_match_input(sample_df):
    """
    Sanity check against the exact counts _make_bouts was built with —
    same style of check used against the real data this session
    (comparing fold output to a hand-verified table).
    """
    folds = list(expanding_year_folds(sample_df, warmup_end_year=2011))
    expected_bout_counts = [4, 5, 6]  # 2012, 2013, 2014 per sample_df fixture

    for (_, val_fold), expected_n in zip(folds, expected_bout_counts, strict=True):
        assert val_fold["bout_id"].nunique() == expected_n


def test_train_window_strictly_expands(sample_df):
    """
    The core "expanding window" property: each fold's train set must
    be a strict superset of the previous fold's — bout_id-wise, not
    just row-count-wise (row count alone wouldn't catch a bug that
    dropped old bouts while adding an equal number of new ones).
    """
    folds = list(expanding_year_folds(sample_df, warmup_end_year=2011))

    prev_train_ids = None
    for train_fold, _ in folds:
        train_ids = set(train_fold["bout_id"])
        if prev_train_ids is not None:
            assert prev_train_ids.issubset(train_ids)
            assert len(train_ids) > len(prev_train_ids)
        prev_train_ids = train_ids


def test_no_bout_overlap_between_train_and_val(sample_df):
    """
    No bout_id should ever appear in both a fold's train_fold and
    that SAME fold's val_fold — the within-fold version of Saturday's
    split_integrity_check() on the outer train/val/test split.
    """
    for train_fold, val_fold in expanding_year_folds(sample_df, warmup_end_year=2011):
        train_ids = set(train_fold["bout_id"])
        val_ids = set(val_fold["bout_id"])
        assert train_ids.isdisjoint(val_ids)


def test_train_dates_never_reach_into_val_year(sample_df):
    """
    Every fold's train_fold max date must fall strictly before that
    fold's val_fold min date — the actual leakage-safety property
    this whole file exists to guarantee. A bug that let one future
    bout slip into a training window would trip this.
    """
    for train_fold, val_fold in expanding_year_folds(sample_df, warmup_end_year=2011):
        assert train_fold["event_date"].max() < val_fold["event_date"].min()


def test_symmetrized_pair_stays_together_within_a_fold(sample_df):
    """
    Both rows of a symmetrized bout_id pair share one event_date, so
    they should always land in the same fold's same split (both in
    that fold's train_fold, or both in that fold's val_fold) — never
    split across the two. Checks every bout_id has exactly 2 rows
    wherever it lands, for every fold.
    """
    for train_fold, val_fold in expanding_year_folds(sample_df, warmup_end_year=2011):
        for fold_df in (train_fold, val_fold):
            counts = fold_df["bout_id"].value_counts()
            assert (counts == 2).all(), (
                f"found bout_id(s) with != 2 rows in one split: "
                f"{counts[counts != 2].to_dict()}"
            )


def test_empty_dataframe_raises():
    """An empty df has nothing to fold over — should fail loudly, not
    silently yield zero folds."""
    empty_df = pd.DataFrame(columns=["bout_id", "event_date", "source_corner"])
    with pytest.raises(ValueError, match="empty"):
        list(expanding_year_folds(empty_df))


def test_no_data_after_warmup_raises(sample_df):
    """
    warmup_end_year set past every year in the data means there's
    nothing left to fold over — should raise, not silently yield zero
    folds and let an Optuna loop quietly do nothing (the exact failure
    mode this check exists to prevent, per models/cv.py's docstring).
    """
    with pytest.raises(ValueError, match="no data found after"):
        list(expanding_year_folds(sample_df, warmup_end_year=2020))


def test_real_warmup_boundary_matches_this_session(sample_df):
    """
    Not a test of expanding_year_folds's logic — a guard against
    models/cv.py's WARMUP_END_YEAR default silently drifting from the
    2010 value this session hand-verified against real bout counts.
    If this ever fails, it means someone changed the default without
    re-checking it against real year-by-year data the way this
    session did.
    """
    from models.cv import WARMUP_END_YEAR

    assert WARMUP_END_YEAR == 2010