"""
Small synthetic frame covering all three column categories
to_differential handles, so the pairing/dropping logic is verified
without needing the real Parquet snapshot at all.
"""

import pandas as pd
import pytest

from features.differential import to_differential


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            # label
            "self_won": [1, 0],
            # identity — must be excluded even though it's a real self_/opp_ pair
            "self_fighter_id": [101, 205],
            "opp_fighter_id": [102, 206],
            # numeric pairs — should become diff_ columns
            "self_slpm": [4.5, 3.0],
            "opp_slpm": [3.0, 4.5],
            "self_reach": [72.0, 68.0],
            "opp_reach": [70.0, 74.0],
            # non-numeric pair — should be dropped, not crash
            "self_stance": ["orthodox", "southpaw"],
            "opp_stance": ["southpaw", "orthodox"],
            # self_-only, no opp_ pair — should be dropped
            "self_is_open_stance_matchup": [True, True],
            # bout-level, no self_/opp_ prefix at all — should never appear in X
            "bout_id": [1, 2],
            "event_date": ["2023-01-01", "2023-02-01"],
            "weight_class": ["Lightweight", "Welterweight"],
            "is_title_fight": [False, False],
            "source_corner": ["red", "blue"],
        }
    )


def test_correct_columns_included(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    assert set(X.columns) == {"diff_slpm", "diff_reach"}


def test_diff_values_hand_computed(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    assert X["diff_slpm"].tolist() == pytest.approx([1.5, -1.5])
    assert X["diff_reach"].tolist() == pytest.approx([2.0, -6.0])


def test_identity_columns_excluded(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    assert "diff_fighter_id" not in X.columns


def test_non_numeric_pair_dropped(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    assert "diff_stance" not in X.columns


def test_self_only_column_dropped(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    assert "self_is_open_stance_matchup" not in X.columns
    assert "diff_is_open_stance_matchup" not in X.columns


def test_bout_level_columns_never_enter_X(sample_df):
    X, _ = to_differential(sample_df, verbose=False)
    for col in [
        "bout_id",
        "event_date",
        "weight_class",
        "is_title_fight",
        "source_corner",
    ]:
        assert col not in X.columns


def test_label_extracted_correctly(sample_df):
    _, y = to_differential(sample_df, verbose=False)
    assert y.tolist() == [1, 0]


def test_index_preserved(sample_df):
    X, y = to_differential(sample_df, verbose=False)
    assert list(X.index) == list(sample_df.index)
    assert list(y.index) == list(sample_df.index)
