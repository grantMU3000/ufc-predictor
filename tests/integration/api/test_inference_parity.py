"""
Training/serving parity — Week 4 Tuesday, Step 5.

THE POINT: everything up to now proves the inference path RUNS and
produces plausible-looking numbers. Plausible is not the bar. This
file proves the inference path produces the SAME numbers the model
was trained on, to floating point.

How: take bouts that already have rows in the training matrix, run
them back through the LIVE inference path with as_of_date set to
their own event date, and compare column by column. A match means the
model being served is the model that was evaluated. A mismatch means
every downstream number — the ledger, settlement, the public track
record — describes a model nobody ever measured.

This is the reason training/serving skew is caught here and not in
Week 6 with a ledger full of predictions built on bad inputs. The
failure has no symptom: a shifted feature produces a confident,
well-formed, wrong probability, and nothing anywhere raises.

Uses train/val bouts only. data/test_locked/test.parquet stays at
chmod 000 — it is not read here and must not be.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from api.services.features import (
    build_bout_features,
    compute_elo_as_of,
    snapshot_connection,
)
from features.build_lgbm_matrix import build_train_val_with_elo
from features.differential import to_differential

# Floating-point tolerance. Both paths run the identical functions on
# the identical data, so the only legitimate difference is float
# accumulation order. Anything larger is a real recipe divergence, not
# noise — this is deliberately far tighter than a "close enough" check.
TOLERANCE = 1e-9

# How many bouts to check. Three is enough to catch a systematic
# recipe error (which would break all of them) while keeping runtime
# sane — each one replays the full ~8,600-bout Elo history.
N_BOUTS = 3

METADATA_PATH = Path("models/v1/metadata.json")


@pytest.fixture(scope="module")
def feature_order() -> tuple[str, ...]:
    """
    The model's positional contract, read from the frozen artifact.

    Deliberately NOT read from the database registry here. This test
    asks "does inference reproduce training," a question about the
    feature recipe — pulling the order from the same frozen file the
    training matrix was validated against keeps a registry drift
    problem from masquerading as a parity failure. Registry/metadata
    agreement is checked separately.
    """
    if not METADATA_PATH.exists():
        pytest.skip(f"{METADATA_PATH} not found")
    with open(METADATA_PATH) as f:
        return tuple(json.load(f)["feature_list"])


@pytest.fixture(scope="module")
def training_matrix() -> pd.DataFrame:
    """
    The ground truth: train + val with Elo attached, exactly as
    build_lgbm_matrix assembles it for the shipped model.

    All three Tier 3 flags left at their defaults (False) — ADR-016
    cut those features, and the __main__ block in build_lgbm_matrix
    that forces them ON is a build smoke test, not the shipped path.

    Module-scoped: this loads two parquet files and replays the full
    Elo history. Once per test session, not once per bout.
    """
    for path in ("data/processed/train.parquet", "data/processed/val.parquet"):
        if not Path(path).exists():
            pytest.skip(f"{path} not found — run the Week 2 build first")

    train, val = build_train_val_with_elo()
    return pd.concat([train, val], ignore_index=True)


def _training_row_for_bout(
    training_matrix: pd.DataFrame, bout_id: int, feature_order: tuple[str, ...]
) -> pd.DataFrame:
    """
    Pull one bout's two symmetrized rows out of the training matrix
    and difference them into the model's input shape.

    RED FIRST. build_bout_features always emits red-perspective
    first, and to_differential emits columns in DataFrame order, so
    comparing a red-first frame against a blue-first one would
    compare row 0 against row 1 and report a spurious sign flip on
    every column. Sorting here rather than trusting parquet row order
    makes that impossible.
    """
    rows = training_matrix[training_matrix["bout_id"] == bout_id].copy()
    assert len(rows) == 2, f"bout {bout_id} has {len(rows)} rows, expected 2"

    rows["_red_first"] = (rows["source_corner"] != "red").astype(int)
    rows = rows.sort_values("_red_first").drop(columns=["_red_first"])

    features, _ = to_differential(rows, verbose=False)
    return features[list(feature_order)].reset_index(drop=True)


@pytest.fixture(scope="module")
def sample_bout_ids(training_matrix: pd.DataFrame) -> list[int]:
    """
    Bouts to check, spread across the training window rather than
    clustered.

    Spread matters: an early bout exercises thin fighter history and
    near-initial Elo, a late one exercises deep history and a fully
    walked rating. A recipe bug that only bites one of those regimes
    (an off-by-one in a rolling window, a debut fallback) would hide
    behind three bouts picked from the same month.
    """
    bouts = (
        training_matrix[["bout_id", "event_date"]]
        .drop_duplicates()
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    # Skip the first 5% — the very earliest bouts have almost no
    # history on either side, so nearly every feature is NaN and a
    # match there proves little.
    start = int(len(bouts) * 0.05)
    usable = bouts.iloc[start:]
    step = len(usable) // N_BOUTS
    return [int(usable.iloc[i * step]["bout_id"]) for i in range(N_BOUTS)]


def test_inference_matches_training_matrix(
    training_matrix: pd.DataFrame,
    sample_bout_ids: list[int],
    feature_order: tuple[str, ...],
) -> None:
    """
    THE test. For each sampled bout, rebuild its features through the
    live inference path and compare against the training matrix row.

    Elo is recomputed per bout (not shared across bouts) because each
    bout needs the ratings as they stood on ITS date, not today's.
    compute_elo_as_of filters to strictly-earlier bouts, which is
    exactly what compute_elo_ratings recorded mid-walk as that bout's
    pre-fight rating — same number, arrived at two ways.

    check_exact=False with an explicit atol, and check_dtype=False:
    the training path round-trips through parquet (which can widen an
    int column to float), the inference path builds in memory. A
    dtype difference on identical values is not a parity failure.
    NaN positions ARE compared strictly — assert_frame_equal treats
    NaN as equal to NaN, so a feature that's missing in one path and
    present in the other still fails, correctly.
    """
    with snapshot_connection() as con:
        for bout_id in sample_bout_ids:
            expected = _training_row_for_bout(
                training_matrix, bout_id, feature_order
            )

            event_date = training_matrix.loc[
                training_matrix["bout_id"] == bout_id, "event_date"
            ].iloc[0]
            elo = compute_elo_as_of(con, event_date)

            actual = build_bout_features(
                con, bout_id, feature_order, elo_ratings=elo
            ).matrix.reset_index(drop=True)

            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_exact=False,
                atol=TOLERANCE,
                rtol=0,
                check_dtype=False,
                obj=f"bout {bout_id}",
            )


def test_column_order_matches_feature_order(
    training_matrix: pd.DataFrame,
    sample_bout_ids: list[int],
    feature_order: tuple[str, ...],
) -> None:
    """
    Separate from the value check on purpose.

    assert_frame_equal already compares column names, so this looks
    redundant — it isn't. If order ever diverged, the value test
    would fail with a wall of numeric mismatches that reads like a
    feature bug. This fails first with a one-line "your columns moved"
    message, which is the actual diagnosis.
    """
    with snapshot_connection() as con:
        result = build_bout_features(con, sample_bout_ids[0], feature_order)
    assert list(result.matrix.columns) == list(feature_order)


def test_rows_are_exact_negatives(
    sample_bout_ids: list[int], feature_order: tuple[str, ...]
) -> None:
    """
    Row 1 must be the exact negation of row 0.

    This is an algebra fact, not a model property: diff = self - opp,
    and the two rows swap self and opp, so row 1 = -row 0 by
    construction. It holds regardless of what the model does.

    Worth asserting anyway — it's the cheapest possible detector for
    _symmetrize_row picking the wrong corner, and unlike the symmetry
    gap in inference.py (which is a real, nonzero model property),
    any failure here is unambiguously a bug.
    """
    with snapshot_connection() as con:
        result = build_bout_features(con, sample_bout_ids[0], feature_order)

    row_red = result.matrix.iloc[0]
    row_blue = result.matrix.iloc[1]
    pd.testing.assert_series_equal(
        row_blue, -row_red, check_names=False, atol=TOLERANCE, rtol=0
    )