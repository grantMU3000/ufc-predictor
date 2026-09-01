"""
Proves a ledger row is self-sufficient — Week 4 Wednesday, ADR-025
Decision 1.

THE CLAIM UNDER TEST: given only a predictions row (specifically its
feature_snapshot JSONB), you can reconstruct the exact probability the
model reported, using nothing else. No DuckDB, no Parquet, no live
feature pipeline. If the snapshot is regenerated or the pipeline
changes shape next month, an existing ledger row must still explain
itself — that self-sufficiency is the entire audit-trail argument.

WHY THIS DOESN'T CALL predict_bout(). predict_bout() takes a
BoutFeatureMatrix, which normally only exists via build_bout_features()
reading a live snapshot. Reusing it here would prove the ledger agrees
with the pipelin vector BY NAME from contributions[] and hands it
straight to the frozen  that wrote it, not that the row stands alone. This
rebuilds the input veBooster — a second, independent path to the
same number.
"""

import numpy as np
import pandas as pd
import pytest

from api.services.queries import fetch_latest_prediction_for_bout

# Same order of magnitude as predict_bout()'s own TreeSHAP
# reconstruction tolerance. Both paths run the identical booster on
# identical float64 inputs, so the only legitimate difference is
# accumulation order inside LightGBM, not anything the JSON round-trip
# introduces (json.dumps of a Python float round-trips exactly).
_REPLAY_TOLERANCE = 1e-6


def _rebuild_feature_frame(
    feature_snapshot: dict, feature_order: tuple[str, ...]
) -> pd.DataFrame:
    """
    Reconstruct the model input, in the correct column order, from the
    ledger's stored contributions alone.

    contributions[] carries (feature, feature_value) pairs but in
    IMPACT order — rank_contributions() sorts by |log_odds| for
    display. This re-sorts back into feature_order, the only order the
    Booster accepts (LightGBM matches POSITIONALLY, per ModelBundle's
    docstring).

    A missing name means the ledger didn't store all 32 features. That
    is a hard failure, not a fill-with-NaN situation — silently
    NaN-filling is the precise failure mode Stage F's set-equality
    check exists to prevent upstream.
    """
    by_name = {
        c["feature"]: c["feature_value"]
        for c in feature_snapshot["contributions"]
    }
    missing = set(feature_order) - set(by_name)
    if missing:
        raise AssertionError(f"ledger row is missing features: {sorted(missing)}")

    # feature_value is None for a genuine NaN (thin history), never
    # for an absent key — np.nan round-trips through the model the
    # same way the original NaN did.
    values = [
        np.nan if by_name[name] is None else by_name[name]
        for name in feature_order
    ]
    return pd.DataFrame([values], columns=list(feature_order), dtype="float64")


def test_ledger_row_reconstructs_stored_probability(
    db_engine, model_bundle, written_prediction
):
    """
    The core replay test.

    Compares against probability_red_as_scored, NOT probability_red.
    The headline probability_red is the average of both corner passes
    (ADR-024 Decision 2); contributions describe only the
    red-perspective pass, so that's the number a single-row replay can
    reproduce. Comparing to the average would fail by roughly the
    symmetry gap — a real number, ~1e-2, which would look like a
    catastrophic replay failure while actually being the wrong target.
    """
    entry, original = written_prediction

    row = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

    assert row is not None, "ledger row not found after write"

    frame = _rebuild_feature_frame(
        row["feature_snapshot"], model_bundle.feature_order
    )
    replayed = float(model_bundle.booster.predict(frame)[0])

    assert replayed == pytest.approx(
        original.probability_red_as_scored, abs=_REPLAY_TOLERANCE
    )


def test_stored_contributions_reconstruct_the_score(
    db_engine, model_bundle, written_prediction
):
    """
    The TreeSHAP additivity invariant, re-checked FROM THE LEDGER.

    predict_bout() already asserts this at write time. Re-asserting it
    on the stored copy is not redundant: it proves the invariant
    survived serialization. A rounding bug in to_record() or a JSONB
    precision loss would leave the live check passing and the stored
    "why" panel no longer describing the number printed beside it.
    """
    entry, original = written_prediction

    row = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

    snapshot = row["feature_snapshot"]
    total = sum(c["log_odds"] for c in snapshot["contributions"])
    reconstructed = total + snapshot["base_log_odds"]
    replayed_prob = 1.0 / (1.0 + np.exp(-reconstructed))

    assert replayed_prob == pytest.approx(
        original.probability_red_as_scored, abs=_REPLAY_TOLERANCE
    )


def test_ledger_row_carries_full_feature_vector(
    db_engine, model_bundle, written_prediction
):
    """
    All 32, not the display top-5.

    Cheap and worth its own test: if score_upcoming.py's
    top_n=len(feature_order) ever regressed to the default 5, the
    replay test above would fail with a confusing "missing features"
    AssertionError from deep inside a helper. This fails first, with
    the actual diagnosis.
    """
    entry, _ = written_prediction

    row = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

    stored = {c["feature"] for c in row["feature_snapshot"]["contributions"]}
    assert stored == set(model_bundle.feature_order)


def test_feature_snapshot_is_detached_and_complete(db_engine, written_prediction):
    """
    Every key a future consumer needs is present in the envelope.

    Guards against a silent regression in to_record(): dropping a key
    there wouldn't fail any write, and the loss would only surface
    months later when Week 5's frontend or a settlement audit reached
    for something that was never stored.
    """
    entry, _ = written_prediction

    row = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

    snapshot = row["feature_snapshot"]
    required = {
        "bout_id",
        "model_version",
        "fighter_red_id",
        "fighter_blue_id",
        "predicted_winner_id",
        "probability_red",
        "probability_red_as_scored",
        "symmetry_gap",
        "base_log_odds",
        "contributions",
        "coverage",
        "is_calibrated",
        "predicted_at",
    }
    assert not (required - snapshot.keys())


def test_bout_status_is_joined_at_read_time(db_engine, written_prediction):
    """
    ADR-025 Decision 5: cancellation lives in bouts.status, resolved
    on read, never copied into the immutable row. Confirms the join is
    actually wired — without it, the API has no way to tell a live
    prediction from one whose fight evaporated.
    """
    entry, _ = written_prediction

    row = fetch_latest_prediction_for_bout(db_engine, entry.bout_id)

    assert row["bout_status"] == "scheduled"