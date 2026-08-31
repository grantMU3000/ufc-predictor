"""
Runs the frozen model against a built feature matrix and explains the
result — Week 4 Tuesday, ADR-024 Decisions 2, 3 and 4.

Framework-free on purpose: no FastAPI imports anywhere in this file.
Everything here is a pure function of (ModelBundle, BoutFeatureMatrix),
so the arithmetic that decides what the project publicly claims can be
unit-tested without spinning up a server or touching a database.

Division of labour: api/services/features.py answers "what were the
numbers going into this fight." This file answers "what does the model
make of them, and why." Keeping those apart means a feature-parity bug
and an attribution bug can never be mistaken for each other.
"""

from dataclasses import dataclass

import numpy as np

from api.dependencies import ModelBundle
from api.services.features import BoutFeatureMatrix, DataCoverage

# Row order inside BoutFeatureMatrix.matrix, fixed by
# build_bout_features. Named rather than inlined as 0/1 because every
# sign convention below depends on knowing which is which.
RED_PERSPECTIVE = 0
BLUE_PERSPECTIVE = 1

# Tolerance for the contributions-sum-to-raw-score invariant. TreeSHAP
# is exact for tree models, so any real gap is a bug, not drift — this
# only absorbs float64 accumulation across ~500 trees.
_RECONSTRUCTION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FeatureContribution:
    """
    One feature's push on a single prediction, from the RED corner's
    point of view.

    `log_odds` is a raw TreeSHAP value, NOT percentage points
    (ADR-024 Decision 3). The same +0.4 moves a near-coinflip a long
    way and a 90/10 barely at all, so it cannot be read as "this
    added 8% to his chances." Converting it is the frontend's problem
    in Week 5, and `favors` plus relative magnitude is enough to
    phrase it directionally without lying.
    """

    feature: str
    log_odds: float
    feature_value: float | None  # the diff_ input; None when NaN
    favors: str  # "red" | "blue"


@dataclass(frozen=True)
class BoutPrediction:
    """
    Everything the API returns for one bout, plus the diagnostics that
    keep the claim honest.

    THREE PROBABILITIES, deliberately, not one:
      - probability_red: the headline. Averaged across both corner
        orderings (ADR-024 Decision 2), so it can't depend on which
        fighter Wikipedia happened to list first.
      - probability_red_as_scored: the red-perspective pass alone.
        This is the number `contributions` actually explains and
        reconstructs.
      - symmetry_gap: the disagreement between the two passes. Should
        sit near zero; a large value means the model retained real
        corner asymmetry despite symmetrized training, which is a
        LEAKAGE_LOG.md entry, not a rounding note.

    is_calibrated is hardcoded False and comes from the registry row,
    not from a constant here — v1 shipped uncalibrated (ADR-017),
    both calibrators having been rejected. Surfacing it as a field
    means a future calibrated v2 changes a database row, and the
    frontend's disclaimer follows automatically.
    """

    bout_id: int
    model_version: str
    fighter_red_id: int
    fighter_blue_id: int
    predicted_winner_id: int
    probability_red: float
    probability_red_as_scored: float
    symmetry_gap: float
    base_log_odds: float
    contributions: tuple[FeatureContribution, ...]
    coverage: DataCoverage
    is_calibrated: bool


def _sigmoid(x: float) -> float:
    """Log-odds to probability. LightGBM's binary objective link function."""
    return 1.0 / (1.0 + np.exp(-x))


def average_symmetric_probability(
    p_red_perspective: float, p_blue_perspective: float
) -> float:
    """
    Fold two corner-swapped predictions into one P(red wins).

    Simple version: you ask a judge to score the fight, then ask again
    with the fighters' names swapped, and average the two answers. If
    the judge is truly neutral you get the same number twice and lose
    nothing. If they lean toward whoever's introduced first, averaging
    cancels it out exactly.

    The model outputs P(self wins), and "self" is red in one pass and
    blue in the other — so the second pass has to be flipped to
    1 - p before it means the same thing as the first.

    Pulled out as its own named function because it is three
    characters away from being wrong (averaging p_A and p_B directly
    would produce a number pinned near 0.5 for every fight) and a
    one-line unit test catches that forever.
    """
    return (p_red_perspective + (1.0 - p_blue_perspective)) / 2.0


def symmetry_gap(p_red_perspective: float, p_blue_perspective: float) -> float:
    """
    How far the two corner-swapped passes disagree, in probability
    points. Zero means perfectly corner-agnostic.

    Logged rather than discarded (ADR-024 Decision 2). Symmetrized
    training TAUGHT the model to ignore corner; it did not guarantee
    it. This is the standing measurement of whether that lesson took,
    computed on every real prediction rather than assumed from a
    training-time property.
    """
    return abs(p_red_perspective - (1.0 - p_blue_perspective))


def rank_contributions(
    feature_order: tuple[str, ...],
    contribution_row: np.ndarray,
    feature_values: np.ndarray,
    top_n: int,
) -> tuple[FeatureContribution, ...]:
    """
    Turn one row of TreeSHAP output into the top-N drivers, biggest
    absolute push first.

    Sorted by ABSOLUTE value, not signed value: the question a "why"
    panel answers is "what mattered most," and a large negative is
    every bit as much of an answer as a large positive. Sorting signed
    would silently return the top N reasons red wins and hide every
    reason he doesn't.

    Parameters
    ----------
    contribution_row : np.ndarray
        One row of predict(pred_contrib=True), WITHOUT the trailing
        base-value column — the caller strips it, since it isn't a
        feature and would otherwise rank as one.
    feature_values : np.ndarray
        The matching diff_ input values, for display alongside.
    """
    order = np.argsort(np.abs(contribution_row))[::-1][:top_n]

    ranked = []
    for i in order:
        value = feature_values[i]
        ranked.append(
            FeatureContribution(
                feature=feature_order[i],
                log_odds=float(contribution_row[i]),
                # NaN is a real state here (thin history), not an
                # error — it becomes null in JSON rather than being
                # coerced to a fake 0.0 that would read as "this
                # fighter's reach advantage is exactly zero."
                feature_value=None if np.isnan(value) else float(value),
                favors="red" if contribution_row[i] > 0 else "blue",
            )
        )
    return tuple(ranked)


def predict_bout(
    model: ModelBundle, features: BoutFeatureMatrix, top_n: int = 5
) -> BoutPrediction:
    """
    Score one bout and explain it.

    Simple version: hand the model the two-row card built upstream,
    read both answers, average them, and ask the model to itemize its
    reasoning on the red-perspective pass.

    Parameters
    ----------
    model : ModelBundle
        From api.dependencies. feature_order is the positional
        contract; the matrix was already reindexed to it upstream.
    features : BoutFeatureMatrix
        From api.services.features.build_bout_features. Exactly 2
        rows, red-perspective first.
    top_n : int
        How many contributions to return.

    Raises
    ------
    ValueError if the matrix isn't 2 rows, or if the contributions
    fail to reconstruct the model's own raw score.
    """
    matrix = features.matrix
    if len(matrix) != 2:
        raise ValueError(
            f"expected 2 rows (red- and blue-perspective), got {len(matrix)}"
        )

    # The DataFrame goes in, not .to_numpy(). LightGBM validates
    # column names against the names baked into the booster at
    # training time and raises on a mismatch — a free second check on
    # top of the set-equality assertion build_bout_features already
    # ran, and the only one of the two that can catch the artifact
    # itself being wrong rather than our matrix.
    probabilities = model.booster.predict(matrix)
    contributions = model.booster.predict(matrix, pred_contrib=True)

    probabilities = np.asarray(probabilities, dtype=float)
    contributions = np.asarray(contributions, dtype=float)

    p_red_pass = float(probabilities[RED_PERSPECTIVE])
    p_blue_pass = float(probabilities[BLUE_PERSPECTIVE])

    probability_red = average_symmetric_probability(p_red_pass, p_blue_pass)
    gap = symmetry_gap(p_red_pass, p_blue_pass)

    # pred_contrib returns n_features + 1 columns; the LAST is the
    # base value (the model's average log-odds output), and the rest
    # sum with it to the raw score. Splitting it off here keeps it out
    # of the ranking, where it would otherwise dominate as the single
    # largest "feature."
    red_row = contributions[RED_PERSPECTIVE]
    feature_contributions = red_row[:-1]
    base_log_odds = float(red_row[-1])

    # INVARIANT: TreeSHAP is exact for tree models, so contributions
    # plus base must equal the raw score, and its sigmoid must equal
    # the probability LightGBM independently reported. If this ever
    # trips, the explanation on screen does not describe the number
    # next to it — which is worse than having no explanation at all,
    # so it raises rather than warns.
    reconstructed = float(feature_contributions.sum()) + base_log_odds
    if abs(_sigmoid(reconstructed) - p_red_pass) > _RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"contributions do not reconstruct the prediction for bout "
            f"{features.bout_id}: sigmoid(sum)={_sigmoid(reconstructed):.9f} "
            f"vs predict()={p_red_pass:.9f}"
        )

    # Ties break to red. Arbitrary but FIXED — an unspecified tie-break
    # would make the ledger non-reproducible on an exact 0.5, and the
    # ledger being replayable is the whole credibility argument.
    predicted_winner_id = (
        features.fighter_red_id
        if probability_red >= 0.5
        else features.fighter_blue_id
    )

    return BoutPrediction(
        bout_id=features.bout_id,
        model_version=model.version,
        fighter_red_id=features.fighter_red_id,
        fighter_blue_id=features.fighter_blue_id,
        predicted_winner_id=predicted_winner_id,
        probability_red=probability_red,
        probability_red_as_scored=p_red_pass,
        symmetry_gap=gap,
        base_log_odds=base_log_odds,
        contributions=rank_contributions(
            model.feature_order,
            feature_contributions,
            matrix.iloc[RED_PERSPECTIVE].to_numpy(dtype=float),
            top_n,
        ),
        coverage=features.coverage,
        # Read from the registry row, not hardcoded — a calibrated v2
        # flips this by registering a row, with no code change here.
        is_calibrated=bool(model.registry_row.get("is_calibrated", False)),
    )


if __name__ == "__main__":
    # End-to-end smoke test against the next scheduled bout. Prints
    # rather than asserts, same as features.py's block — the real
    # assertions live in tests/, and the parity test (Step 5) is what
    # actually proves this matches training.
    import os

    from sqlalchemy import create_engine

    from api.dependencies import load_model_bundle
    from api.services.features import build_bout_features, snapshot_connection

    engine = create_engine(os.environ["DATABASE_URL"])
    bundle = load_model_bundle(engine)

    with snapshot_connection() as con:
        scheduled = con.execute(
            "SELECT b.id FROM bouts b JOIN events e ON e.id = b.event_id "
            "WHERE b.status = 'scheduled' ORDER BY e.event_date LIMIT 1"
        ).fetchone()
        if scheduled is None:
            raise SystemExit("No scheduled bouts in the snapshot.")

        features = build_bout_features(
            con, int(scheduled[0]), bundle.feature_order
        )

    result = predict_bout(bundle, features)

    print(f"bout {result.bout_id}  model {result.model_version}")
    print(f"P(red) = {result.probability_red:.4f}   "
          f"(red pass {result.probability_red_as_scored:.4f})")
    print(f"symmetry gap: {result.symmetry_gap:.2e}   "
          f"base log-odds: {result.base_log_odds:+.4f}")
    print(f"predicted winner id: {result.predicted_winner_id}")
    print(f"coverage: {result.coverage.fraction:.0%}  "
          f"calibrated: {result.is_calibrated}")
    print("\ntop contributions (log-odds, red perspective):")
    for c in result.contributions:
        value = "NaN" if c.feature_value is None else f"{c.feature_value:+.3f}"
        print(f"  {c.log_odds:+.4f}  {c.feature:<32} "
              f"value={value:>10}  favors {c.favors}")