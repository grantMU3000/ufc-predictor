"""
Builds the model-ready feature matrix for a single bout at inference
time — Week 4 Tuesday, ADR-024.

THE ONE JOB: produce, for a scheduled bout, a matrix that is
indistinguishable from the row the training matrix would have held
for that same bout. Not "similar." Identical, to floating point.

That's why this module computes almost nothing itself. Every value
comes from features/store.py, features/symmetrize.py,
features/elo.py and features/differential.py — the exact functions
that built train.parquet in Week 2 (ADR-024 Decision 1). A second
implementation of any of them would produce plausible, confident,
wrong probabilities with nothing anywhere to flag it. This file is
glue and nothing else.

DuckDB, not SQLAlchemy: the whole feature layer takes a
DuckDBPyConnection. Rather than port ~30 leakage-audited functions
to a second database driver, inference runs them over the same
Parquet snapshot (features/snapshot.py) they were built against.
Wednesday's ledger makes prediction a batch operation — each bout is
predicted once and served from the ledger afterward — so snapshot
freshness, not per-request latency, is the thing that matters.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from features.bout_history import get_prior_bouts
from features.bout_stats_history import get_prior_bout_stats
from features.differential import to_differential
from features.elo import (
    DEFAULT_INITIAL_RATING,
    compute_current_ratings,
    tuned_k_factor,
)
from features.labels import get_completed_decided_bouts
from features.store import _get_bout_context, build_feature_row
from features.symmetrize import _symmetrize_row
from features.tier2 import total_ufc_fights

# NOTE on the two underscore imports above: _get_bout_context and
# _symmetrize_row are private to their modules, and importing across
# that line is normally a smell. It is the correct call here — the
# alternative is reimplementing them, which is exactly the skew this
# module exists to prevent. Worth promoting both to public names in a
# follow-up PR; not worth touching leakage-defense files today.

SNAPSHOT_DIR = Path("data/processed")

# Only the four tables the Tier 1/2 feature functions and Elo touch.
# snapshot.py exports six; fighter_aliases and odds_snapshots are
# deliberately absent — odds must never be reachable from an
# inference feature path (ADR-002).
SNAPSHOT_TABLES = ("fighters", "events", "bouts", "bout_stats")


@dataclass(frozen=True)
class DataCoverage:
    """
    How much history the model actually had to work with — ADR-024
    Decision 4.

    A prediction built from 6 of 32 features is not the same claim as
    one built from 32, even though both come back as a confident-
    looking probability. Wikipedia stub fighters (ADR-010), true
    debutants, and short-notice replacements all land here.

    This never gates a prediction. It labels one.
    """

    n_features_present: int
    n_features_total: int
    red_prior_bouts: int
    blue_prior_bouts: int

    @property
    def fraction(self) -> float:
        """Present / total, as a 0-1 float. 32/32 -> 1.0."""
        return self.n_features_present / self.n_features_total


@dataclass(frozen=True)
class BoutFeatureMatrix:
    """
    Everything the inference service needs for one bout.

    `matrix` holds TWO rows, red-perspective first, blue second, in
    that order always. Two rows because ADR-024 Decision 2 predicts
    both corner orderings and averages them. Red first because
    to_differential emits columns in DataFrame order, and a
    blue-first frame produces the same 32 column NAMES in a different
    ORDER — which LightGBM, matching by position, would happily
    consume and silently misread.

    Columns are already reindexed to the model's feature_order, so
    the caller hands `matrix.to_numpy()` straight to Booster.predict.
    """

    bout_id: int
    fighter_red_id: int
    fighter_blue_id: int
    as_of_date: date
    matrix: pd.DataFrame
    coverage: DataCoverage


@contextmanager
def snapshot_connection(
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    A DuckDB connection with views over the Parquet snapshot, which
    clears the feature-store caches on the way out.

    THE CACHE CLEARING IS NOT OPTIONAL. get_prior_bouts and
    get_prior_bout_stats are decorated with @cache, and both of their
    docstrings flag the exact situation this API creates: in a
    long-lived process the cache never expires, so after a snapshot
    refresh they keep serving pre-refresh history until the process
    restarts. It also grows without bound.

    A fresh connection per batch means new cache keys (the connection
    object is part of the key), so a stale entry can't be hit. The
    explicit clear on exit stops the memory growth. Both are handled
    here so no caller has to remember either.

    Raises
    ------
    FileNotFoundError if the snapshot is missing — a clear message
    beats a DuckDB parse error thrown from inside a feature function
    six frames down.
    """
    for table in SNAPSHOT_TABLES:
        path = snapshot_dir / f"{table}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Snapshot table missing: {path}. Run `uv run python -m "
                f"features.snapshot` to refresh it from Postgres."
            )

    con = duckdb.connect()
    try:
        for table in SNAPSHOT_TABLES:
            con.execute(
                f"CREATE VIEW {table} AS SELECT * FROM "
                f"read_parquet('{snapshot_dir / f'{table}.parquet'}')"
            )
        yield con
    finally:
        get_prior_bouts.cache_clear()
        get_prior_bout_stats.cache_clear()
        con.close()


def compute_elo_as_of(
    con: duckdb.DuckDBPyConnection, as_of_date: date
) -> dict[int, float]:
    """
    Every fighter's Elo rating going into `as_of_date`.

    Replays the full history of completed, decided bouts occurring
    strictly before that date, using ADR-014's tuned K.

    WHY `< as_of_date` RATHER THAN THE FULL HISTORY: for an upcoming
    fight those are the same set, so it costs nothing. For a
    COMPLETED bout it's the difference between a correct answer and
    a wrong one — the training matrix holds that bout's mid-walk
    pre-fight rating, not today's. Filtering here means the parity
    test and live inference run the identical code path, which is
    the only way the parity test proves anything.

    EXPENSIVE: walks ~8,600 bouts. Compute once per batch and pass
    the result into build_bout_features, rather than once per bout.
    """
    labels = get_completed_decided_bouts(con)
    labels = labels[
        pd.to_datetime(labels["event_date"]) < pd.Timestamp(as_of_date)
    ].reset_index(drop=True)
    return compute_current_ratings(labels, k_factor=tuned_k_factor)

def _coerce_model_features_to_float(
    df: pd.DataFrame, feature_order: tuple[str, ...]
) -> pd.DataFrame:
    """
    Force every self_/opp_ column the model expects to float64 before
    differencing.

    WHY THIS EXISTS (caught by the Step 5 parity test, bout 5756):
    pandas infers a column's dtype from its values. A training frame
    has ~15,000 rows, so a mostly-missing feature like
    submission_success_rate still holds real floats somewhere and
    types as float64 with NaNs. An inference frame has TWO rows — if
    both fighters are missing that stat, the column is all-None,
    pandas types it `object`, and to_differential drops it through
    the "non-numeric, can't subtract" branch meant for stance.

    Same code, same inputs, different output, purely as a function of
    row count. No exception, no warning — the exact silent-skew class
    this whole day exists to prevent.

    Only columns named in feature_order are touched, so genuinely
    non-numeric ones (self_stance, self_stance_matchup_descriptive)
    stay object and keep getting dropped, exactly as training did.
    A real string in a numeric slot raises here rather than being
    quietly swallowed.
    """
    coerced = df.copy()
    for name in feature_order:
        suffix = name[len("diff_"):]
        for side in ("self", "opp"):
            column = f"{side}_{suffix}"
            if column in coerced.columns:
                coerced[column] = coerced[column].astype("float64")
    return coerced

def build_bout_features(
    con: duckdb.DuckDBPyConnection,
    bout_id: int,
    feature_order: tuple[str, ...],
    elo_ratings: dict[int, float] | None = None,
) -> BoutFeatureMatrix:
    """
    The full Stage A -> F recipe for one bout, in training order.

    Simple version: this rebuilds one row of the training matrix from
    scratch, for a fight that hasn't happened yet — using the same
    functions, called in the same order, that built every row the
    model learned from.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        From snapshot_connection().
    bout_id : int
        Any bout in `bouts`. Works for status='scheduled' (the live
        case) and 'completed' (the parity test) alike.
    feature_order : tuple[str, ...]
        ModelBundle.feature_order — the model's positional contract.
    elo_ratings : dict[int, float] | None
        Precomputed output of compute_elo_as_of for this bout's
        as_of_date. Computed internally if omitted; pass it in when
        scoring a whole card so the ~8,600-bout walk happens once.

    Returns
    -------
    BoutFeatureMatrix — 2 rows (red-perspective first), columns
    already in feature_order.

    Raises
    ------
    ValueError if bout_id doesn't exist (from _get_bout_context), or
    if the built feature set doesn't exactly match feature_order.
    """
    # --- Stage A: raw red_/blue_ row, as_of the bout's own date ---
    # as_of_date is the bout's event_date, NOT today (ADR-024
    # Decision 5). build_feature_row does date ARITHMETIC with it —
    # age_at_fight and days_since_last_fight both subtract from it —
    # so passing today() for a fight three weeks out would
    # systematically understate both against every training row.
    context = _get_bout_context(con, bout_id)
    fighter_red_id = int(context["fighter_red_id"])
    fighter_blue_id = int(context["fighter_blue_id"])
    as_of_date = context["event_date"]

    row = build_feature_row(con, bout_id)

    # --- Stage B: symmetrize, RED FIRST ---
    # Order is load-bearing, not stylistic. _symmetrize_row walks the
    # input dict in insertion order, and that dict is red-keys-first.
    # From red's perspective those become self_* first; from blue's
    # they become opp_* first. pandas unions keys in first-seen
    # order, so a blue-first frame reverses the column layout that
    # to_differential (and therefore the trained model) expects.
    self_rows = [
        _symmetrize_row(row, fighter_red_id, fighter_blue_id, self_id)
        for self_id in (fighter_red_id, fighter_blue_id)
    ]

    # --- Stage D: Elo, appended last ---
    # Appended after symmetrization so self_elo_pre/opp_elo_pre land
    # at the END of the frame — mirroring attach_by_corner's merge,
    # which is why diff_elo_pre is the 32nd and final feature.
    if elo_ratings is None:
        elo_ratings = compute_elo_as_of(con, as_of_date)

    red_elo = elo_ratings.get(fighter_red_id, DEFAULT_INITIAL_RATING)
    blue_elo = elo_ratings.get(fighter_blue_id, DEFAULT_INITIAL_RATING)

    for self_row in self_rows:
        is_red = self_row["source_corner"] == "red"
        self_row["self_elo_pre"] = red_elo if is_red else blue_elo
        self_row["opp_elo_pre"] = blue_elo if is_red else red_elo

    symmetrized = pd.DataFrame(self_rows)

    # --- Stage E: difference ---
    # Coerce FIRST. On a 2-row frame, an all-None column types as
    # object and gets silently dropped downstream — see
    # _coerce_model_features_to_float.
    symmetrized = _coerce_model_features_to_float(symmetrized, feature_order)

    # require_label=False: a scheduled bout has no self_won. The
    # alternative — attaching a fake label — would put a counterfeit
    # answer key inside the inference path, the exact thing
    # features/labels.py was split out to make impossible.
    features, _ = to_differential(symmetrized, verbose=False, require_label=False)

    # --- Stage F: positional contract ---
    # Set equality checked BEFORE the reindex. Reindexing alone would
    # silently drop an unexpected extra column and silently insert
    # NaN for a missing one; neither would raise, and both would
    # produce a confident wrong number.
    built = set(features.columns)
    expected = set(feature_order)
    if built != expected:
        raise ValueError(
            f"feature mismatch for bout {bout_id}: "
            f"missing={sorted(expected - built)}, "
            f"unexpected={sorted(built - expected)}"
        )
    matrix = features[list(feature_order)]

    coverage = DataCoverage(
        n_features_present=int(matrix.iloc[0].notna().sum()),
        n_features_total=len(feature_order),
        red_prior_bouts=total_ufc_fights(con, fighter_red_id, as_of_date),
        blue_prior_bouts=total_ufc_fights(con, fighter_blue_id, as_of_date),
    )

    return BoutFeatureMatrix(
        bout_id=bout_id,
        fighter_red_id=fighter_red_id,
        fighter_blue_id=fighter_blue_id,
        as_of_date=as_of_date,
        matrix=matrix,
        coverage=coverage,
    )


if __name__ == "__main__":
    # Smoke test: build features for one scheduled bout and eyeball
    # the shape, the column order, and the coverage. Deliberately
    # prints the matrix rather than asserting on values — real
    # assertions belong in the parity test (Step 5), against a
    # completed bout whose training-matrix row already exists.
    import json

    with open("models/v1/metadata.json") as f:
        feature_order = tuple(json.load(f)["feature_list"])

    with snapshot_connection() as con:
        scheduled = con.execute(
            "SELECT b.id, e.event_date FROM bouts b "
            "JOIN events e ON e.id = b.event_id "
            "WHERE b.status = 'scheduled' ORDER BY e.event_date LIMIT 1"
        ).fetchone()

        if scheduled is None:
            raise SystemExit(
                "No scheduled bouts in the snapshot. Refresh upcoming "
                "events, then re-run features.snapshot."
            )

        result = build_bout_features(con, int(scheduled[0]), feature_order)

    print(f"bout_id={result.bout_id}  as_of={result.as_of_date}")
    print(f"matrix shape: {result.matrix.shape}  (expect (2, 32))")
    print(f"column order matches feature_order: "
          f"{list(result.matrix.columns) == list(feature_order)}")
    print(f"coverage: {result.coverage.n_features_present}"
          f"/{result.coverage.n_features_total} "
          f"({result.coverage.fraction:.0%})")
    print(f"prior bouts — red: {result.coverage.red_prior_bouts}, "
          f"blue: {result.coverage.blue_prior_bouts}")
    print(result.matrix.T)