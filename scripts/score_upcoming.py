"""
Scores every upcoming bout on the next N weeks of cards — Week 4
Tuesday, Step 6 (ADR-024).

WHY A SCRIPT AND NOT A ROUTE: computing a prediction means replaying
~8,600 bouts of Elo history (features/elo.py is a sequential walk, not
a lookup). That is fine once per card and wrong once per HTTP request.
More importantly, ADR-024 Decision 1 already committed to inference
being a batch operation: Wednesday's ledger records each prediction
once, and every read after that serves the recorded row. Recomputing
per request would let a published number drift as new data lands,
which is exactly the audit trail this project's credibility rests on.

TODAY THIS WRITES NOTHING TO THE DATABASE. It scores, reports, and
dumps JSON. The ledger write is Wednesday's job and needs things this
script deliberately does not touch — market odds at prediction time,
the feature snapshot, immutability guarantees. Scoring and recording
are two jobs; keeping them in two files means a bug in one cannot hide
inside the other.

Usage
-----
    uv run python -m scripts.score_upcoming
    uv run python -m scripts.score_upcoming --weeks 2
    uv run python -m scripts.score_upcoming --out data/predictions/card.json

Run `uv run python -m features.snapshot` first if the snapshot is
stale — this reads Parquet, not Postgres.
"""

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
from sqlalchemy import create_engine

from api.dependencies import ModelBundle, load_model_bundle
from api.services.features import (
    build_bout_features,
    compute_elo_as_of,
    snapshot_connection,
)
from api.services.inference import BoutPrediction, predict_bout

DEFAULT_WEEKS = 4
DEFAULT_OUT = Path("data/predictions")

# Below this, the prediction is served with a visible warning rather
# than suppressed. Two fighters with real records land at 1.0; a
# Wikipedia stub (ADR-010) or a true debutant drags it down. Chosen as
# a reporting threshold only — it gates nothing, per ADR-024 Decision
# 4, which says coverage LABELS a prediction rather than blocking it.
LOW_COVERAGE_WARN = 0.75

# Reference ceiling for the corner-symmetry check, read from the
# frozen artifact rather than picked here. metadata.json records
# max_pair_dev = 0.0806 for artifact B — the worst corner disagreement
# measured across the whole test set at unlock. A bout above that is
# outside the range the model was characterized on and belongs in
# LEAKAGE_LOG.md; anything under it is documented, expected behavior.
SYMMETRY_GAP_KEY = "max_pair_dev"


def fetch_upcoming_bout_ids(
    con: duckdb.DuckDBPyConnection, weeks: int
) -> list[tuple[int, str, str]]:
    """
    Scheduled bouts in the next `weeks` weeks, oldest event first.

    Reads the Parquet snapshot, not Postgres — the same source the
    feature functions read, so a bout can never be scored against a
    snapshot that does not contain it.

    Returns
    -------
    list of (bout_id, event_date, event_name), ordered by date then
    bout id. Grouping by date matters downstream: Elo is replayed once
    per distinct date, not once per bout.
    """
    today = datetime.now(UTC).date()
    horizon = today + timedelta(weeks=weeks)

    rows = con.execute(
        """
        SELECT b.id, CAST(e.event_date AS VARCHAR), e.name
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE b.status = 'scheduled'
          AND e.event_date >= ?
          AND e.event_date <= ?
        ORDER BY e.event_date ASC, b.id ASC
        """,
        [today, horizon],
    ).fetchall()

    return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]


def score_card(
    con: duckdb.DuckDBPyConnection,
    model: ModelBundle,
    bouts: list[tuple[int, str, str]],
) -> tuple[list[BoutPrediction], list[tuple[int, str]]]:
    """
    Score a list of bouts, replaying Elo once per distinct event date.

    Simple version: the expensive part is rebuilding the league
    standings from scratch, and every fight on the same card shares
    the same standings. So compute them once per date and reuse.

    ONE BAD BOUT DOES NOT KILL THE CARD. A single fighter who failed
    to resolve (a Wikipedia stub with no real record, a bout row
    missing a corner) raises inside build_bout_features. Letting that
    abort the run would mean one unresolvable prelim blocks the main
    event from ever being scored. Failures are collected and reported
    at the end instead — loudly, in the summary and the exit code, not
    swallowed.

    Returns
    -------
    (predictions, failures) — failures are (bout_id, error message).
    """
    predictions: list[BoutPrediction] = []
    failures: list[tuple[int, str]] = []

    by_date: dict[str, list[int]] = {}
    for bout_id, event_date, _ in bouts:
        by_date.setdefault(event_date, []).append(bout_id)

    for event_date, bout_ids in by_date.items():
        # as_of_date is each bout's own event_date (ADR-024 Decision
        # 5) — shared across a card because a card is one date.
        elo = compute_elo_as_of(con, datetime.fromisoformat(event_date).date())

        for bout_id in bout_ids:
            try:
                features = build_bout_features(
                    con, bout_id, model.feature_order, elo_ratings=elo
                )
                predictions.append(predict_bout(model, features))
            except (ValueError, KeyError) as exc:
                failures.append((bout_id, f"{type(exc).__name__}: {exc}"))

    return predictions, failures


def to_record(prediction: BoutPrediction) -> dict:
    """
    Flatten a BoutPrediction into a JSON-serializable dict.

    Kept as its own function because Wednesday's ledger writer needs
    the same flattening. Better one shape defined once than two that
    drift — the ledger and this file's JSON should always describe the
    same prediction identically.
    """
    record = asdict(prediction)
    record["coverage"]["fraction"] = prediction.coverage.fraction
    record["scored_at"] = datetime.now(UTC).isoformat()
    return record


def main() -> int:
    """Score upcoming bouts, write JSON, print a summary. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    engine = create_engine(os.environ["DATABASE_URL"])
    model = load_model_bundle(engine)

    # The symmetry reference from the frozen artifact, not a guess.
    max_pair_dev = max(
        (m.get(SYMMETRY_GAP_KEY, 0.0) for m in model.registry_row.get("metrics", [])),
        default=None,
    )

    with snapshot_connection() as con:
        bouts = fetch_upcoming_bout_ids(con, args.weeks)
        if not bouts:
            print(f"No scheduled bouts in the next {args.weeks} weeks.")
            print("Refresh upcoming events, then re-run features.snapshot.")
            return 0

        print(f"Scoring {len(bouts)} bout(s) with model {model.version}...\n")
        predictions, failures = score_card(con, model, bouts)

    names = {b[0]: (b[1], b[2]) for b in bouts}
    for p in sorted(predictions, key=lambda x: names[x.bout_id][0]):
        event_date, event_name = names[p.bout_id]
        flags = []
        if p.coverage.fraction < LOW_COVERAGE_WARN:
            flags.append(f"LOW COVERAGE {p.coverage.fraction:.0%}")
        if max_pair_dev is not None and p.symmetry_gap > max_pair_dev:
            flags.append(f"SYMMETRY GAP {p.symmetry_gap:.4f} > {max_pair_dev:.4f}")
        suffix = f"   [{'; '.join(flags)}]" if flags else ""
        print(
            f"  {event_date}  bout {p.bout_id:>6}  "
            f"P(red)={p.probability_red:.4f}  "
            f"pick={p.predicted_winner_id}{suffix}"
        )

    if failures:
        print(f"\n{len(failures)} bout(s) failed to score:")
        for bout_id, message in failures:
            print(f"  bout {bout_id}: {message}")

    out_dir = args.out.parent if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (
        out_dir / f"scored_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_version": model.version,
                "scored_at": datetime.now(UTC).isoformat(),
                "weeks": args.weeks,
                "n_scored": len(predictions),
                "n_failed": len(failures),
                "predictions": [to_record(p) for p in predictions],
                "failures": [
                    {"bout_id": b, "error": m} for b, m in failures
                ],
            },
            f,
            indent=2,
        )

    print(f"\nScored {len(predictions)}/{len(bouts)} -> {out_path}")
    # Non-zero exit on any failure, so Friday's GitHub Action notices.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())