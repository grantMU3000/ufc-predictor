"""
Scores every upcoming bout on the next N weeks of cards, and optionally
writes the results to the immutable prediction ledger — Week 4 Tuesday
(ADR-024) and Week 4 Wednesday (ADR-025).

WHY A SCRIPT AND NOT A ROUTE: computing a prediction means replaying
~8,600 bouts of Elo history (features/elo.py is a sequential walk, not
a lookup). That is fine once per card and wrong once per HTTP request.
More importantly, ADR-024 Decision 1 already committed to inference
being a batch operation: the ledger records each prediction once, and
every read after that serves the recorded row. Recomputing per request
would let a published number drift as new data lands, which is exactly
the audit trail this project's credibility rests on.

DRY-RUN BY DEFAULT. Scoring and writing stay two separate concerns —
this script always scores and dumps JSON; it only touches Postgres
when --write is passed. Pass --write to persist the card to the
predictions ledger (ADR-025). Each bout gets its own transaction, so
one bad write never loses the other 49, and a bout cancelled since the
last `features.snapshot` refresh is skipped, not written — checked
live against Postgres inside write_prediction(), not against the
(possibly stale) snapshot this script scores from (ADR-025 Decision 5).

Usage
-----
    uv run python -m scripts.score_upcoming
    uv run python -m scripts.score_upcoming --weeks 2
    uv run python -m scripts.score_upcoming --out data/predictions/card.json
    uv run python -m scripts.score_upcoming --write
    uv run python -m scripts.score_upcoming --weeks 2 --write

Run `uv run python -m features.snapshot` first if the snapshot is
stale — this reads Parquet, not Postgres.
"""

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
from sqlalchemy.engine import Engine

from api.dependencies import ModelBundle, build_engine, load_model_bundle
from api.services.features import (
    build_bout_features,
    compute_elo_as_of,
    snapshot_connection,
)
from api.services.inference import BoutPrediction, predict_bout
from api.services.ledger import BoutNotScheduledError, LedgerEntry, write_prediction
from api.services.records import to_record

from dotenv import load_dotenv

load_dotenv()

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
    snapshot that does not contain it. This is the right source for
    SCORING. It is deliberately NOT the source write_prediction() uses
    to check whether a bout is still happening — a bout cancelled
    since the last snapshot refresh still reads 'scheduled' here, and
    fight-week withdrawals are exactly when that staleness is likely.

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
                # top_n = every feature, not the top-5 display default.
                # ADR-025 Decision 1 commits the ledger to storing the
                # full 32-value input vector, not just the drivers a
                # "why panel" would show. contributions stays sorted
                # by |impact|, so a future top-N slice for display is
                # free — it's just the front of this same list.
                predictions.append(
                    predict_bout(model, features, top_n=len(model.feature_order))
                )
            except (ValueError, KeyError) as exc:
                failures.append((bout_id, f"{type(exc).__name__}: {exc}"))

    return predictions, failures


def write_predictions_to_ledger(
    engine: Engine,
    predictions: list[BoutPrediction],
    as_of: datetime,
) -> tuple[list[LedgerEntry], list[tuple[int, str]], list[tuple[int, str]]]:
    """
    Persist a scored card to the immutable ledger, one bout at a time.

    THREE BUCKETS, NOT TWO. A bout cancelled since scoring is not the
    same kind of event as a real write failure (ADR-025 Decision 5) —
    it's expected behavior, and lumping it in with "errors" would make
    Friday's cron alert on something that isn't a bug.

    EACH BOUT GETS ITS OWN TRANSACTION. Postgres requires a rollback
    before a connection can run anything else after a failed
    statement — sharing one transaction across 50 inserts would mean
    bout 43 failing silently kills bouts 44-50 too. This mirrors
    score_card()'s "one bad bout does not kill the card" principle,
    one layer further down the pipeline.

    Parameters
    ----------
    as_of : datetime
        One timestamp for the whole batch — captured once by the
        caller, not re-read per bout — so every prediction in this
        run resolves odds against the same moment rather than
        drifting slightly later for bout 50 than for bout 1.

    Returns
    -------
    (written, skipped, errors)
      written : successful ledger rows
      skipped : (bout_id, reason) — no longer scheduled, not an error
      errors  : (bout_id, reason) — anything else; these should fail the run
    """
    written: list[LedgerEntry] = []
    skipped: list[tuple[int, str]] = []
    errors: list[tuple[int, str]] = []

    with engine.connect() as conn:
        for prediction in predictions:
            try:
                with conn.begin():
                    entry = write_prediction(conn, prediction, as_of=as_of)
                written.append(entry)
            except BoutNotScheduledError as exc:
                skipped.append((prediction.bout_id, str(exc)))
            except Exception as exc:  # noqa: BLE001 - isolate one bad bout
                errors.append((prediction.bout_id, f"{type(exc).__name__}: {exc}"))

    return written, skipped, errors


def main() -> int:
    """Score upcoming bouts, optionally write to the ledger, print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write predictions to the immutable ledger (ADR-025). "
            "Default is dry-run: score and dump JSON only."
        ),
    )
    args = parser.parse_args()

    # One canonical timestamp for the whole run — used for the output
    # filename, the JSON body, and (if --write) the odds cutoff for
    # every bout in this batch.
    run_started_at = datetime.now(UTC)

    engine = build_engine(os.environ["DATABASE_URL"])
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

    written: list[LedgerEntry] = []
    skipped: list[tuple[int, str]] = []
    errors: list[tuple[int, str]] = []

    if args.write:
        written, skipped, errors = write_predictions_to_ledger(
            engine, predictions, as_of=run_started_at
        )
        print(
            f"\nledger: {len(written)} written, {len(skipped)} skipped "
            f"(cancelled), {len(errors)} error(s)"
        )
        if skipped:
            print("  skipped (bout no longer scheduled):")
            for bout_id, reason in skipped:
                print(f"    bout {bout_id}: {reason}")
        if errors:
            print("  ERRORS (need investigation):")
            for bout_id, reason in errors:
                print(f"    bout {bout_id}: {reason}")

    out_dir = args.out.parent if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (
        out_dir / f"scored_{run_started_at:%Y%m%d_%H%M%S}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "model_version": model.version,
                "scored_at": run_started_at.isoformat(),
                "weeks": args.weeks,
                "n_scored": len(predictions),
                "n_failed": len(failures),
                "write_mode": args.write,
                "predictions": [to_record(p) for p in predictions],
                "failures": [
                    {"bout_id": b, "error": m} for b, m in failures
                ],
                "ledger": (
                    {
                        "written": [
                            {
                                "bout_id": e.bout_id,
                                "prediction_id": e.prediction_id,
                                "created_at": e.created_at.isoformat(),
                            }
                            for e in written
                        ],
                        "skipped": [
                            {"bout_id": b, "reason": r} for b, r in skipped
                        ],
                        "errors": [
                            {"bout_id": b, "reason": r} for b, r in errors
                        ],
                    }
                    if args.write
                    else None
                ),
            },
            f,
            indent=2,
        )

    print(f"\nScored {len(predictions)}/{len(bouts)} -> {out_path}")
    # Non-zero exit if scoring failed OR a ledger write genuinely
    # failed, so Friday's cron notices. A cancelled-bout skip is
    # expected behavior (ADR-025 Decision 5) and does NOT count —
    # alerting on it would train you to ignore the alert.
    return 1 if (failures or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())