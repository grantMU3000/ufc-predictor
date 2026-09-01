"""
Grades committed predictions against real results, and optionally writes
them to prediction_results — Week 4 Thursday (ADR-026).

WHY A SCRIPT AND NOT A ROUTE: settlement is an after-the-fact batch job
that runs once per card, on Greco's ingest schedule rather than on user
traffic. Nothing about it belongs on a request path. It is the mirror
image of scripts/score_upcoming.py — that script commits a claim before
the fight, this one grades it after — and the two share a shape on
purpose, so Friday's cron treats them identically.

DRY-RUN BY DEFAULT. Classification and persistence stay separate: this
script always classifies every unsettled prediction and dumps JSON; it
only touches prediction_results when --write is passed. The
classification pass is byte-identical in both modes, so a dry run is a
real preview, not a lookalike code path.

THREE BUCKETS, NOT TWO, same as the ledger writer. A cancelled bout and
a draw are permanent, expected states (ADR-026 Decisions 3 and 4) — not
failures. Exiting non-zero on those would put the cron permanently red
and train everyone to ignore it.

STALE BOUTS ARE REPORTED, NEVER AUTO-CANCELLED (ADR-026 Decision 8).
A bout still 'scheduled' on a completed card has four possible causes,
and only one of them is benign to fix automatically. Guessing would
bury the dangerous case: if BOTH fighters appear in completed bouts on
that card, the row should have been claimed and wasn't — almost
certainly an unreconciled duplicate fighter (ADR-013).

Usage
-----
    uv run python -m scripts.settle_predictions
    uv run python -m scripts.settle_predictions --write
    uv run python -m scripts.settle_predictions --stale-days 5
    uv run python -m scripts.settle_predictions --out data/settlements/run.json
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from api.dependencies import build_engine
from api.services.settlement import (
    StaleDiagnosis,
    diagnose_stale,
    find_stale_scheduled_bouts,
    settle_predictions,
)

load_dotenv()

DEFAULT_OUT = Path("data/settlements")

# How long after an event a bout may sit 'scheduled' before it is
# reported. Greco ingests on its own cadence, so a card graded the
# morning after would flag its whole undercard as stale. Two days
# absorbs that lag without letting a genuinely broken row hide for
# a week. Reporting threshold only — it gates nothing.
DEFAULT_STALE_DAYS = 2


def main() -> int:
    """Grade unsettled predictions, optionally persist, print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write prediction_results rows (ADR-026). "
            "Default is dry-run: classify and dump JSON only."
        ),
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=(
            "How many days past an event date a bout may stay 'scheduled' "
            f"before it is reported. Default {DEFAULT_STALE_DAYS}."
        ),
    )
    args = parser.parse_args()

    # One canonical timestamp for the run — output filename and JSON body,
    # matching score_upcoming.py. settled_at itself is a Postgres
    # server_default, so this never becomes a competing source of truth
    # for when a row was actually written.
    run_started_at = datetime.now(UTC)

    engine = build_engine(os.environ["DATABASE_URL"])

    mode = "WRITE" if args.write else "DRY RUN"
    print(f"Settling predictions — {mode}\n")

    report = settle_predictions(engine, write=args.write)

    verb = "settled" if args.write else "would settle"
    print(f"{verb}: {len(report.settled)}")
    for outcome in report.settled:
        mark = "correct" if outcome.correct else "wrong  "
        flag = (
            "   [CORNER DERIVATION DISAGREES]"
            if outcome.corner_derivation_disagrees
            else ""
        )
        print(
            f"  prediction {outcome.prediction_id:>6}  bout {outcome.bout_id:>6}  "
            f"{mark}  log_loss={outcome.log_loss_contribution:.6f}  "
            f"brier={outcome.brier_contribution:.6f}{flag}"
        )

    if report.already_settled:
        print(f"\nalready settled, no-op: {report.already_settled}")

    # Grouped by reason so "45 unsettled" is never an ambiguous number.
    if report.skipped:
        print(f"\nskipped: {len(report.skipped)}")
        by_reason: dict[str, list[int]] = {}
        for skip in report.skipped:
            by_reason.setdefault(skip.reason.value, []).append(skip.bout_id)
        for reason, bout_ids in sorted(by_reason.items()):
            preview = ", ".join(str(b) for b in bout_ids[:8])
            more = f" (+{len(bout_ids) - 8} more)" if len(bout_ids) > 8 else ""
            print(f"  {reason}: {len(bout_ids)} — bouts {preview}{more}")

    if report.errors:
        print(f"\n{len(report.errors)} prediction(s) could not be graded:")
        for err in report.errors:
            print(f"  prediction {err.prediction_id} (bout {err.bout_id}): {err.reason}")

    # Staleness pass runs independently of settlement — a card nobody
    # predicted can still have a broken row on it.
    stale_rows = find_stale_scheduled_bouts(engine, stale_days=args.stale_days)
    stale_records = [
        {
            "bout_id": int(row["bout_id"]),
            "event_id": int(row["event_id"]),
            "event_name": str(row["event_name"]),
            "event_date": str(row["event_date"]),
            "diagnosis": diagnose_stale(row).value,
        }
        for row in stale_rows
    ]
    suspect = [
        r
        for r in stale_records
        if r["diagnosis"] == StaleDiagnosis.SUSPECT_DUPLICATE_FIGHTER.value
    ]

    if stale_records:
        print(
            f"\n{len(stale_records)} stale scheduled bout(s) "
            f"(>{args.stale_days}d past event date):"
        )
        for record in stale_records:
            print(
                f"  bout {record['bout_id']:>6}  {record['event_date']}  "
                f"{record['event_name']}: {record['diagnosis']}"
            )
        print("  Not auto-cancelled by design (ADR-026 Decision 8).")

    if not args.write and report.settled:
        print("\nNothing written. Re-run with --write to persist.")

    out_dir = args.out.parent if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (
        out_dir / f"settled_{run_started_at:%Y%m%d_%H%M%S}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_started_at": run_started_at.isoformat(),
                "write_mode": args.write,
                "stale_days": args.stale_days,
                "n_settled": len(report.settled),
                "n_skipped": len(report.skipped),
                "n_errors": len(report.errors),
                "n_already_settled": report.already_settled,
                "settled": [
                    {
                        "prediction_id": o.prediction_id,
                        "bout_id": o.bout_id,
                        "actual_winner_id": o.actual_winner_id,
                        "correct": o.correct,
                        "log_loss_contribution": o.log_loss_contribution,
                        "brier_contribution": o.brier_contribution,
                        "corner_derivation_disagrees": o.corner_derivation_disagrees,
                    }
                    for o in report.settled
                ],
                "skipped": [
                    {
                        "prediction_id": s.prediction_id,
                        "bout_id": s.bout_id,
                        "reason": s.reason.value,
                    }
                    for s in report.skipped
                ],
                "errors": [
                    {
                        "prediction_id": e.prediction_id,
                        "bout_id": e.bout_id,
                        "reason": e.reason,
                    }
                    for e in report.errors
                ],
                "stale_bouts": stale_records,
            },
            f,
            indent=2,
        )

    print(f"\nSettled {len(report.settled)} -> {out_path}")
    # Non-zero if a prediction could not be graded, or if a stale bout
    # looks like an unreconciled duplicate fighter — both need a human.
    # Cancelled bouts, draws, and Greco lag are expected states and do
    # NOT count; alerting on them would train you to ignore the alert.
    return 1 if (report.errors or suspect) else 0


if __name__ == "__main__":
    raise SystemExit(main())