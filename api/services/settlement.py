"""
Scores committed predictions against real results — Week 4 Thursday, ADR-026.

The ledger's write half (api/services/ledger.py) makes a claim permanent.
This is the read-and-grade half: it walks unsettled predictions, finds the
actual winner, and writes one prediction_results row per graded claim.

Three things this file refuses to do, each for a reason:

  1. IT NEVER INFERS THE PICK FROM CORNER POSITION. `correct` is
     predicted_winner_id == bouts.winner_id, full stop (ADR-026 Decision 1).
     ADR-013 established that corners can flip between the pre-fight
     Wikipedia row and the post-fight Greco row, so predicted_prob_red > 0.5
     is not a safe way to recover who was picked.

  2. IT NEVER MUTATES A PREDICTION. Settlement writes to prediction_results
     only. The ADR-025 trigger would reject an UPDATE anyway; this is the
     application-side half of the same rule.

  3. IT NEVER FILTERS SKIPS OUT IN SQL. Classification happens in Python so
     every unsettled prediction is explained — cancelled, not yet fought, or
     a draw. A WHERE clause would make "not settled" indistinguishable from
     "never looked at," which is exactly the silent failure this job exists
     to prevent.

Reads that serve the API live in api/services/queries.py. The query here is
job-internal (nothing in the HTTP layer needs "unsettled predictions"), so it
stays local rather than widening that module's surface.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy import Connection, text
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import SQLAlchemyError

# ADR-026 Decision 5. Matches the clip in models/test_unlock.py's
# bootstrap_logloss_ci. Only ever binds when a stored probability rounds
# to exactly 0.0000 or 1.0000 at Numeric(5,4), but it must match the
# offline harness or frozen and live log loss are not comparable numbers.
LOG_LOSS_EPS = 1e-15

# log_loss_contribution is Numeric(8,6) -> max 99.999999.
# -ln(1e-15) = 34.538776, so the clip above can never overflow the column.
_CONTRIBUTION_DP = 6


class SkipReason(str, Enum):
    """Why an unsettled prediction stayed unsettled. All are non-errors."""

    NOT_YET_FOUGHT = "not_yet_fought"
    BOUT_CANCELLED = "bout_cancelled"
    NO_WINNER_RECORDED = "no_winner_recorded"  # draw or no-contest


class ErrorReason(str, Enum):
    """Conditions that need human eyes before anything is written."""

    WINNER_NOT_A_PARTICIPANT = "winner_not_a_participant"
    PREDICTED_FIGHTER_NOT_IN_BOUT = "predicted_fighter_not_in_bout"


class StaleDiagnosis(str, Enum):
    """Cause assigned to a bout still 'scheduled' after its event date.

    Mirrors the four cases in find_stale_scheduled_bouts's docstring.
    SUSPECT_DUPLICATE_FIGHTER is the only one that needs a human — see
    diagnose_stale.
    """

    NOT_YET_INGESTED = "greco_has_not_ingested_this_card"
    LIKELY_CANCELLED = "likely_cancelled_neither_fighter_competed"
    LIKELY_CANCELLED_REBOOKED = "likely_cancelled_one_fighter_rebooked"
    SUSPECT_DUPLICATE_FIGHTER = "SUSPECT_DUPLICATE_FIGHTER_both_fighters_competed"


@dataclass(frozen=True)
class Settleable:
    """A prediction that can be graded right now, with its scores computed."""

    prediction_id: int
    bout_id: int
    actual_winner_id: int
    correct: bool
    log_loss_contribution: float
    brier_contribution: float
    corner_derivation_disagrees: bool


@dataclass(frozen=True)
class Skipped:
    """A prediction deliberately left unsettled, with the reason recorded."""

    prediction_id: int
    bout_id: int
    reason: SkipReason


@dataclass(frozen=True)
class Errored:
    """A prediction that could not be graded safely."""

    prediction_id: int
    bout_id: int
    reason: str


@dataclass
class SettlementReport:
    """
    Three buckets, deliberately kept separate.

    A skip is not an error. A cancelled bout will never settle, and Friday's
    cron must not treat that as a failure — same three-bucket discipline as
    score_upcoming.py's --write path (ADR-025).
    """

    settled: list[Settleable] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    errors: list[Errored] = field(default_factory=list)
    already_settled: int = 0

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


def score_prediction(prob_predicted_winner: float, correct: bool) -> tuple[float, float]:
    """
    Log loss and Brier score for one graded prediction.

    Simple version: both numbers ask "how confident were you, and were you
    right?" Log loss punishes confident wrongness harshly — being 95% sure
    and wrong costs about 3.0, being 55% sure and wrong costs about 0.8.
    Brier is gentler and bounded at 1.0. Reporting both is standard because
    log loss is the training objective and Brier is easier to read.

    Both are computed from the PREDICTED WINNER's probability, never the red
    corner's. The two are algebraically identical when corners are stable
    (flipping p -> 1-p and y -> 1-y leaves both scores unchanged), but only
    this form survives a post-prediction corner flip (ADR-013).

    Parameters
    ----------
    prob_predicted_winner : float
        The model's probability on the fighter it actually picked. Always
        >= 0.5 by construction.
    correct : bool
        Did that fighter win.
    """
    p = min(max(prob_predicted_winner, LOG_LOSS_EPS), 1.0 - LOG_LOSS_EPS)
    y = 1.0 if correct else 0.0

    log_loss = -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    brier = (p - y) ** 2

    return round(log_loss, _CONTRIBUTION_DP), round(brier, _CONTRIBUTION_DP)


def fetch_unsettled(engine: Engine) -> Sequence[RowMapping]:
    """
    Every prediction with no prediction_results row, whatever its bout's
    state. Classification happens in Python, not here — see the module
    docstring's point 3.

    Ordered oldest-event-first so a dry-run's output reads as a timeline.
    """
    stmt = text("""
        SELECT
            p.id                  AS prediction_id,
            p.bout_id             AS bout_id,
            p.predicted_prob_red  AS predicted_prob_red,
            p.predicted_winner_id AS predicted_winner_id,
            b.status              AS bout_status,
            b.winner_id           AS actual_winner_id,
            b.fighter_red_id      AS fighter_red_id,
            b.fighter_blue_id     AS fighter_blue_id,
            e.event_date          AS event_date,
            e.name                AS event_name
        FROM predictions p
        JOIN bouts b  ON b.id = p.bout_id
        JOIN events e ON e.id = b.event_id
        LEFT JOIN prediction_results r ON r.prediction_id = p.id
        WHERE r.id IS NULL
        ORDER BY e.event_date ASC, p.id ASC
    """)
    with engine.connect() as conn:
        return conn.execute(stmt).mappings().all()


def classify(row: RowMapping) -> Settleable | Skipped | Errored:
    """
    Decide what to do with one unsettled prediction. Pure — no database, no
    side effects, so every branch below is unit-testable without a fixture.

    Order matters. Status is checked before winner_id because a cancelled
    bout also has winner_id NULL, and reporting it as "no winner recorded"
    would make a scratched fight look like a draw.
    """
    prediction_id = int(row["prediction_id"])
    bout_id = int(row["bout_id"])
    status = str(row["bout_status"])

    if status == "cancelled":
        return Skipped(prediction_id, bout_id, SkipReason.BOUT_CANCELLED)
    if status != "completed":
        return Skipped(prediction_id, bout_id, SkipReason.NOT_YET_FOUGHT)

    # Completed with no winner: draw or no-contest. ADR-026 Decision 3 —
    # prediction_results.actual_winner_id is NOT NULL, so this outcome is
    # structurally unstorable. It stays unsettled and gets counted apart
    # from "hasn't happened yet."
    if row["actual_winner_id"] is None:
        return Skipped(prediction_id, bout_id, SkipReason.NO_WINNER_RECORDED)

    actual_winner_id = int(row["actual_winner_id"])
    predicted_winner_id = int(row["predicted_winner_id"])
    red_id = int(row["fighter_red_id"])
    blue_id = int(row["fighter_blue_id"])
    participants = {red_id, blue_id}

    if actual_winner_id not in participants:
        return Errored(
            prediction_id,
            bout_id,
            f"{ErrorReason.WINNER_NOT_A_PARTICIPANT.value}: winner "
            f"{actual_winner_id} is not {red_id} or {blue_id}",
        )

    # The fighter we picked is no longer in this bout at all — a full
    # replacement that slipped past load_bout()'s swap logic, or an
    # unreconciled duplicate fighter row. Grading this False would record a
    # loss the model never actually took.
    if predicted_winner_id not in participants:
        return Errored(
            prediction_id,
            bout_id,
            f"{ErrorReason.PREDICTED_FIGHTER_NOT_IN_BOUT.value}: picked "
            f"{predicted_winner_id}, bout is {red_id} vs {blue_id}",
        )

    correct = predicted_winner_id == actual_winner_id

    p_red = float(row["predicted_prob_red"])
    p_winner = max(p_red, 1.0 - p_red)

    # Cross-check: if corners are stable AND predicted_winner_id is the
    # model's argmax, the corner-anchored derivation must agree. Disagreement
    # means one of those two assumptions broke. p_winner above is still the
    # defensible value, so this warns rather than blocks.
    p_winner_by_corner = p_red if predicted_winner_id == red_id else 1.0 - p_red
    disagrees = abs(p_winner - p_winner_by_corner) > 1e-9

    log_loss, brier = score_prediction(p_winner, correct)

    return Settleable(
        prediction_id=prediction_id,
        bout_id=bout_id,
        actual_winner_id=actual_winner_id,
        correct=correct,
        log_loss_contribution=log_loss,
        brier_contribution=brier,
        corner_derivation_disagrees=disagrees,
    )


def insert_result(conn: Connection, outcome: Settleable) -> bool:
    """
    Write one prediction_results row. Returns False if one already existed.

    ON CONFLICT DO NOTHING against uq_prediction_results_prediction_id is
    what makes this job safe to re-run: settlement is a cron job, and a cron
    job that double-counts on a retry produces a track record that inflates
    with every infrastructure hiccup.

    Takes a Connection rather than an Engine so the caller owns the
    transaction boundary — same contract as ledger.write_prediction, and the
    reason one bad row can't roll back a whole card.
    """
    row = conn.execute(
        text("""
            INSERT INTO prediction_results (
                prediction_id, actual_winner_id, correct,
                log_loss_contribution, brier_contribution
            ) VALUES (
                :prediction_id, :actual_winner_id, :correct,
                :log_loss_contribution, :brier_contribution
            )
            ON CONFLICT ON CONSTRAINT uq_prediction_results_prediction_id
            DO NOTHING
            RETURNING id
        """),
        {
            "prediction_id": outcome.prediction_id,
            "actual_winner_id": outcome.actual_winner_id,
            "correct": outcome.correct,
            "log_loss_contribution": outcome.log_loss_contribution,
            "brier_contribution": outcome.brier_contribution,
        },
    ).one_or_none()
    return row is not None


def settle_predictions(engine: Engine, *, write: bool = False) -> SettlementReport:
    """
    Classify every unsettled prediction and, when write=True, persist the
    settleable ones.

    Dry-run by default, matching score_upcoming.py. The classification pass
    is identical in both modes, so a dry run is a genuine preview rather
    than a different code path that happens to print something similar.

    One transaction per prediction. A constraint violation on bout 34500
    must not discard the twelve rows already graded on the same card.
    """
    report = SettlementReport()

    for row in fetch_unsettled(engine):
        decision = classify(row)

        if isinstance(decision, Skipped):
            report.skipped.append(decision)
            continue
        if isinstance(decision, Errored):
            report.errors.append(decision)
            continue

        if not write:
            report.settled.append(decision)
            continue

        try:
            with engine.begin() as conn:
                inserted = insert_result(conn, decision)
        except SQLAlchemyError as exc:
            report.errors.append(
                Errored(decision.prediction_id, decision.bout_id, f"insert failed: {exc}")
            )
            continue

        if inserted:
            report.settled.append(decision)
        else:
            report.already_settled += 1

    return report


def find_stale_scheduled_bouts(engine: Engine, stale_days: int = 2) -> Sequence[RowMapping]:
    """
    Bouts still marked 'scheduled' on an event that already happened.

    This exists because of bout 34419 (UFC FN 2026-08-29): the Wikipedia
    pipeline never re-scraped between a booking change and the event, so
    load_bout()'s swap logic never ran and the row sat 'scheduled' forever
    with nothing to notice it. Detection only — never auto-cancel
    (ADR-026 Decision 8).

    The three EXISTS flags separate three very different causes:
      card_has_results False -> Greco simply hasn't ingested this card yet.
      neither fighter fought -> genuine cancellation.
      exactly one fought     -> cancellation plus a rebooking (34419's case).
      BOTH fighters fought   -> the dangerous one. Both are in completed
                                bouts on this card, so this row should have
                                been claimed and wasn't: almost certainly an
                                unreconciled duplicate fighter (ADR-013's
                                stub-fighter failure mode). Auto-cancelling
                                would bury exactly this signal.
    """
    stmt = text("""
        SELECT
            b.id             AS bout_id,
            e.id             AS event_id,
            e.name           AS event_name,
            e.event_date     AS event_date,
            b.fighter_red_id AS fighter_red_id,
            b.fighter_blue_id AS fighter_blue_id,
            EXISTS (
                SELECT 1 FROM bouts c
                WHERE c.event_id = e.id AND c.status = 'completed'
            ) AS card_has_results,
            EXISTS (
                SELECT 1 FROM bouts c
                WHERE c.event_id = e.id AND c.status = 'completed'
                  AND b.fighter_red_id IN (c.fighter_red_id, c.fighter_blue_id)
            ) AS red_fought,
            EXISTS (
                SELECT 1 FROM bouts c
                WHERE c.event_id = e.id AND c.status = 'completed'
                  AND b.fighter_blue_id IN (c.fighter_red_id, c.fighter_blue_id)
            ) AS blue_fought
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE b.status = 'scheduled'
          AND e.event_date < CURRENT_DATE - CAST(:stale_days AS integer)
        ORDER BY e.event_date ASC, b.id ASC
    """)
    with engine.connect() as conn:
        return conn.execute(stmt, {"stale_days": stale_days}).mappings().all()


def diagnose_stale(row: RowMapping) -> StaleDiagnosis:
    """Human-readable cause for one stale scheduled bout. Pure."""
    if not row["card_has_results"]:
        return StaleDiagnosis.NOT_YET_INGESTED
    fought = int(bool(row["red_fought"])) + int(bool(row["blue_fought"]))
    if fought == 0:
        return StaleDiagnosis.LIKELY_CANCELLED
    if fought == 1:
        return StaleDiagnosis.LIKELY_CANCELLED_REBOOKED
    return StaleDiagnosis.SUSPECT_DUPLICATE_FIGHTER


if __name__ == "__main__":
    # SMOKE TEST — READ ONLY. Never passes write=True. Running this file
    # must never change the settlement record.
    import os

    from api.dependencies import build_engine

    pg_engine = build_engine(os.environ["DATABASE_URL"])

    result = settle_predictions(pg_engine, write=False)
    print(
        f"would settle: {len(result.settled)}  "
        f"skipped: {len(result.skipped)}  errors: {len(result.errors)}"
    )
    for stale in find_stale_scheduled_bouts(pg_engine):
        print(f"  stale bout {stale['bout_id']} ({stale['event_name']}): "
              f"{diagnose_stale(stale).value}")