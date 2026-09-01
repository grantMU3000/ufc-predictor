"""
Computes the LIVE (since-deployment) performance block — Week 4 Thursday,
ADR-026.

DISTINCT FROM api/services/metrics.py, deliberately. That module reshapes
the FROZEN test-unlock metrics stored in model_registry.metrics: a pure
function over a JSON list, no database, one artifact, never changes. This
one aggregates prediction_results as they accumulate and grows every time a
card settles. ADR-026 requires the two to stay separate claims in the API
response; keeping them in separate modules makes that structural rather than
a convention. GET /model/performance imports from both.

Scoring rules (log loss, Brier) are reused from api.services.settlement
rather than reimplemented, so a live aggregate and a stored
prediction_results row can never silently disagree about how they were
computed.

ECE IS DELIBERATELY OMITTED HERE. A reliability curve needs real bucket
sizes to say anything; at the row counts this block will have for a while,
it would be noise wearing a number. Accuracy, log loss, Brier, and a Wilson
interval on accuracy are the honest set until there's enough settled volume
to revisit (see IDEAS.md).

DEDUPE, NOT FILTER. A bout re-predicted before the fight (odds update, bug
fix) can have more than one settled prediction_results row. Every one of
them is graded — ADR-026 Decision 2, each row was a real claim — but rolling
metrics count each BOUT once, using its latest prediction with
created_at < event_date. Skipping this would let one re-predicted fight
count twice in the accuracy denominator.

MARKET "ACCURACY" IS DERIVED, NOT STORED. Odds are only ever resolved for
odds_fighter_id == predicted_winner_id (ADR-025 Decision 2) — there is no
column for what the market thought of the fighter the model DIDN'T pick.
So market_correct is inferred: implied_prob >= 0.5 means the market agreed
with the pick (market_correct == correct); implied_prob < 0.5 means the
market favored the other fighter (market_correct == not correct). Sound
arithmetic, but it's an inference layered on stored data, not a fact Greco
or the sportsbook ever asserted directly — worth saying so wherever this
number is displayed.
"""

import math
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine, RowMapping

from api.services.odds import moneyline_to_implied_prob
from api.services.settlement import score_prediction

Who = Literal["model", "market"]


@dataclass(frozen=True)
class LiveMetricRow:
    """One scored population — mirrors models.metrics.evaluate()'s shape
    closely enough that the frozen and live blocks read as the same kind
    of claim, minus ece (see module docstring)."""

    who: Who
    n: int
    n_correct: int
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    log_loss: float
    brier: float


@dataclass(frozen=True)
class PaperRoiResult:
    """
    Flat 1-unit-per-bet paper ROI (ADR-026 Decision 6: derived at read
    time, never stored). roi is None, not 0.0, at n=0 — a flat zero would
    read as "break-even" when the truth is "no odds-covered bets exist yet."
    """

    n: int
    total_staked: float
    total_return: float
    roi: float | None


@dataclass(frozen=True)
class SettlementCounts:
    """
    The four denominators Step 7's endpoint must expose as distinct
    integers — ADR-026's explicit requirement, so "unsettled" is never
    ambiguous between "hasn't happened yet" and "excluded by design."
    """

    n_predictions: int
    n_settled: int
    n_excluded_no_result: int
    n_cancelled: int
    n_pending: int


@dataclass(frozen=True)
class LiveMetrics:
    counts: SettlementCounts
    full: list[LiveMetricRow]
    odds_covered: list[LiveMetricRow]
    paper_roi: PaperRoiResult


def wilson_interval(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    95% Wilson score interval on a proportion.

    Simple version: 9-for-12 reads as "75% accurate," but at n=12 the real
    range is roughly 47-91%. This is that range, so a small-sample week
    doesn't get read as a settled-season number.

    z=1.96 is the standard-normal 95% value (Wilson's original derivation,
    also used in Evan Miller's "How Not To Sort By Average Rating").
    """
    if n == 0:
        return (0.0, 0.0)
    phat = n_correct / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((center - margin) / denom, (center + margin) / denom)


def moneyline_payout(odds: int) -> float:
    """Net profit on a winning 1-unit bet, American odds."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def fetch_settlement_counts(engine: Engine) -> SettlementCounts:
    """
    Raw prediction-row counts, undeduped — this answers "how many claims
    are in which state," not "how many distinct fights," so a re-predicted
    bout correctly contributes more than once here even though it collapses
    to one row in the aggregate metrics below.
    """
    stmt = text("""
        SELECT
            COUNT(*) AS n_predictions,
            COUNT(*) FILTER (WHERE r.id IS NOT NULL) AS n_settled,
            COUNT(*) FILTER (
                WHERE r.id IS NULL AND b.status = 'completed' AND b.winner_id IS NULL
            ) AS n_excluded_no_result,
            COUNT(*) FILTER (
                WHERE r.id IS NULL AND b.status = 'cancelled'
            ) AS n_cancelled,
            COUNT(*) FILTER (
                WHERE r.id IS NULL AND b.status = 'scheduled'
            ) AS n_pending
        FROM predictions p
        JOIN bouts b ON b.id = p.bout_id
        LEFT JOIN prediction_results r ON r.prediction_id = p.id
    """)
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().one()
    return SettlementCounts(
        n_predictions=int(row["n_predictions"]),
        n_settled=int(row["n_settled"]),
        n_excluded_no_result=int(row["n_excluded_no_result"]),
        n_cancelled=int(row["n_cancelled"]),
        n_pending=int(row["n_pending"]),
    )


def fetch_deduped_settled(engine: Engine) -> list[RowMapping]:
    """
    One row per BOUT — its latest prediction with created_at < event_date,
    provided that prediction is settled. ADR-026 Decision 2: a re-predicted
    fight must not count twice in the accuracy denominator.

    p.created_at < e.event_date is defensive, not load-bearing: write_
    prediction() already refuses to write against a non-'scheduled' bout
    (ADR-025 Decision 5), so a post-event prediction row shouldn't exist.
    Keeping the check here means if that guarantee ever breaks, this
    aggregate fails safe instead of silently absorbing a leaked-result row.
    """
    stmt = text("""
        WITH ranked AS (
            SELECT
                p.id AS prediction_id,
                p.bout_id,
                p.odds_at_prediction_time,
                r.correct,
                r.log_loss_contribution,
                r.brier_contribution,
                ROW_NUMBER() OVER (
                    PARTITION BY p.bout_id ORDER BY p.created_at DESC, p.id DESC
                ) AS rn
            FROM predictions p
            JOIN prediction_results r ON r.prediction_id = p.id
            JOIN bouts b  ON b.id = p.bout_id
            JOIN events e ON e.id = b.event_id
            WHERE p.created_at < e.event_date
        )
        SELECT prediction_id, bout_id, odds_at_prediction_time,
               correct, log_loss_contribution, brier_contribution
        FROM ranked
        WHERE rn = 1
    """)
    with engine.connect() as conn:
        return conn.execute(stmt).mappings().all()


def _build_model_row(rows: list[RowMapping]) -> LiveMetricRow:
    """Model's own row — log loss/Brier already computed and stored at
    settlement time (settlement.score_prediction), just aggregated here.
    
    Returns None on an empty population rather than a zeroed row. A
    LiveMetricRow with log_loss=0.0 is not "no data," it is a PERFECT
    score, and it would render on the frontend as the best result the
    model has ever produced. The counts block is what explains an absent
    row; a fabricated one would misrepresent it.
    """
    n = len(rows)
    if n == 0:
        return None
    
    n_correct = sum(1 for r in rows if r["correct"])
    lo, hi = wilson_interval(n_correct, n)
    mean_ll = sum(float(r["log_loss_contribution"]) for r in rows) / n if n else 0.0
    mean_brier = sum(float(r["brier_contribution"]) for r in rows) / n if n else 0.0
    return LiveMetricRow(
        who="model",
        n=n,
        n_correct=n_correct,
        accuracy=n_correct / n if n else 0.0,
        accuracy_ci_low=lo,
        accuracy_ci_high=hi,
        log_loss=round(mean_ll, 6),
        brier=round(mean_brier, 6),
    )


def _build_market_row(rows: list[RowMapping]) -> LiveMetricRow:
    """
    Market's row over the SAME odds-covered rows, scored with the same
    score_prediction() the model itself is graded with (settlement.py) —
    nothing here is graded on a curve. See module docstring for how
    market_correct is derived from a one-sided implied probability.

    Returns None on an empty population, same reasoning as _build_model_row.
    """
    n = len(rows)
    if n == 0:
        return None

    n_correct = 0
    total_ll = 0.0
    total_brier = 0.0

    for r in rows:
        implied = moneyline_to_implied_prob(int(r["odds_at_prediction_time"]))
        market_correct = r["correct"] if implied >= 0.5 else (not r["correct"])
        p_favored = max(implied, 1.0 - implied)
        ll, brier = score_prediction(p_favored, market_correct)
        total_ll += ll
        total_brier += brier
        n_correct += int(market_correct)

    lo, hi = wilson_interval(n_correct, n)
    return LiveMetricRow(
        who="market",
        n=n,
        n_correct=n_correct,
        accuracy=n_correct / n if n else 0.0,
        accuracy_ci_low=lo,
        accuracy_ci_high=hi,
        log_loss=round(total_ll / n, 6) if n else 0.0,
        brier=round(total_brier / n, 6) if n else 0.0,
    )


def compute_paper_roi(rows: list[RowMapping]) -> PaperRoiResult:
    """
    Flat 1-unit-per-bet paper ROI over settled, odds-covered predictions.

    Flat staking on purpose (ADR-026 Decision 6) — Kelly or edge-threshold
    staking is a real decision this project hasn't made yet, and there is
    currently zero odds coverage to tune a policy against. Revisit once
    Friday's odds refresh lands and there's real data to size it on.
    """
    covered = [r for r in rows if r["odds_at_prediction_time"] is not None]
    n = len(covered)
    if n == 0:
        return PaperRoiResult(n=0, total_staked=0.0, total_return=0.0, roi=None)

    ret = sum(
        moneyline_payout(int(r["odds_at_prediction_time"])) if r["correct"] else -1.0
        for r in covered
    )
    return PaperRoiResult(n=n, total_staked=float(n), total_return=round(ret, 4), roi=round(ret / n, 6))


def compute_live_metrics(engine: Engine) -> LiveMetrics:
    """
    Everything Step 7's endpoint needs for the 'live' block, in one call.

    Both metric lists are empty until something settles. That is the
    correct representation of "no track record yet" — counts carries the
    explanation, and an empty list cannot be misread as a score.
    """
    counts = fetch_settlement_counts(engine)
    deduped = fetch_deduped_settled(engine)
    covered = [r for r in deduped if r["odds_at_prediction_time"] is not None]

    full = [row for row in (_build_model_row(deduped),) if row is not None]
    odds_covered = [
        row
        for row in (_build_model_row(covered), _build_market_row(covered))
        if row is not None
    ]

    return LiveMetrics(
        counts=counts,
        full=full,
        odds_covered=odds_covered,
        paper_roi=compute_paper_roi(deduped),
    )


if __name__ == "__main__":
    # READ ONLY — no writes possible from this module at all.
    import os

    from api.dependencies import build_engine

    pg_engine = build_engine(os.environ["DATABASE_URL"])
    live = compute_live_metrics(pg_engine)

    print(f"counts: {live.counts}\n")
    print("full:")
    for row in live.full:
        print(f"  {row}")
    print("\nodds_covered:")
    if not live.odds_covered:
        print("  (no rows — zero odds coverage)")
    for row in live.odds_covered:
        print(f"  {row}")
    print(f"\npaper_roi: {live.paper_roi}")