"""
Resolves the market price for a fighter as it stood at a given moment
— Week 4 Wednesday, ADR-025 Decision 2.

WHY THIS IS ITS OWN FILE: this is the only place in the project that
answers "what did the market think BEFORE we knew the answer." Getting
it wrong in the obvious way — reading today's odds for a fight that
already happened — is the same class of mistake as training-time
leakage, just on the market side instead of the feature side. It gets
isolated, and it gets its own tests.

WHAT IT DOES NOT DO: no de-vigging. A book's implied probabilities for
both fighters sum to more than 1 (that excess is the book's margin),
so the number stored here is a raw price, not a fair probability.
De-vigging needs both fighters' prices at once and belongs at analysis
time, in Thursday's model-vs-market comparison, where both sides of the
bout are in hand. Storing a de-vigged number here would bake one
particular de-vig method into an immutable record.

COVERAGE REALITY TODAY: odds_snapshots holds 59,972 rows across 23
books, none later than 2026-08-01. Every upcoming bout resolves to
None until Friday's refresh job runs. That is expected, not a bug —
the ledger writes NULL and says so.
"""

from dataclasses import dataclass
from datetime import datetime
from statistics import median

from sqlalchemy import Connection, text

# Below this many books, the median is one or two opinions rather than
# a consensus. Recorded, never suppressed — same principle as ADR-024
# Decision 4's coverage field: label the claim, don't gate it.
THIN_MARKET_BOOKS = 3


@dataclass(frozen=True)
class ResolvedOdds:
    """
    A consensus market price for one fighter at one point in time.

    moneyline is reconstructed FROM the median implied probability,
    not taken directly from any single book — so it is a synthetic
    consensus price, not a quote you could have actually placed.
    Thursday's ROI math should treat it as the market's estimate, not
    as a fill price.
    """

    fighter_id: int
    moneyline: int
    implied_prob: float
    n_books: int
    collected_at: datetime  # newest snapshot that fed the median

    @property
    def is_thin(self) -> bool:
        """True when too few books priced this bout to call it a consensus."""
        return self.n_books < THIN_MARKET_BOOKS


def moneyline_to_implied_prob(moneyline: int) -> float:
    """
    American odds to the probability the book is charging for.

    Two formulas because the scale is split around zero: a negative
    line says how much you must risk to win $100, a positive line says
    how much you win risking $100. -150 -> 0.600, +130 -> 0.435.

    Includes the book's margin. Both fighters' values sum to >1.
    """
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def implied_prob_to_moneyline(prob: float) -> int:
    """
    Inverse of the above, rounded to a whole American line.

    Raises on 0 or 1: those correspond to an infinite price, which no
    book can post and which would silently store a nonsense integer.
    A real median can never land there, so hitting this means the
    inputs were already corrupt.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"implied probability must be strictly in (0, 1), got {prob}")

    if prob >= 0.5:
        return round(-100.0 * prob / (1.0 - prob))
    return round(100.0 * (1.0 - prob) / prob)


def resolve_odds_at(
    conn: Connection,
    bout_id: int,
    fighter_id: int,
    as_of: datetime,
) -> ResolvedOdds | None:
    """
    The consensus price on `fighter_id` in `bout_id`, as of `as_of`.

    Simple version: ask every sportsbook what they were charging, take
    each one's most recent answer that predates the cutoff, and take
    the middle number.

    Two rules doing the real work:

    1. `collected_at <= as_of`. A snapshot collected after the
       prediction was made cannot inform that prediction. This is the
       whole point of the function.
    2. The median runs over IMPLIED PROBABILITY, then converts back to
       a moneyline (ADR-025 Decision 2). Moneyline is discontinuous —
       no legal value exists strictly between -100 and +100 — so
       median-ing an even count of raw lines can produce a price no
       book could post. Probability is continuous, so the midpoint of
       two probabilities is always meaningful.

    Returns None when no book priced this fighter before the cutoff.
    None means "no market data," never "the market said 50/50."
    """
    # DISTINCT ON gives one row per sportsbook — the latest snapshot
    # at or before the cutoff. id is the final tiebreak so the result
    # is deterministic even in the (constraint-prevented) case of two
    # rows sharing a collected_at.
    rows = conn.execute(
        text(
            """
            SELECT DISTINCT ON (sportsbook)
                   sportsbook, moneyline, implied_prob, collected_at
            FROM odds_snapshots
            WHERE bout_id = :bout_id
              AND fighter_id = :fighter_id
              AND collected_at <= :as_of
            ORDER BY sportsbook, collected_at DESC, id DESC
            """
        ),
        {"bout_id": bout_id, "fighter_id": fighter_id, "as_of": as_of},
    ).all()

    if not rows:
        return None

    probabilities = []
    for row in rows:
        # implied_prob is nullable in the schema. Falling back to the
        # moneyline rather than dropping the book keeps a real quote
        # in the consensus instead of thinning it over a missing
        # derived column.
        if row.implied_prob is not None:
            probabilities.append(float(row.implied_prob))
        else:
            probabilities.append(moneyline_to_implied_prob(int(row.moneyline)))

    consensus_prob = median(probabilities)

    return ResolvedOdds(
        fighter_id=fighter_id,
        moneyline=implied_prob_to_moneyline(consensus_prob),
        implied_prob=consensus_prob,
        n_books=len(rows),
        # The newest snapshot in the set, so the ledger records how
        # stale the consensus was — not how stale its oldest member was.
        collected_at=max(row.collected_at for row in rows),
    )


if __name__ == "__main__":
    # Smoke test against a HISTORICAL bout, since no upcoming bout has
    # odds yet. Prints rather than asserts; the real assertions live in
    # tests/ and run on synthetic snapshots, because the historical
    # data has exactly one capture per book per bout and therefore
    # cannot exercise the point-in-time branch at all.
    import os

    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()

    engine = create_engine(os.environ["DATABASE_URL"])

    with engine.connect() as conn:
        target = conn.execute(
            text(
                """
                SELECT os.bout_id, os.fighter_id, e.event_date, e.name
                FROM odds_snapshots os
                JOIN bouts b  ON b.id = os.bout_id
                JOIN events e ON e.id = b.event_id
                ORDER BY os.collected_at DESC
                LIMIT 1
                """
            )
        ).one()

        # Cutoff at the event date: what the market said going in.
        as_of = datetime.combine(target.event_date, datetime.max.time())
        resolved = resolve_odds_at(
            conn, int(target.bout_id), int(target.fighter_id), as_of
        )

    print(f"{target.name}  bout {target.bout_id}  fighter {target.fighter_id}")
    if resolved is None:
        print("  no market data before cutoff")
    else:
        print(f"  moneyline   {resolved.moneyline:+d}")
        print(f"  implied     {resolved.implied_prob:.4f}")
        print(f"  books       {resolved.n_books}"
              f"{'  [THIN]' if resolved.is_thin else ''}")
        print(f"  collected   {resolved.collected_at}")

    # Cutoff BEFORE any snapshot exists — must return None. The one
    # point-in-time check the historical data can actually support.
    with engine.connect() as conn:
        empty = resolve_odds_at(
            conn,
            int(target.bout_id),
            int(target.fighter_id),
            datetime(2020, 1, 1),
        )
    print(f"\npre-coverage cutoff returns None: {empty is None}")