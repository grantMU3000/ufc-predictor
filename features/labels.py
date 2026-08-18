"""
Identifies which bouts are eligible to become labeled training rows,
and carries the one piece of information store.py is deliberately
NOT allowed to touch: who won.

Simple version: store.py builds the "question" (everything you'd
know about a fight beforehand). This file holds the "answer key" —
kept in a separate file on purpose, the same way a teacher doesn't
staple the answer key to the test. If winner_id ever ended up
sitting in the same file as the feature functions, it's one bad
copy-paste away from becoming a feature itself — and a model that
can see the answer will look great and mean nothing.
"""

import duckdb
import pandas as pd


def get_completed_decided_bouts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Every bout eligible to become a training row: fought, finished,
    and with a real winner.

    Two filters, two different reasons:
      - status = 'completed'   -> drops 'scheduled' (hasn't happened
        yet) and 'cancelled' (booked, then pulled) bouts. Neither has
        a real outcome to learn from.
      - winner_id IS NOT NULL  -> drops draws and no-contests. Same
        rule tier2.py's _decided_bouts already applies per-fighter
        (ADR-003 territory) — applied here once, at the whole-
        dataset level, instead of leaving every downstream consumer
        of this list to remember to filter it out themselves.

    DQ bouts are NOT filtered out here — a disqualification still
    has a real winner_id (whoever wasn't DQ'd), so it's a legitimate
    decided bout and passes through. (tier2.py excludes DQ losses
    from ko_loss_rate/sub_loss_rate specifically because a DQ isn't
    a durability signal — that's a feature-level judgment call, not
    a "was this fight decided" question, so it doesn't belong here.)

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection pointed at the local Parquet snapshot.

    Returns
    -------
    pd.DataFrame with one row per eligible bout:
        bout_id, event_date, fighter_red_id, fighter_blue_id, winner_id
    Sorted oldest-first (matters later for the temporal train/val/
    test split — might as well hand back data in the order that
    split will need to reason about).
    """
    query = """
        SELECT
            b.id AS bout_id,
            e.event_date,
            b.fighter_red_id,
            b.fighter_blue_id,
            b.winner_id
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE b.status = 'completed'
          AND b.winner_id IS NOT NULL
        ORDER BY e.event_date ASC
    """
    return con.execute(query).df()


def raw_red_corner_win_rate(labels: pd.DataFrame) -> float:
    """
    Fraction of decided bouts the RED corner won, straight from the
    raw label data — no symmetrization needed. This is the ADR-004
    baseline number: if a trained model's implied red-corner edge
    ever approaches THIS number, that's the signal symmetrization
    failed and the model found a way to key off corner position
    instead of fighter-specific signal. Record it in LEAKAGE_LOG.md
    once computed; recompute only if the underlying bout data changes
    materially (e.g. a fresh Greco pull with many new bouts).

    Parameters
    ----------
    labels : pd.DataFrame
        Output of get_completed_decided_bouts — needs winner_id,
        fighter_red_id.

    Returns
    -------
    float — e.g. 0.5583 means red corner won 55.83% of decided bouts.
    """
    return (labels["winner_id"] == labels["fighter_red_id"]).mean()
