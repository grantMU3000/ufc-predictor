"""
Converts store.py's one-row-per-bout (red_/blue_) output into the
self_/opp_ perspective needed to defeat corner-ordering leakage
(docs/PLAN.md Section 0.2, ADR-004). Deliberately separate from
store.py and features/labels.py — assembly, labeling, and
symmetrization are three different jobs, and keeping them in three
different files means a bug in one can't hide inside another.
"""

# stance_matchup's two fields are inherently "self vs opp" already —
# store.py computes them once per corner, each from that corner's own
# point of view (see tier1.py's stance_matchup docstring). Symmetrizing
# them the same way as every other feature would produce an opp_
# version that's just a reworded copy of the self_ version (e.g.
# self_stance_matchup_descriptive="Orthodox vs Southpaw" and
# opp_stance_matchup_descriptive="Southpaw vs Orthodox" — same fact,
# same information, nothing new). Dropped here rather than fed to the
# model twice under two different column names.

import duckdb
import pandas as pd

from features.labels import get_completed_decided_bouts
from features.store import build_feature_row

_STANCE_MATCHUP_SUFFIXES = {"stance_matchup_descriptive", "is_open_stance_matchup"}


def _symmetrize_row(
    row: dict, fighter_red_id: int, fighter_blue_id: int, self_fighter_id: int
) -> dict:
    """
    Reshapes one red_/blue_-prefixed row from store.build_feature_row
    into a single self_/opp_-prefixed row, told from self_fighter_id's
    point of view.

    Simple version: the input row is a fight card with two labeled
    columns, "RED" and "BLUE." This function doesn't change a single
    number on the card — it just relabels one column "ME" and the
    other "THEM," based on which fighter you tell it to look at it
    through the eyes of. Called once per fighter per bout (twice
    total), it's what turns one bout into the two training rows this
    whole file exists to produce.

    Three cases per key in `row`:
      1. Key belongs to self_fighter_id's corner (red_ or blue_,
         whichever matches self_fighter_id) -> renamed to self_.
      2. Key belongs to the other corner -> renamed to opp_, UNLESS
         it's one of stance_matchup's two fields, which are dropped
         as a redundant mirror of self_'s own version (see above).
      3. Key has no red_/blue_ prefix at all (bout_id, event_date,
         weight_class, is_title_fight, scheduled_rounds) -> a fact
         about the matchup itself, not either fighter — copied through
         unchanged.

    Does NOT compute self_won. That's deliberate: this function's only
    job is reshaping red/blue into self/opp. Attaching the label is
    the orchestration step's job (features/labels.py already holds
    winner_id; this function should never need to know it), so a bug
    in "who won" and a bug in "whose column is whose" can never be
    confused for each other while debugging.

    Parameters
    ----------
    row : dict
        One row of store.build_feature_row's output.
    fighter_red_id : int
    fighter_blue_id : int
        The two corners for this bout — same source as
        features/labels.py's get_completed_decided_bouts.
    self_fighter_id : int
        Must equal fighter_red_id or fighter_blue_id. Whichever one
        it is decides which original corner becomes "self" and which
        becomes "opp" for this call.

    Returns
    -------
    dict — same bout, self_/opp_-prefixed, plus self_fighter_id and
    opp_fighter_id added explicitly (not present in store.py's
    output, needed downstream for Tier 3 Elo lookups and for tracing
    a specific row back to a specific fighter during debugging).

    Raises
    ------
    ValueError if self_fighter_id is neither corner of this bout —
    fails loudly here rather than silently producing a row with no
    self_ columns at all.
    """
    if self_fighter_id == fighter_red_id:
        self_source, opp_source = "red_", "blue_"
        opp_fighter_id = fighter_blue_id
    elif self_fighter_id == fighter_blue_id:
        self_source, opp_source = "blue_", "red_"
        opp_fighter_id = fighter_red_id
    else:
        raise ValueError(
            f"self_fighter_id {self_fighter_id} is neither corner "
            f"(red={fighter_red_id}, blue={fighter_blue_id}) of this bout"
        )

    symmetrized = {}
    for key, value in row.items():
        if key.startswith(self_source):
            suffix = key[len(self_source):]
            symmetrized[f"self_{suffix}"] = value
        elif key.startswith(opp_source):
            suffix = key[len(opp_source):]
            if suffix in _STANCE_MATCHUP_SUFFIXES:
                continue  # dropped — redundant mirror, see module note
            symmetrized[f"opp_{suffix}"] = value
        else:
            symmetrized[key] = value  # bout-level fact, no corner

    symmetrized["self_fighter_id"] = self_fighter_id
    symmetrized["opp_fighter_id"] = opp_fighter_id
    symmetrized["source_corner"] = "red" if self_source == "red_" else "blue"
    return symmetrized

def build_symmetrized_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    The full training matrix: every decided bout, turned into two
    self_/opp_ rows with flipped labels.

    Simple version: features/labels.py hands over the guest list
    (which bouts count, who won each one). This function walks that
    list one bout at a time, builds the raw red_/blue_ card for it
    (store.py — the expensive part, since it's ~32 features x 2
    corners of real querying), then photocopies that card twice
    through _symmetrize_row — once as each fighter's "self." The
    label (self_won) is attached HERE, not inside _symmetrize_row,
    because comparing self_fighter_id to winner_id is the one place
    in this whole file that's allowed to know who won — keeping that
    logic out of _symmetrize_row means a bug in "who won" can never
    get confused with a bug in "whose column is whose."

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection pointed at the local Parquet snapshot.

    Returns
    -------
    pd.DataFrame — 2 rows per eligible bout. self_won is True for
    exactly one of each pair, False for the other; never both,
    never neither, since every decided bout has exactly one winner.

    Performance note: builds every row in Python (a list of dicts),
    then does ONE pd.DataFrame(...) call at the end — not an
    incremental concat per bout. Repeated concat is O(n^2) with
    pandas (it copies the whole growing DataFrame every time), so at
    ~8,456 bouts this matters. Runs once, all bouts, no incremental
    saving mid-run — if it dies partway through you restart from
    scratch, which is fine for a build step you'll run a handful of
    times, not something running unattended in production.
    """
    labels = get_completed_decided_bouts(con)

    symmetrized_rows = []
    for _, bout in labels.iterrows():
        bout_id = int(bout["bout_id"])
        fighter_red_id = int(bout["fighter_red_id"])
        fighter_blue_id = int(bout["fighter_blue_id"])
        winner_id = int(bout["winner_id"])

        row = build_feature_row(con, bout_id)

        for self_fighter_id in (fighter_red_id, fighter_blue_id):
            symmetrized = _symmetrize_row(
                row, fighter_red_id, fighter_blue_id, self_fighter_id
            )
            symmetrized["self_won"] = self_fighter_id == winner_id
            symmetrized_rows.append(symmetrized)

    return pd.DataFrame(symmetrized_rows)