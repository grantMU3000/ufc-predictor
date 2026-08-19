"""
Step 4 spot-check: pulls three hand-picked fighters' Elo trajectories
across their UFC careers (train+val era only — same leakage
discipline as the rest of this week) so we can eyeball whether the
numbers match reality. Three different fighter TYPES on purpose:
Islam Makhachev (longtime dominant), Neil Magny (durable average),
Israel Adesanya (real skid) — each should produce a visibly
different SHAPE if Elo is actually tracking form, not just
accumulating fights.

Not part of the pipeline. Notebooks/scratch territory, not tested.
"""

import duckdb
import pandas as pd

from features.elo import compute_elo_ratings
from features.labels import get_completed_decided_bouts
from features.split import TEST_START

FIGHTERS_TO_CHECK = ["Islam Makhachev", "Neil Magny", "Israel Adesanya"]


def _resolve_fighter_id(con: duckdb.DuckDBPyConnection, name: str) -> int | None:
    """
    Exact match first. Falls back to a case-insensitive ILIKE search
    and prints candidates on a miss, instead of just saying "not
    found" — real_name formatting quirks are a known thing in this
    dataset (see DECISIONS.md's name-collision notes), so a debug
    script should help you find the right spelling, not leave you
    guessing.
    """
    exact = con.execute(
        "SELECT id FROM fighters WHERE real_name = $name", {"name": name}
    ).fetchone()
    if exact:
        return exact[0]

    candidates = con.execute(
        "SELECT id, real_name FROM fighters WHERE real_name ILIKE $pattern",
        {"pattern": f"%{name}%"},
    ).df()
    if candidates.empty:
        print(f"  no match at all for '{name}' — check spelling")
    else:
        print(f"  no EXACT match for '{name}'; closest candidates:")
        print(candidates.to_string(index=False))
    return None


def main():
    con = duckdb.connect()
    for table in ["fighters", "events", "bouts"]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('data/processed/{table}.parquet')"
        )

    labels = get_completed_decided_bouts(con)
    # Same rule as everywhere else this week: only bouts we're
    # currently allowed to see.
    labels = labels[
        pd.to_datetime(labels["event_date"]) < TEST_START
    ].reset_index(drop=True)

    # No k_factor passed — uses whatever default is already baked
    # into your elo.py, whatever you landed on.
    elo = compute_elo_ratings(labels)
    merged = labels.merge(elo, on="bout_id")

    fighters = con.execute("SELECT id, real_name FROM fighters").df()
    name_by_id = dict(zip(fighters["id"], fighters["real_name"]))

    # Unfold each bout into two fighter-centric rows — same idea as
    # symmetrize.py's self_/opp_ split, just for eyeballing here, not
    # for training.
    rows = []
    for bout in merged.itertuples(index=False):
        rows.append({
            "event_date": bout.event_date, "fighter_id": bout.fighter_red_id,
            "opponent_id": bout.fighter_blue_id, "elo_pre": bout.red_elo_pre,
            "won": bout.winner_id == bout.fighter_red_id,
        })
        rows.append({
            "event_date": bout.event_date, "fighter_id": bout.fighter_blue_id,
            "opponent_id": bout.fighter_red_id, "elo_pre": bout.blue_elo_pre,
            "won": bout.winner_id == bout.fighter_blue_id,
        })
    timeline = pd.DataFrame(rows).sort_values("event_date")
    timeline["opponent"] = timeline["opponent_id"].map(name_by_id)

    for name in FIGHTERS_TO_CHECK:
        fighter_id = _resolve_fighter_id(con, name)
        print(f"\n=== {name} ===")
        if fighter_id is None:
            continue
        subset = timeline[timeline["fighter_id"] == fighter_id]
        print(subset[["event_date", "opponent", "won", "elo_pre"]].to_string(index=False))


if __name__ == "__main__":
    main()
