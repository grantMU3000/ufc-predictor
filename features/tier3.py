"""
Tier 3 relational features — Week 3 Tuesday (docs/PLAN.md Section 2,
"Contextual & relational (the differentiators)").

Strength of schedule is the headline item here, and it exists to
answer a question Elo structurally cannot. Elo says "this fighter is
1650." It does not say HOW they got there. Two fighters can sit at
identical ratings having taken completely different roads — one
grinding out wins over other contenders, one padding a record against
short-notice debutants. Elo's math treats those as the same number,
because a win is a win and the rating already accounts for opponent
strength AT UPDATE TIME. What it doesn't retain is the *pattern*: the
model can't see that one guy has been in deep water for two years and
the other hasn't.

Simple version: two students both have a 3.8 GPA. One took AP
everything, one took the easiest classes on offer. The GPA number is
identical; you'd bet on them very differently. SoS is the course
difficulty, sitting next to the GPA.

WHY THIS LIVES HERE AND NOT IN store.py / tier2.py: every Tier 1/2
feature is f(con, fighter_id, as_of_date) reading straight from
DuckDB. SoS depends on Elo, and Elo isn't in the database — it's a
sequential pandas computation (features/elo.py) that gets recomputed
whenever K-factor params change (ADR-014). Persisting it to Postgres
just to preserve a call signature would mean a migration for a
derived number with no stable value. So SoS is computed in pandas
alongside Elo and attached the same way, in build_lgbm_matrix.py.

LEAKAGE NOTE: this is a cross-row transformation but NOT a fitted
one. Nothing here learns a parameter from the dataset — each bout's
SoS reads only that fighter's strictly-earlier bouts, exactly like
every Tier 2 rate feature. That's why it's safe to compute once over
the full train+val timeline rather than fitting on train and applying
to val (which is what style clustering WILL require, and the reason
that's a different kind of problem).
"""

import pandas as pd

# Window sizes to emit. Both get built; models/feature_deltas.py
# decides which (if either) earns a place. n=5 is the plan's default
# ("opponent Elo at time of fight, averaged over last N"); n=3 is
# here because "last 3" is closer to how matchmaking and rankings
# actually reason about a fighter's current run, and it's nearly free
# to compute alongside. Picking one on intuition would be exactly the
# kind of unearned choice ADR-014 and ADR-015 both pushed back on.
SOS_WINDOWS = (3, 5)


def build_fighter_bout_history(
    labels: pd.DataFrame, elo_ratings: pd.DataFrame
) -> pd.DataFrame:
    """
    Reshapes bout-level data into one row per FIGHTER per BOUT, with
    the opponent's pre-fight Elo attached — the shape SoS needs.

    Simple version: the bouts table has one row per fight with two
    fighters on it. To ask "what did THIS fighter's last five
    opponents look like," you need one row per fighter per fight
    instead, so you can sort a single fighter's career into a
    timeline. This does that flip: every bout becomes two rows, one
    from each corner's point of view.

    The opponent Elo recorded is their PRE-FIGHT rating from THAT
    specific past bout — not their rating today. That distinction is
    the whole leakage defense here: using an opponent's current Elo
    would mean a 2019 win over a then-unknown Islam Makhachev gets
    credited at 2024 Makhachev's rating, which is time travel.

    Parameters
    ----------
    labels : pd.DataFrame
        Decided bouts with bout_id, event_date, fighter_red_id,
        fighter_blue_id, winner_id — features.labels
        .get_completed_decided_bouts output, already filtered to the
        window the caller is allowed to see.
    elo_ratings : pd.DataFrame
        features.elo.compute_elo_ratings output — bout_id,
        red_elo_pre, blue_elo_pre. Must cover the same bout
        population as labels.

    Returns
    -------
    pd.DataFrame sorted by (fighter_id, event_date), columns:
        fighter_id, bout_id, event_date, opponent_id, opponent_elo_pre

    Raises
    ------
    ValueError if any bout in labels has no matching Elo row — that
    would silently produce NaN SoS for real fights rather than
    failing where the mismatch was introduced.
    """
    merged = labels.merge(elo_ratings, on="bout_id", how="left")

    missing = merged["red_elo_pre"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} bout(s) in labels had no matching Elo row — "
            f"check both were built from the same decided-bout "
            f"population and date window."
        )

    # Each bout emits two rows: once from red's perspective (opponent
    # = blue), once from blue's (opponent = red).
    red_view = pd.DataFrame(
        {
            "fighter_id": merged["fighter_red_id"],
            "bout_id": merged["bout_id"],
            "event_date": merged["event_date"],
            "opponent_id": merged["fighter_blue_id"],
            "opponent_elo_pre": merged["blue_elo_pre"],
        }
    )
    blue_view = pd.DataFrame(
        {
            "fighter_id": merged["fighter_blue_id"],
            "bout_id": merged["bout_id"],
            "event_date": merged["event_date"],
            "opponent_id": merged["fighter_red_id"],
            "opponent_elo_pre": merged["red_elo_pre"],
        }
    )

    history = pd.concat([red_view, blue_view], ignore_index=True)
    history["event_date"] = pd.to_datetime(history["event_date"])

    # Sort is load-bearing, not cosmetic: the rolling window below
    # assumes chronological order within each fighter. An unsorted
    # frame wouldn't error — it would quietly average the wrong five
    # fights.
    return history.sort_values(["fighter_id", "event_date", "bout_id"]).reset_index(
        drop=True
    )


def strength_of_schedule(history: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    For every (fighter, bout), the average pre-fight Elo of that
    fighter's last `n` opponents BEFORE this bout.

    Simple version: walk down one fighter's career in order. At each
    fight, look back at the previous n fights and average how good
    those opponents were at the time. That number is this fighter's
    strength of schedule going into the current fight.

    THE .shift(1) IS THE LEAKAGE DEFENSE. Without it, the rolling
    window would include the CURRENT bout's opponent — meaning the
    feature would encode who the fighter is fighting right now, which
    is information about the fight being predicted, not about the
    fighter's history. Same discipline as compute_elo_ratings
    recording the pre-fight rating before applying the update.

    min_periods=1 means a fighter with only 2 prior fights gets the
    average of those 2, rather than NaN until they hit n. That's the
    honest read — "we know a little" beats "we know nothing" — and
    it matches how Tier 2 rate features already handle thin history.
    True debutants get NaN, correctly: no prior fights means no
    schedule to be strong or weak.

    Parameters
    ----------
    history : pd.DataFrame
        build_fighter_bout_history output. MUST already be sorted by
        (fighter_id, event_date) — this function assumes it and does
        not re-sort, so a caller that reorders the frame in between
        would get silently wrong answers.
    n : int
        Window size — how many prior opponents to average over.

    Returns
    -------
    pd.DataFrame: fighter_id, bout_id, sos_last_{n}
    """
    col = f"sos_last_{n}"

    result = history[["fighter_id", "bout_id"]].copy()
    result[col] = (
        history.groupby("fighter_id")["opponent_elo_pre"]
        .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    )
    return result


def build_sos_by_bout(
    labels: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    windows: tuple[int, ...] = SOS_WINDOWS,
) -> pd.DataFrame:
    """
    Full SoS pipeline, returning bout-level red_/blue_ columns — the
    same shape features.elo.compute_elo_ratings hands back, so
    build_lgbm_matrix.py can attach it with the identical
    source_corner relabeling it already does for Elo.

    Simple version: does the fighter-timeline math, then folds the
    answers back onto one row per bout with a red column and a blue
    column, ready to merge.

    Parameters
    ----------
    labels, elo_ratings : pd.DataFrame
        See build_fighter_bout_history.
    windows : tuple[int, ...]
        Window sizes to compute. Each produces its own
        red_sos_last_{n} / blue_sos_last_{n} pair.

    Returns
    -------
    pd.DataFrame — bout_id, plus red_sos_last_{n}/blue_sos_last_{n}
    for each n in windows. One row per bout in labels.
    """
    history = build_fighter_bout_history(labels, elo_ratings)

    # Which fighter was in which corner, per bout — needed to fold
    # the per-fighter answers back into bout-level red/blue columns.
    corners = labels[["bout_id", "fighter_red_id", "fighter_blue_id"]]

    out = corners[["bout_id"]].copy()

    for n in windows:
        sos = strength_of_schedule(history, n=n)
        col = f"sos_last_{n}"

        red = corners.merge(
            sos, left_on=["bout_id", "fighter_red_id"],
            right_on=["bout_id", "fighter_id"], how="left",
        )[["bout_id", col]].rename(columns={col: f"red_{col}"})

        blue = corners.merge(
            sos, left_on=["bout_id", "fighter_blue_id"],
            right_on=["bout_id", "fighter_id"], how="left",
        )[["bout_id", col]].rename(columns={col: f"blue_{col}"})

        out = out.merge(red, on="bout_id").merge(blue, on="bout_id")

    if len(out) != len(labels):
        raise ValueError(
            f"row count changed ({len(labels)} -> {len(out)}) — check "
            f"for duplicate bout_id or a fighter appearing twice in "
            f"the same bout."
        )

    return out