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

import duckdb
import pandas as pd
import re

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

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds self_/opp_ multiplicative interactions to an already-
    symmetrized, Elo/SoS-attached split. Pure row-wise arithmetic —
    no DB access, no merge, because both inputs are already present
    and already leakage-safe.

    self_layoff_x_age: days-since-last-fight times age. Trees don't
    automatically find multiplicative interactions — a 38-year-old
    off 18 months and a 26-year-old off 18 months carry the same raw
    layoff number even though they're very different bets. This makes
    that difference explicit instead of hoping two separate diff_
    columns imply it.

    self_age_x_experience: age times total_ufc_fights — the mileage
    signal flagged during the ADR-015 21+ fight bucket discussion.
    NOT built because that bucket proved anything — the permutation
    check showed it didn't, it was noise. Built because the underlying
    idea (a 23-fight veteran and a 23-year-old prodigy with 23 fights
    are different, even at the same age or the same fight count alone)
    is sound on its own, and it's cheap enough to just test rather
    than argue about. Step 6's CV delta is the actual verdict.

    Requires self_/opp_ age, days_since_last_fight, total_ufc_fights
    already present — call this AFTER attach_by_corner, on the full
    symmetrized frame.
    """
    out = df.copy()
    for prefix in ("self", "opp"):
        out[f"{prefix}_layoff_x_age"] = (
            out[f"{prefix}_days_since_last_fight"] * out[f"{prefix}_age"]
        )
        out[f"{prefix}_age_x_experience"] = (
            out[f"{prefix}_age"] * out[f"{prefix}_total_ufc_fights"]
        )
    return out

def recent_damage_absorbed(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date
) -> float | None:
    """
    Total significant strikes ABSORBED by fighter_id in the 24 months
    strictly before as_of_date. Not the lifetime-cumulative version
    already in Tier 2 — that can't tell "took a lot of damage early,
    untouchable for 3 years since" apart from "just survived three
    wars back to back." This can.

    "Absorbed" means the OPPONENT's landed significant strikes in
    that bout — bout_stats rows are keyed by who did the striking, so
    this joins to the other corner in the same bout to find what came
    back the other way.

    Point-in-time rule: event_date < as_of_date, strict — same as
    every Tier 1/2 function. The bout on as_of_date is the fight
    being predicted, not history.

    Returns
    -------
    float | None — None means no bouts in the 24-month window (a
    debutant, or a layoff longer than 2 years), NOT zero damage.
    Don't impute this to 0 downstream — same NaN discipline as every
    other Tier 2 rate feature.
    """
    query = """
        WITH damage AS (
            SELECT
                bs.bout_id,
                e.event_date,
                CASE WHEN b.fighter_red_id = bs.fighter_id
                     THEN b.fighter_blue_id
                     ELSE b.fighter_red_id
                END AS absorbing_fighter_id,
                bs.sig_strikes_landed AS strikes_against
            FROM bout_stats bs
            JOIN bouts b ON b.id = bs.bout_id
            JOIN events e ON e.id = b.event_id
        )
        SELECT SUM(strikes_against)
        FROM damage
        WHERE absorbing_fighter_id = $fighter_id
          AND event_date < $as_of_date
          AND event_date >= $as_of_date - INTERVAL 24 MONTH
    """
    result = con.execute(
        query, {"fighter_id": fighter_id, "as_of_date": as_of_date}
    ).fetchone()
    return float(result[0]) if result and result[0] is not None else None


def build_recent_damage_by_bout(
    con: duckdb.DuckDBPyConnection, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Runs recent_damage_absorbed for every fighter in every bout,
    producing the same red_/blue_ bout-level shape build_sos_by_bout
    hands back — ready for attach_by_corner.

    Same per-bout-per-corner loop store.py's build_feature_row
    already uses for every Tier 1/2 feature. Kept separate from
    store.py (not added to FIGHTER_FEATURES) because that list is
    already materialized into train.parquet/val.parquet — adding to
    it means re-running the full Week 2 materialize step, not just
    today. Follows the same attach-after-the-fact pattern as Elo and
    SoS instead.
    """
    rows = []
    for bout in labels.itertuples(index=False):
        rows.append({
            "bout_id": bout.bout_id,
            "red_recent_damage_24mo": recent_damage_absorbed(
                con, bout.fighter_red_id, bout.event_date
            ),
            "blue_recent_damage_24mo": recent_damage_absorbed(
                con, bout.fighter_blue_id, bout.event_date
            ),
        })
    return pd.DataFrame(rows)

# Ordered lightest -> heaviest, men's and women's as SEPARATE ladders.
# Men's Flyweight (125) and Women's Flyweight (125) are the same
# poundage but not the same division — a fighter never "moves"
# between them, so putting them on one ladder would be meaningless.
# Two ladders, and a cross-ladder comparison returns None.
MENS_LADDER = {
    "Flyweight": 0,
    "Bantamweight": 1,
    "Featherweight": 2,
    "Lightweight": 3,
    "Welterweight": 4,
    "Middleweight": 5,
    "Light Heavyweight": 6,
    "Heavyweight": 7,
}

WOMENS_LADDER = {
    "Women's Strawweight": 0,
    "Women's Flyweight": 1,
    "Women's Bantamweight": 2,
    "Women's Featherweight": 3,
}

# Divisions with no ladder position by nature — a catchweight has no
# official poundage in this data (the column carries no number), and
# Open/Super Heavyweight are historical formats with no modern
# equivalent. These map to None, NOT to a guessed position.
UNLADDERED = {"Catch Weight", "Catchweight", "Open Weight", "Super Heavyweight"}


def normalize_weight_class(raw: str | None) -> str | None:
    """
    Collapses the ~110 distinct weight_class strings in `bouts` down
    to a canonical division name.

    Simple version: the raw column is a mess of the same handful of
    divisions written a dozen different ways — "Lightweight Bout,"
    "Lightweight," "UFC Lightweight Title Bout," "UFC Interim
    Lightweight Title Bout," and "Ultimate Fighter 22 Lightweight
    Tournament Title Bout" are all just Lightweight. This strips the
    decoration and returns the division.

    WHY A NORMALIZER AND NOT A PLAIN DICT LOOKUP: a dict keyed on
    "Lightweight" silently returns None for all four of the other
    spellings above. That failure is invisible — no error, just a
    NaN where a real value belonged, on exactly the highest-signal
    bouts in the dataset (title fights). Normalizing first makes the
    unmatched set small enough to actually eyeball.

    Order of matching matters: "Women's Bantamweight" must be tested
    BEFORE "Bantamweight", and "Light Heavyweight" before
    "Heavyweight", or the shorter name matches first and mislabels
    the division. Longest-first sorting handles this without needing
    the dicts themselves ordered carefully.

    Returns
    -------
    str | None — a canonical division name (a key of MENS_LADDER,
    WOMENS_LADDER, or a member of UNLADDERED), or None if nothing
    matched. None should be rare; run the __main__ audit below to
    see exactly what falls through before trusting this.
    """
    if raw is None:
        return None

    text = str(raw)

    candidates = list(WOMENS_LADDER) + list(MENS_LADDER) + list(UNLADDERED)
    # Longest first: "Women's Bantamweight" wins over "Bantamweight",
    # "Light Heavyweight" over "Heavyweight", "Super Heavyweight"
    # over "Heavyweight".
    for name in sorted(candidates, key=len, reverse=True):
        if re.search(re.escape(name), text, flags=re.IGNORECASE):
            return name

    return None


def _ladder_position(division: str | None) -> tuple[str, int] | None:
    """
    Maps a canonical division to (ladder_name, position), or None for
    unladdered/unmatched divisions. The ladder_name is returned so
    the caller can refuse to compare across ladders.
    """
    if division is None or division in UNLADDERED:
        return None
    if division in MENS_LADDER:
        return ("mens", MENS_LADDER[division])
    if division in WOMENS_LADDER:
        return ("womens", WOMENS_LADDER[division])
    return None


def weight_class_change(
    con: duckdb.DuckDBPyConnection, fighter_id: int, as_of_date
) -> int | None:
    """
    Did this fighter move up, down, or stay put since their last
    bout? -1 = moved down, 0 = same division, +1 = moved up.

    Simple version: a guy cutting to a new, lower division is a
    different bet than the same guy moving up to face bigger
    opponents — the physical advantages flip. This captures the
    direction of that move.

    Deliberately just DIRECTION, not magnitude. A two-division jump
    could be encoded as +2, but there are few enough of those that
    the model would be fitting a rule off a handful of bouts, and
    +1/-1 already carries the part that matters. Same "don't build a
    rule off 2 bouts" instinct as min_child_samples in the tuned
    params.

    Returns None (not 0) when: no prior bout (debut), the prior or
    current bout was a catchweight/open weight, or the two bouts sit
    on different ladders (a men's-to-women's comparison, which
    shouldn't happen but is refused rather than guessed). None means
    "can't say," which is different from 0's "no change."
    """
    query = """
        SELECT b.weight_class
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE $fighter_id IN (b.fighter_red_id, b.fighter_blue_id)
          AND e.event_date < $as_of_date
        ORDER BY e.event_date DESC
        LIMIT 1
    """
    prior = con.execute(
        query, {"fighter_id": fighter_id, "as_of_date": as_of_date}
    ).fetchone()

    if prior is None:
        return None  # debut

    current_query = """
        SELECT b.weight_class
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE $fighter_id IN (b.fighter_red_id, b.fighter_blue_id)
          AND e.event_date = $as_of_date
        LIMIT 1
    """
    current = con.execute(
        current_query, {"fighter_id": fighter_id, "as_of_date": as_of_date}
    ).fetchone()

    if current is None:
        return None

    prior_pos = _ladder_position(normalize_weight_class(prior[0]))
    current_pos = _ladder_position(normalize_weight_class(current[0]))

    if prior_pos is None or current_pos is None:
        return None
    if prior_pos[0] != current_pos[0]:  # different ladders
        return None

    delta = current_pos[1] - prior_pos[1]
    return (delta > 0) - (delta < 0)  # sign only: -1, 0, or +1


def build_weight_class_change_by_bout(
    con: duckdb.DuckDBPyConnection, labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Bout-level red_/blue_ shape, same as build_sos_by_bout and
    build_recent_damage_by_bout — ready for attach_by_corner.
    """
    rows = []
    for bout in labels.itertuples(index=False):
        rows.append({
            "bout_id": bout.bout_id,
            "red_weight_class_change": weight_class_change(
                con, bout.fighter_red_id, bout.event_date
            ),
            "blue_weight_class_change": weight_class_change(
                con, bout.fighter_blue_id, bout.event_date
            ),
        })
    return pd.DataFrame(rows)