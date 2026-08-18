"""
Computes each fighter's Elo rating over time, and the pre-fight rating
of both corners for every bout — Tier 3's headline feature per
docs/PLAN.md Section 2 ("historically the strongest single feature in
this domain").

Why this file works differently from every other feature file so far:
tier1.py / tier2.py answer "what was true about ONE fighter right
before THIS bout" with a single self-contained query — safe no matter
what order you call them in, since the `event_date < as_of_date`
filter lives inside each query. Elo can't work that way: a fighter's
rating today depends on their rating yesterday, which depends on the
day before that, all the way back to their first pro fight. It's a
running tally, not a snapshot — like a season-long point standings
board. You can't know today's standings without having tracked every
point scored on every day before it, in order.

IMPORTANT — this file does NOT decide which bouts it's allowed to
see. That's the caller's job (models/baselines.py), same as tier1/
tier2 never decide their own as_of_date. Feed this function ONLY
train+val bout history until the Week 3 test-set unlock
(event_date < features.split.TEST_START). Nothing in this function's
own math would actually leak backward into a val prediction if you
fed it test-era bouts too (a bout's PRE-fight rating only ever
depends on strictly earlier bouts) — but the project's rule is "don't
even look at the locked drawer," not just "don't let it change
earlier answers," so the filtering happens upstream, deliberately,
every time this gets called before Week 3.
"""

import pandas as pd


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    The classic Elo formula: given two ratings, what's fighter A's
    probability of beating fighter B?

    Simple version: this one line IS Elo. A 200-point gap works out
    to roughly a 76% win probability for the higher-rated fighter; a
    400-point gap is about 91%. The curve is symmetric —
    expected_score(a, b) always equals 1 - expected_score(b, a).
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_ratings(
    bouts: pd.DataFrame, k_factor: float = 32.0, initial_rating: float = 1500.0
) -> pd.DataFrame:
    """
    Walks every bout in `bouts`, oldest first, and records each
    fighter's PRE-fight rating (what they carried INTO that specific
    fight) before updating both ratings based on the outcome.

    Simple version: for every fight — first WRITE DOWN what each
    fighter's rating already was (that's what this function hands
    back), THEN do the math to move both ratings. Writing it down
    before updating is what makes this leakage-safe: a fight's own
    result can never influence its own prediction.

    K-FACTOR: how many points a single win/loss can move a rating.
    Higher K reacts fast to a recent result but swings more on noise
    (a lucky punch). Lower K is stable but slow to reflect real
    improvement. Kept CONSTANT for every fighter and every fight, per
    your call — Step 5 tunes the actual number against val log loss;
    if that tuning shows a flat K is leaving real signal on the
    table, an experience-based K (higher for a fighter's first
    10-15 fights) is the natural next thing to try, not built yet.

    INITIAL RATING: every fighter starts at `initial_rating` (1500,
    the standard convention) the moment they first appear — there's
    no way to know anything about a true UFC debutant's skill before
    fight one, so "exactly average" is the only honest starting
    point.

    WEIGHT CLASS: deliberately ignored — one global rating per
    fighter, carried across any weight-class moves, per your call.
    Worth revisiting long-term (Section 2 flags this too), not today.

    Parameters
    ----------
    bouts : pd.DataFrame
        Sorted oldest-first (event_date ascending). Needs bout_id,
        event_date, fighter_red_id, fighter_blue_id, winner_id — the
        same shape features.labels.get_completed_decided_bouts
        returns. Pass ONLY bouts you're currently allowed to see.
    k_factor : float
        Points moved per fight.
    initial_rating : float
        Starting rating for a fighter's first-ever appearance.

    Returns
    -------
    pd.DataFrame, one row per input bout:
        bout_id, red_elo_pre, blue_elo_pre
    Join back onto anything by bout_id. Final/"current" ratings after
    the last bout aren't returned here — a different, smaller need,
    better served by its own thin wrapper if it comes up.

    Raises
    ------
    ValueError if `bouts` isn't sorted oldest-first. Elo is entirely
    order-dependent — feeding it out of order wouldn't crash, it
    would just silently compute wrong ratings for every bout after
    the first out-of-order one. Fails loudly here instead.
    """
    if not bouts["event_date"].is_monotonic_increasing:
        raise ValueError(
            "bouts must be sorted oldest-first by event_date — Elo "
            "ratings are order-dependent, and an out-of-order input "
            "would silently produce wrong pre-fight ratings."
        )

    ratings: dict[int, float] = {}
    rows = []

    for bout in bouts.itertuples(index=False):
        red_id = bout.fighter_red_id
        blue_id = bout.fighter_blue_id

        red_rating = ratings.get(red_id, initial_rating)
        blue_rating = ratings.get(blue_id, initial_rating)

        # Record BEFORE any update touches these numbers.
        rows.append(
            {
                "bout_id": bout.bout_id,
                "red_elo_pre": red_rating,
                "blue_elo_pre": blue_rating,
            }
        )

        red_expected = expected_score(red_rating, blue_rating)
        blue_expected = 1.0 - red_expected

        red_actual = 1.0 if bout.winner_id == red_id else 0.0
        blue_actual = 1.0 - red_actual

        ratings[red_id] = red_rating + k_factor * (red_actual - red_expected)
        ratings[blue_id] = blue_rating + k_factor * (blue_actual - blue_expected)

    return pd.DataFrame(rows)