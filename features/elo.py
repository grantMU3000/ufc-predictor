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
import math
from typing import Callable


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
    bouts: pd.DataFrame, 
    k_factor: float | Callable[[int], float] = 32.0, 
    initial_rating: float = 1500.0
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
    fight_counts: dict[int, int] = {}
    rows = []

    for bout in bouts.itertuples(index=False):
        red_id = bout.fighter_red_id
        blue_id = bout.fighter_blue_id

        red_rating = ratings.get(red_id, initial_rating)
        blue_rating = ratings.get(blue_id, initial_rating)
        red_fight_count = fight_counts.get(red_id, 0)
        blue_fight_count = fight_counts.get(blue_id, 0)

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

        k_red = _resolve_k(k_factor, red_fight_count)
        k_blue = _resolve_k(k_factor, blue_fight_count)

        ratings[red_id] = red_rating + k_red * (red_actual - red_expected)
        ratings[blue_id] = blue_rating + k_blue * (blue_actual - blue_expected)

        fight_counts[red_id] = red_fight_count + 1
        fight_counts[blue_id] = blue_fight_count + 1

    return pd.DataFrame(rows)

def k_factor_by_experience(
    fight_count: int,
    k_new: float = 80.0,
    k_veteran: float = 24.0,
    decay_scale: float = 3.0,
) -> float:
    """
    A fighter's K-factor as a function of how many fights they've
    already had — the smooth-decay design, built after the constant-K
    grid search showed log loss wanting a high K (~64-96) while ECE
    wanted a low K (~32) and got dramatically worse past it. Two
    metrics disagreeing about the "best" single number is evidence
    that one number is the wrong shape for this problem — a
    debutant's rating should swing hard (we know nothing about them
    yet); a 30-fight veteran's rating swinging just as hard on one
    result is mostly noise.

    Simple version: like a fresh cup of coffee cooling down. Right
    when it's poured (fight_count=0), it's at its hottest — K equals
    k_new exactly. As fights pile up, K cools toward k_veteran and
    levels off there, same way a cooling cup approaches room
    temperature without ever quite reaching it.

    Formula: k_veteran + (k_new - k_veteran) * e^(-fight_count / decay_scale)
      - fight_count=0            -> exactly k_new
      - fight_count -> infinity  -> approaches k_veteran
      - decay_scale              -> how many fights it takes to cool
        down. At the defaults below, a fighter is already more than
        halfway cooled by fight #7 or so.

    Parameters
    ----------
    fight_count : int
        Prior completed fights this fighter has going into the
        current one. 0 for a debut.
    k_new, k_veteran, decay_scale : float
        Ceiling, floor, and decay speed — all three get grid-searched
        together next, same discipline as the constant-K sweep.
    """
    return k_veteran + (k_new - k_veteran) * math.exp(-fight_count / decay_scale)


def _resolve_k(k_factor, fight_count: int) -> float:
    """
    Small dispatcher: k_factor can be a plain constant (the original
    design) or a callable like k_factor_by_experience (this
    addition). Keeps compute_elo_ratings's main loop from needing an
    if/else every iteration — it just always calls this.
    """
    if callable(k_factor):
        return k_factor(fight_count)
    return k_factor