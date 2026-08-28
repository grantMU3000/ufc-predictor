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

import math
from collections.abc import Callable
from typing import cast, overload

import numpy as np
import pandas as pd

# ADR-014's tuned K-factor parameters. Today these same three numbers
# also live as signature defaults in build_lgbm_matrix
# ._load_labels_and_elo. Named here so the inference path references
# one source instead of becoming a third copy of three magic numbers
# that must never silently disagree.
TUNED_K_NEW = 80.0
TUNED_K_VETERAN = 24.0
TUNED_DECAY_SCALE = 3.0

# The starting rating for a fighter's first-ever appearance. Named so
# callers that look a fighter up in a ratings dict fall back to the
# same value the walk itself would have used.
DEFAULT_INITIAL_RATING = 1500.0

@overload
def expected_score(rating_a: float, rating_b: float) -> float: ...
@overload
def expected_score(rating_a: np.ndarray, rating_b: np.ndarray) -> np.ndarray: ...
def expected_score(rating_a, rating_b):
    """
    The classic Elo formula: given two ratings, what's fighter A's
    probability of beating fighter B?

    Simple version: this one line IS Elo. A 200-point gap works out
    to roughly a 76% win probability for the higher-rated fighter; a
    400-point gap is about 91%. The curve is symmetric —
    expected_score(a, b) always equals 1 - expected_score(b, a).

    Works elementwise on either scalars or numpy arrays — the two
    @overload signatures above just tell mypy which shape comes back
    for which input, since the arithmetic itself needs no branching.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _walk_elo(
    bouts: pd.DataFrame,
    k_factor: float | Callable[[int], float],
    initial_rating: float,
) -> tuple[list[dict], dict[int, float]]:
    """
    The single Elo walk, shared by compute_elo_ratings (which wants
    the per-bout PRE-fight ratings) and compute_current_ratings
    (which wants the final standings after the last bout).

    Extracted rather than duplicated: two copies of an order-dependent
    sequential update is two chances for the training path and the
    inference path to drift apart by a K-factor or an update rule,
    and that drift would be invisible — both would still return
    plausible ratings.

    Returns
    -------
    (rows, ratings)
        rows: one dict per input bout — bout_id, red_elo_pre,
            blue_elo_pre. Exactly what compute_elo_ratings returned
            before this refactor.
        ratings: fighter_id -> rating AFTER every bout in `bouts` has
            been applied. For a fight that hasn't happened yet, this
            IS each fighter's pre-fight rating.
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
        red_id = cast(int, bout.fighter_red_id)
        blue_id = cast(int, bout.fighter_blue_id)

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

    return rows, ratings


def compute_elo_ratings(
    bouts: pd.DataFrame,
    k_factor: float | Callable[[int], float] = 32.0,
    initial_rating: float = DEFAULT_INITIAL_RATING,
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
    rows, _ = _walk_elo(bouts, k_factor, initial_rating)
    return pd.DataFrame(rows)


def compute_current_ratings(
    bouts: pd.DataFrame,
    k_factor: float | Callable[[int], float] = 32.0,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> dict[int, float]:
    """
    Every fighter's rating AFTER the last bout in `bouts` — the thin
    wrapper this module's original docstring anticipated ("Final /
    'current' ratings after the last bout aren't returned here — a
    different, smaller need").

    Simple version: compute_elo_ratings hands back the league
    standings as they stood before each game of the season. This
    hands back the standings at the end. For predicting a game that
    hasn't been played, the end-of-season table IS what both teams
    carry into it.

    THIS IS NOT AN APPROXIMATION. A bout's pre-fight rating depends
    exclusively on strictly-earlier bouts, so for a fight that hasn't
    happened, "rating after every completed bout" and "rating going
    into this bout" are the same number by construction.

    Feed this ONLY completed bouts that occurred strictly before the
    fight being predicted — same caller-owns-the-cutoff rule as
    compute_elo_ratings (see module docstring).

    Returns
    -------
    dict[int, float] — fighter_id -> rating. A fighter absent from
    the dict has no prior bouts in `bouts`; callers should fall back
    to DEFAULT_INITIAL_RATING, matching what the walk itself would
    have used on their debut.
    """
    _, ratings = _walk_elo(bouts, k_factor, initial_rating)
    return ratings


def tuned_k_factor(fight_count: int) -> float:
    """
    k_factor_by_experience pinned to ADR-014's tuned parameters —
    the exact callable the training matrix was built with.

    Exists so the inference path can't accidentally pass raw defaults
    or a re-typed 80/24/3. One name, one meaning.
    """
    return k_factor_by_experience(
        fight_count,
        k_new=TUNED_K_NEW,
        k_veteran=TUNED_K_VETERAN,
        decay_scale=TUNED_DECAY_SCALE,
    )

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