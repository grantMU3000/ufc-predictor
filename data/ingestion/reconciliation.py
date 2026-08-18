"""
The Greco <-> Wikipedia bridge that ADR-011 flagged as a known gap.

The problem in one sentence
--------------------------
Two pipelines write into the same `events`/`bouts` tables and key on
different columns. Wikipedia writes first (keyed on `wikipedia_pageid`,
`source_url` NULL). Greco writes later, after the fight (keyed on
`source_url`). Because Greco's upsert only looks for a matching
`source_url` and there isn't one yet, it inserts a brand-new row instead
of updating the row that's already sitting there — and you end up with the
same real card in the database twice.

The fix, ELI5
-------------
Think of `source_url` as a name tag on a seat. Greco walks in, looks for a
seat with its name tag on it, finds none, and drags in a new chair. This
module walks the room *before* Greco does and writes Greco's name tag onto
the chair the Wikipedia pipeline already set out. Greco then finds its name
tag, sits in the existing chair, and no second chair appears.

Why that's the right shape
--------------------------
It requires zero changes to the upsert logic in `loaders.py`. That logic is
already correct — "find the row with this source_url and update it in
place." All that was missing was making sure the source_url landed on the
pre-existing row first. Keeping the existing row's `id` also means anything
already pointing at it (odds_snapshots, and predictions from Week 4 onward)
stays valid without a repointing step. That's the direction the one-off
UFC 330 cleanup had to be forced into by hand; from here it happens
automatically.

Both functions are idempotent: they only ever touch rows where
`source_url IS NULL`, so a second run finds nothing left to claim.
"""

import re
import unicodedata
from typing import SupportsInt, cast

import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy import Engine, text

# How far apart two dates can be and still be the same card. Wikipedia
# lists the local calendar date of the event; ufcstats sometimes lands on
# the other side of midnight UTC for cards in Australia or Abu Dhabi.
DATE_WINDOW_DAYS = 1

# Deliberately low. The date window plus "is this even a UFC-numbered
# event" check below already does almost all of the filtering — the UFC
# runs one card per date, so two events on the same date that are both
# numbered the same are the same event. This threshold only exists as a
# last-resort tripwire for Fight Night cards, where neither name carries
# a number.
NAME_SIMILARITY_FLOOR = 55

# Matches "UFC 330" / "UFC330" but deliberately NOT "UFC on ESPN 60" or
# "UFC Fight Night 250" (the digits there aren't a numbered-event number).
_UFC_NUMBER_RE = re.compile(r"\bufc\s*(\d{1,4})\b")


def _normalize(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Comparison only, never stored."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _ufc_number(name: str) -> str | None:
    """Return the numbered-event number ('330') if this is a numbered card, else None."""
    match = _UFC_NUMBER_RE.search(_normalize(name))
    return match.group(1) if match else None


def event_names_compatible(name_a: str, name_b: str) -> bool:
    """
    Could these two event names describe the same real card?

    Three cases:
      1. Both are numbered cards -> the numbers must match, and nothing
         else matters. 'UFC 330' vs 'UFC 330: Makhachev vs. Garry' is the
         exact case that broke, and a plain fuzzy score on those two
         strings is unreliable because one is a strict prefix of a much
         longer string. Comparing the number is both stricter AND more
         permissive in the right places.
      2. One numbered, one not -> reject. A numbered PPV and a Fight Night
         are never the same card, even on the same date (which does happen
         — e.g. an APEX card and a PPV in the same weekend window).
      3. Neither numbered (Fight Nights) -> fall back to a fuzzy score.
         token_set_ratio rather than WRatio because the useful signal is
         the shared fighter surnames, in any order, ignoring differing
         boilerplate ('UFC Fight Night:' vs 'UFC on ABC 8:').
    """
    num_a, num_b = _ufc_number(name_a), _ufc_number(name_b)

    if num_a is not None and num_b is not None:
        return num_a == num_b
    if (num_a is None) != (num_b is None):
        return False

    return fuzz.token_set_ratio(_normalize(name_a), _normalize(name_b)) >= NAME_SIMILARITY_FLOOR


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------

def claim_existing_events_for_greco(engine: Engine, events_df: pd.DataFrame) -> int:
    """
    For each incoming Greco event, if an unreconciled Wikipedia-sourced row
    for that same real card already exists, write Greco's `source_url` onto
    it so the normal upsert updates that row instead of inserting a new one.

    Args:
        engine: SQLAlchemy engine.
        events_df: the Greco events frame, pre-load. Needs columns
            `name`, `event_date`, `source_url`.

    Returns:
        How many existing rows were claimed. Normally 0 or 1 per run — you
        only ever have a handful of unreconciled Wikipedia events, and only
        the ones whose card just happened become claimable.

    Ambiguity is never resolved by guessing: if one Wikipedia row plausibly
    matches two different incoming Greco events, the row is left alone and
    a warning is printed. `check_duplicate_events` will then keep flagging
    it, which is the correct outcome — a visible unresolved duplicate beats
    a silent wrong merge.
    """
    unreconciled = pd.read_sql(
        """
        SELECT id, name, event_date
        FROM events
        WHERE source_url IS NULL
          AND wikipedia_pageid IS NOT NULL
        """,
        engine,
        parse_dates=["event_date"],
    )
    if unreconciled.empty:
        return 0

    incoming = events_df[["name", "event_date", "source_url"]].copy()
    incoming["event_date"] = pd.to_datetime(incoming["event_date"])

    # URLs already present in `events` can't be claimed onto a second row —
    # uq_events_source_url would reject it. This is the guard against
    # re-running the claim after a previous run already reconciled the pair.
    taken_urls = set(
        pd.read_sql("SELECT source_url FROM events WHERE source_url IS NOT NULL", engine)[
            "source_url"
        ]
    )

    claimed = 0
    for _, wiki_row in unreconciled.iterrows():
        day_gap = (incoming["event_date"] - wiki_row["event_date"]).abs().dt.days
        nearby = incoming[day_gap <= DATE_WINDOW_DAYS]

        candidates = [
            row for _, row in nearby.iterrows()
            if row["source_url"] not in taken_urls
            and event_names_compatible(wiki_row["name"], row["name"])
        ]

        if not candidates:
            continue

        if len(candidates) > 1:
            print(
                f"reconciliation: events.id={wiki_row['id']} '{wiki_row['name']}' "
                f"matched {len(candidates)} incoming Greco events "
                f"({[c['name'] for c in candidates]}) — skipped, needs manual "
                f"resolution. This will keep failing check_duplicate_events "
                f"until you resolve it, which is intended."
            )
            continue

        match = candidates[0]
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE events
                    SET source_url = :url
                    WHERE id = :id AND source_url IS NULL
                """),
                {"url": match["source_url"], "id": int(wiki_row["id"])},
            )
        if result.rowcount:
            taken_urls.add(match["source_url"])
            claimed += 1
            print(
                f"reconciliation: claimed events.id={wiki_row['id']} "
                f"'{wiki_row['name']}' for Greco event '{match['name']}' "
                f"— Greco will now UPDATE this row rather than insert a duplicate."
            )

    return claimed


# ---------------------------------------------------------------------
# Bouts
# ---------------------------------------------------------------------

def claim_existing_bouts_for_greco(engine: Engine, bouts_df: pd.DataFrame) -> int:
    """
    Same trick, one level down: write Greco's bout `source_url` onto the
    already-existing scheduled bout row for that matchup.

    MUST be called with a `bouts_df` whose `event_id`, `fighter_red_id` and
    `fighter_blue_id` have already been remapped to real database ids — i.e.
    after `_remap_ids` inside `load_bouts`, not before.

    Matching is on (event_id, unordered fighter pair). Unordered because
    Wikipedia and Greco don't always agree on who's in the red corner, and
    the pair is unique within an event — the same two fighters can't be
    booked twice on one card, so there's no rematch ambiguity to worry about.

    Performance note: only bouts with `source_url IS NULL` and
    `status = 'scheduled'` are claimable, and there are only ever ~100 of
    those (one or two upcoming cards). So the candidate set is loaded once
    into a dict and the ~8,600-row incoming frame is filtered against it in
    memory. No per-row database round trip for the 8,500 bouts that could
    never match anyway.
    """
    candidates = pd.read_sql(
        """
        SELECT id, event_id,
               LEAST(fighter_red_id, fighter_blue_id)    AS pair_low,
               GREATEST(fighter_red_id, fighter_blue_id) AS pair_high,
               fighter_red_id
        FROM bouts
        WHERE source_url IS NULL
          AND status = 'scheduled'
        """,
        engine,
    )
    if candidates.empty:
        return 0

    by_key: dict[tuple[int, int, int], tuple[int, int]] = {
        (
            int(cast(SupportsInt, r.event_id)),
            int(cast(SupportsInt, r.pair_low)),
            int(cast(SupportsInt, r.pair_high)),
        ): (
            int(cast(SupportsInt, r.id)),
            int(cast(SupportsInt, r.fighter_red_id)),
        )
        for r in candidates.itertuples()
    }

    df = bouts_df.dropna(subset=["event_id", "fighter_red_id", "fighter_blue_id", "source_url"])

    claimed = 0
    for row in df.itertuples():
        red, blue = (
            int(cast(SupportsInt, row.fighter_red_id)),
            int(cast(SupportsInt, row.fighter_blue_id)),
        )
        key = (int(cast(SupportsInt, row.event_id)), min(red, blue), max(red, blue))

        hit = by_key.get(key)
        if hit is None:
            continue

        bout_id, existing_red = hit

        # A corner flip is harmless for the model (features are symmetrized
        # per ADR-004) but NOT harmless for a logged prediction:
        # `predicted_prob_red` was written against the old ordering. Warn
        # loudly so it's visible, and rely on `predicted_winner_id` — which
        # is ordering-independent — as the source of truth at settlement.
        if existing_red != red:
            print(
                f"reconciliation: bouts.id={bout_id} corners flipped by Greco "
                f"(red {existing_red} -> {red}). Any existing prediction's "
                f"predicted_prob_red now refers to the other fighter — settle "
                f"on predicted_winner_id, not the corner."
            )

        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    UPDATE bouts
                    SET source_url = :url
                    WHERE id = :id AND source_url IS NULL
                      AND NOT EXISTS (SELECT 1 FROM bouts WHERE source_url = :url)
                """),
                {"url": row.source_url, "id": bout_id},
            )
        if result.rowcount:
            del by_key[key]  # one claim per existing row
            claimed += 1

    if claimed:
        print(f"reconciliation: claimed {claimed} existing scheduled bout(s) for Greco results")

    return claimed
