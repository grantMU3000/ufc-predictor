"""
data/ingestion/quality_checks.py

Automated "health inspector" checks for the UFC Predictor database.

Why this file exists:
Two different pipelines write into the same `events` table — the Wikipedia
pipeline (writes rows early, before a fight happens, keyed on
`wikipedia_pageid`) and the Greco pipeline (writes rows after a fight
happens, keyed on `source_url`). Nothing currently tells these two pipelines
"hey, we're talking about the same event" (see ADR-011's Consequences
section). This file holds repeatable checks that catch that kind of problem
automatically, instead of a human noticing it by accident during a demo.

Each check function returns a QualityCheckResult so a runner script (built
later) can run all of them and print a clean pass/fail report.
"""

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz
from sqlalchemy import Engine, text


@dataclass
class QualityCheckResult:
    """
    The report card for a single quality check.

    check_name: short machine-readable label, e.g. "duplicate_events"
    passed: True if nothing suspicious was found
    details: human-readable explanation, useful in a terminal or a
             GitHub Actions log
    """

    check_name: str
    passed: bool
    details: str


def check_duplicate_events(
    engine: Engine,
    date_window_days: int = 1,
    name_similarity_threshold: int = 70,
) -> QualityCheckResult:
    """
    Look for pairs of `events` rows that are probably the SAME real-world
    UFC card, entered twice by two different pipelines.

    The specific pattern we're hunting for (per ADR-011's known gap):
      - One row came from Wikipedia: `source_url IS NULL`,
        `wikipedia_pageid` IS NOT NULL (an "upcoming" row that hasn't
        been reconciled with Greco yet)
      - Another row came from Greco: `source_url IS NOT NULL`
      - Both rows have a close event_date and a similar name

    Why fuzzy name matching instead of an exact match: Wikipedia and
    ufcstats.com don't format event names identically (e.g. "UFC 331:
    Ankalaev vs. Pereira 2" vs. a slightly different ufcstats title), so
    an exact string match would miss real duplicates. rapidfuzz is already
    a project dependency (used for fighter-name matching), so this reuses
    a tool you already trust rather than adding something new.

    Why a date *window* instead of an exact date match: occasionally a
    date gets parsed slightly differently across sources (timezone
    rounding, "night of" vs. "calendar day" conventions). A 1-day window
    catches that without being so loose it starts matching unrelated events.

    Args:
        engine: a SQLAlchemy engine connected to the Postgres database.
        date_window_days: how many days apart two events can be and still
            be considered a possible match. Default 1 day.
        name_similarity_threshold: minimum rapidfuzz WRatio score (0-100)
            to treat two event names as "probably the same event."
            Default 70 — loose enough to survive minor title differences,
            tight enough to avoid flagging genuinely different cards.

    Returns:
        QualityCheckResult — passed=True means no likely duplicates found.
    """
    events_df = pd.read_sql(
        """
        SELECT id, name, event_date, source_url, wikipedia_pageid
        FROM events
        ORDER BY event_date
        """,
        engine,
        parse_dates=["event_date"],
    )

    # Rows still waiting on a Greco match: came from Wikipedia, no
    # ufcstats URL attached yet.
    wiki_only = events_df[
        events_df["source_url"].isna() & events_df["wikipedia_pageid"].notna()
    ]
    # Rows Greco has already written a real result for.
    greco_rows = events_df[events_df["source_url"].notna()]

    suspects = []
    for _, wiki_row in wiki_only.iterrows():
        date_diff_days = (
            (greco_rows["event_date"] - wiki_row["event_date"]).abs().dt.days
        )
        nearby_greco = greco_rows[date_diff_days <= date_window_days]

        for _, greco_row in nearby_greco.iterrows():
            similarity = fuzz.WRatio(wiki_row["name"], greco_row["name"])
            if similarity >= name_similarity_threshold:
                suspects.append(
                    {
                        "wiki_event_id": wiki_row["id"],
                        "wiki_name": wiki_row["name"],
                        "wiki_date": wiki_row["event_date"].date(),
                        "greco_event_id": greco_row["id"],
                        "greco_name": greco_row["name"],
                        "greco_date": greco_row["event_date"].date(),
                        "name_similarity": round(similarity, 1),
                    }
                )

    passed = len(suspects) == 0

    if passed:
        details = (
            f"No likely duplicates found across {len(wiki_only)} "
            f"unreconciled Wikipedia-origin event(s) and {len(greco_rows)} "
            f"Greco-origin event(s)."
        )
    else:
        lines = [
            f"  - events.id={s['wiki_event_id']} '{s['wiki_name']}' "
            f"({s['wiki_date']}, Wikipedia) looks like "
            f"events.id={s['greco_event_id']} '{s['greco_name']}' "
            f"({s['greco_date']}, Greco) — name similarity "
            f"{s['name_similarity']}/100"
            for s in suspects
        ]
        details = (
            f"Found {len(suspects)} likely duplicate event pair(s) — these "
            f"need manual reconciliation until the Greco↔Wikipedia bridge "
            f"(ADR-011 follow-up) is built:\n" + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="duplicate_events",
        passed=passed,
        details=details,
    )


def check_duplicate_bouts(engine: Engine) -> QualityCheckResult:
    """
    Look for two `bouts` rows describing the same fight on the same card.

    ELI5: on a real UFC card, a given pair of fighters appears exactly
    once. If the database says Makhachev vs. Garry happened twice on
    UFC 330, something duplicated a row — this is the check that would
    have caught that before it reached the feature store.

    Why this matters more than it looks: this is a leakage-class bug, not
    just untidiness. A duplicated bout means the feature store's
    `get_prior_bouts` counts that fight twice toward both fighters'
    records, Elo updates twice off one result, and symmetrization emits
    the same fight as four training rows instead of two. Any of those
    quietly shifts metrics without ever throwing an error — exactly the
    failure mode PLAN.md §0.2 warns about.

    Detection is on (event_id, unordered fighter pair) rather than on
    source_url, because the duplicated row is precisely the one whose
    source_url started out NULL — keying on source_url would miss it by
    construction. This is the same reasoning `claim_existing_bouts_for_greco`
    (data/ingestion/reconciliation.py) uses to prevent the duplicate in the
    first place; this check is the second line of defense in case that
    prevention step ever misses a case (e.g. a stub-fighter row on one side
    that never got reconciled to the real fighter).

    'cancelled' bouts are excluded: a cancelled row plus its replacement is
    the *intended* result of the ADR-010 fighter-swap logic, not a
    duplicate.

    Returns QualityCheckResult — passed=True means no fight is booked
    twice on any card.
    """
    dupes = pd.read_sql(
        """
        SELECT b.event_id,
               e.name AS event_name,
               LEAST(b.fighter_red_id, b.fighter_blue_id)    AS pair_low,
               GREATEST(b.fighter_red_id, b.fighter_blue_id) AS pair_high,
               count(*)                          AS row_count,
               array_agg(b.id ORDER BY b.id)      AS bout_ids,
               array_agg(b.status ORDER BY b.id)  AS statuses
        FROM bouts b
        JOIN events e ON e.id = b.event_id
        WHERE b.status <> 'cancelled'
        GROUP BY b.event_id, e.name, pair_low, pair_high
        HAVING count(*) > 1
        ORDER BY b.event_id
        """,
        engine,
    )

    passed = dupes.empty
    if passed:
        details = "No fight is booked more than once on the same card."
    else:
        lines = [
            f"  - event_id={r.event_id} '{r.event_name}': fighters "
            f"({r.pair_low}, {r.pair_high}) appear in {r.row_count} bouts "
            f"{r.bout_ids} with statuses {r.statuses}"
            for r in dupes.itertuples()
        ]
        details = (
            f"Found {len(dupes)} duplicated bout group(s). These must be "
            f"merged (repoint bout_stats/odds_snapshots/predictions onto "
            f"the surviving row, then delete the empty one) BEFORE any "
            f"training snapshot is rebuilt — a duplicated bout double-"
            f"counts in the feature store and in Elo:\n" + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="duplicate_bouts", passed=passed, details=details
    )


def check_bout_fk_integrity(engine: Engine) -> QualityCheckResult:
    """
    Make sure every bout's fighter/event references point at rows that
    actually exist.

    ELI5: every bout row points at two fighters and one event using ID
    numbers, like a sticky note that says "see fighter #42" instead of
    writing that fighter's whole bio out again. This check walks through
    every sticky note and makes sure #42 actually exists — not a page
    that got torn out of the book.

    Why bother, when the database already has foreign key constraints
    that should make this impossible? FK constraints only stop a *new*
    write that has a dangling reference at the moment it's written.
    They don't protect against every path that could disturb the data
    afterward — manual fixes (like the Ozzy Diaz/Choi Doo-ho cleanup
    this session), a bug in the remap logic in loaders.py's
    `_remap_ids`, or a future schema change with a looser constraint.
    This check verifies "is the data actually consistent right now,"
    which is a different question than "did Postgres allow the insert."

    Also checks fighter_red_id != fighter_blue_id (a fighter can't be
    booked against themselves) — this mirrors the DB's own CHECK
    constraint from ADR-005, included here as a second line of defense,
    not because it's expected to ever actually fire.

    Returns QualityCheckResult — passed=True means every reference on
    every bout resolves to a real row.
    """
    query = text("""
        SELECT b.id AS bout_id, b.fighter_red_id, b.fighter_blue_id,
               b.event_id, b.winner_id
        FROM bouts b
        LEFT JOIN fighters fr ON b.fighter_red_id = fr.id
        LEFT JOIN fighters fb ON b.fighter_blue_id = fb.id
        LEFT JOIN events e ON b.event_id = e.id
        LEFT JOIN fighters fw ON b.winner_id = fw.id
        WHERE fr.id IS NULL
           OR fb.id IS NULL
           OR e.id IS NULL
           OR (b.winner_id IS NOT NULL AND fw.id IS NULL)
           OR b.fighter_red_id = b.fighter_blue_id
    """)
    with engine.connect() as conn:
        broken = conn.execute(query).fetchall()

    passed = len(broken) == 0
    if passed:
        details = (
            "Every bout's fighter_red_id, fighter_blue_id, event_id, and "
            "winner_id (when set) resolves to a real row, and no bout has "
            "a fighter facing themselves."
        )
    else:
        lines = [
            f"  - bouts.id={row.bout_id} "
            f"(red={row.fighter_red_id}, blue={row.fighter_blue_id}, "
            f"event={row.event_id}, winner={row.winner_id})"
            for row in broken
        ]
        details = (
            f"Found {len(broken)} bout(s) with a broken or invalid "
            f"reference:\n" + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="bout_fk_integrity", passed=passed, details=details
    )


def check_no_leaked_results_on_scheduled_bouts(engine: Engine) -> QualityCheckResult:
    """
    Make sure no bout still marked "scheduled" (hasn't happened yet) has
    a result attached.

    ELI5: a bout with status='scheduled' is a movie that hasn't come out
    yet. This check makes sure nobody's snuck spoilers — a winner, a
    method, a round — onto a movie that's still "coming soon."

    Why this matters specifically for this project: ADR-010 is explicit
    that Wikipedia's own live-edited result fields are never trusted —
    status stays 'scheduled' until Greco confirms a result through the
    normal ingestion path. This check is the automated version of that
    rule. If it ever fires, either that rule got violated somewhere, or
    the still-unbuilt Greco<->Wikipedia reconciliation bridge (ADR-011's
    known follow-up) wrote a result onto the wrong row instead of
    properly flipping status to 'completed'.

    Returns QualityCheckResult — passed=True means no scheduled bout
    carries any result data.
    """
    query = text("""
        SELECT id, winner_id, method, method_detail,
               ending_round, ending_time_seconds
        FROM bouts
        WHERE status = 'scheduled'
          AND (
              winner_id IS NOT NULL
              OR method IS NOT NULL
              OR method_detail IS NOT NULL
              OR ending_round IS NOT NULL
              OR ending_time_seconds IS NOT NULL
          )
    """)
    with engine.connect() as conn:
        leaked = conn.execute(query).fetchall()

    passed = len(leaked) == 0
    if passed:
        details = "No 'scheduled' bout has result data attached."
    else:
        lines = [
            f"  - bouts.id={row.id} status='scheduled' but has "
            f"winner_id={row.winner_id}, method={row.method!r}"
            for row in leaked
        ]
        details = (
            f"Found {len(leaked)} scheduled bout(s) with leaked result "
            f"data:\n" + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="no_leaked_results_on_scheduled_bouts",
        passed=passed,
        details=details,
    )


def check_title_fights_have_five_rounds(engine: Engine) -> QualityCheckResult:
    """
    Make sure every bout flagged as a title fight is scheduled for 5
    rounds.

    ELI5: in the UFC, a fight for the belt is always 5 rounds — that's
    just the rule, like how a basketball game is always 4 quarters. This
    check makes sure every bout marked "title fight" in the database
    actually has scheduled_rounds=5.

    Scope note — why this checks title fights, not "main events":
    See the discussion above `check_duplicate_events` for the full
    reasoning, but in short: `bouts` has no persisted is_main_event flag
    or bout-order column — is_main_event only ever existed as a
    temporary argument passed into `upcoming_events_loader.load_bout()`
    at insert time (see `infer_bout_details()`), and was never saved.
    There's no reliable way to ask the database "which bout was the main
    event" after the fact, so that specific check can't be written
    against the current schema. This check covers the piece that IS
    derivable from stored data. A persisted is_main_event column (small
    migration) would be needed to check the full rule later.

    Returns QualityCheckResult — passed=True means every title fight is
    scheduled for 5 rounds.
    """
    query = text("""
        SELECT id, event_id, weight_class, scheduled_rounds
        FROM bouts
        WHERE is_title_fight = true AND scheduled_rounds != 5 AND NOT rounds_confirmed
    """)
    with engine.connect() as conn:
        mismatches = conn.execute(query).fetchall()

    passed = len(mismatches) == 0
    if passed:
        details = "Every title fight in the database is scheduled for 5 rounds."
    else:
        lines = [
            f"  - bouts.id={row.id} (event_id={row.event_id}, "
            f"{row.weight_class}) is_title_fight=true but "
            f"scheduled_rounds={row.scheduled_rounds}"
            for row in mismatches
        ]
        details = (
            f"Found {len(mismatches)} title fight(s) with the wrong round "
            f"count:\n" + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="title_fights_have_five_rounds", passed=passed, details=details
    )


def check_orphaned_fighter_aliases(engine: Engine) -> QualityCheckResult:
    """
    Make sure every row in fighter_aliases still points at a fighter
    that exists.

    ELI5: fighter_aliases is a box of nickname tags, each one tied by a
    piece of string to one specific fighter's file. If a fighter's file
    ever gets thrown away without untying the string first, you're left
    with a nickname tag pointing at nothing. This check goes through the
    box and finds any tag whose string leads to an empty spot.

    Why this exact check exists: When Ozzy Diaz and Choi Doo-ho's stub
    fighters were manually merged into their real fighter rows, it
    required deleting the stub's alias rows BEFORE deleting the stub
    fighter row — by hand, in the right order. Miss a step or get the
    order wrong, and you're left with an orphaned alias silently
    pointing at a fighter_id that no longer exists — nothing else in the
    pipeline reads fighter_aliases in a way that would ever notice.

    Returns QualityCheckResult — passed=True means every alias points
    at a fighter that actually exists.
    """
    query = text("""
        SELECT fa.fighter_id, fa.alias_name
        FROM fighter_aliases fa
        LEFT JOIN fighters f ON fa.fighter_id = f.id
        WHERE f.id IS NULL
    """)
    with engine.connect() as conn:
        orphans = conn.execute(query).fetchall()

    passed = len(orphans) == 0
    if passed:
        details = "Every fighter_aliases row points at an existing fighter."
    else:
        lines = [
            f"  - alias {row.alias_name!r} points at "
            f"fighter_id={row.fighter_id}, which does not exist"
            for row in orphans
        ]
        details = f"Found {len(orphans)} orphaned alias row(s):\n" + "\n".join(lines)

    return QualityCheckResult(
        check_name="orphaned_fighter_aliases", passed=passed, details=details
    )


def _diff_snapshots(
    before: pd.DataFrame, after: pd.DataFrame, key_col: str, compare_cols: list[str]
) -> list[str]:
    """
    Compare two "before / after" snapshots of the same table and describe,
    in plain English, exactly what changed.

    ELI5: imagine photographing your fridge, then photographing it again
    five minutes later. This function is the friend who looks at both
    photos and tells you exactly what's different — "the milk moved,"
    "a new yogurt showed up" — instead of just shrugging and saying
    "something's off."

    key_col: the column that uniquely identifies a row (e.g. "id"), used
        to match rows up between the two snapshots.
    compare_cols: which columns to check for value changes on rows that
        exist in both snapshots.

    Returns a list of human-readable diff lines. Empty list = no
    differences found.
    """
    before_ids = set(before[key_col])
    after_ids = set(after[key_col])

    lines = []

    for missing_id in sorted(before_ids - after_ids):
        lines.append(
            f"{key_col}={missing_id} was present before the rerun, missing after"
        )

    for new_id in sorted(after_ids - before_ids):
        lines.append(f"{key_col}={new_id} is new — wasn't present before the rerun")

    before_indexed = before.set_index(key_col)
    after_indexed = after.set_index(key_col)
    for shared_id in sorted(before_ids & after_ids):
        before_row = before_indexed.loc[shared_id]
        after_row = after_indexed.loc[shared_id]
        for col in compare_cols:
            both_missing = pd.isna(before_row[col]) and pd.isna(after_row[col])
            if before_row[col] != after_row[col] and not both_missing:
                lines.append(
                    f"{key_col}={shared_id}: {col} changed from "
                    f"{before_row[col]!r} to {after_row[col]!r}"
                )

    return lines


def _snapshot_wikipedia_events(engine: Engine) -> pd.DataFrame:
    """
    Wikipedia-origin events only (wikipedia_pageid IS NOT NULL). Greco-
    origin events are never touched by the upcoming-events pipeline, so
    including them would just be noise for a check that's specifically
    about whether *that* pipeline is idempotent.
    """
    return pd.read_sql(
        """
        SELECT id, name, event_date, venue, location, wikipedia_pageid
        FROM events
        WHERE wikipedia_pageid IS NOT NULL
        ORDER BY id
        """,
        engine,
    )


def _snapshot_wikipedia_bouts(engine: Engine) -> pd.DataFrame:
    """
    Bouts belonging to a Wikipedia-origin event, including 'cancelled'
    rows as well as 'scheduled' ones — a rerun that spuriously triggers
    the cancel-and-reinsert swap logic (ADR-010) with no real underlying
    change is exactly the kind of bug this check exists to catch, and
    that would be invisible if cancelled rows were filtered out.
    """
    return pd.read_sql(
        """
        SELECT b.id, b.event_id, b.fighter_red_id, b.fighter_blue_id,
               b.weight_class, b.is_title_fight, b.scheduled_rounds,
               b.status, b.card_position, b.rounds_confirmed
        FROM bouts b
        JOIN events e ON b.event_id = e.id
        WHERE e.wikipedia_pageid IS NOT NULL
        ORDER BY b.id
        """,
        engine,
    )


def check_upcoming_events_idempotency(
    engine: Engine, rerun_fn=None
) -> QualityCheckResult:
    """
    Rerun the Wikipedia upcoming-events ingestion pipeline and confirm it
    doesn't change anything that shouldn't have changed.

    ELI5: "idempotent" just means "pressing the button twice does the
    same thing as pressing it once." This check presses the ingestion
    button twice, a few seconds apart, with nothing else going on in
    between, and checks that nothing moved that shouldn't have.

    IMPORTANT — how this check is different from every other one above:
      1. It is NOT read-only. It actually re-runs the real ingestion
         pipeline (`data.scraping.ingest_upcoming_events.run`), which
         writes to the database.
      2. It hits the live Wikipedia API (per ADR-009, upcoming-event
         lookups intentionally bypass the cache), so it's slower and
         network-dependent, unlike the pure-SQL checks above.
      3. It can rarely show a "failure" that isn't actually a bug — if a
         fight card genuinely gets edited on Wikipedia in the few
         seconds between the two snapshots, that's a real change, not a
         broken pipeline. This is unlikely enough at a few-second gap to
         not worry about day-to-day, but it's why this check's `details`
         are written to be read, not just trusted blindly on a failure —
         they show exactly what changed, so you can judge "real edit"
         vs. "bug" yourself.

    Because of points 1 and 2, this check is deliberately left OUT of
    `ALL_CHECKS` below — it doesn't belong in a quick, frequent,
    read-only pass/fail loop. Run it deliberately, on its own, whenever
    you've touched the ingestion pipeline and want to confirm reruns are
    still safe — e.g.
    `check_upcoming_events_idempotency(engine)` from a script or REPL.

    Args:
        engine: SQLAlchemy engine.
        rerun_fn: the function to call between snapshots. Defaults to
            the real ingestion pipeline's `run()`. Overridable so this
            can be unit-tested with a no-op instead of hitting the live
            network and a real database.

    Returns QualityCheckResult — passed=True means the second run
    produced identical events/bouts data to the first.
    """
    if rerun_fn is None:
        from data.scraping.ingest_upcoming_events import run as rerun_fn

    events_before = _snapshot_wikipedia_events(engine)
    bouts_before = _snapshot_wikipedia_bouts(engine)

    rerun_fn()

    events_after = _snapshot_wikipedia_events(engine)
    bouts_after = _snapshot_wikipedia_bouts(engine)

    event_compare_cols = ["name", "event_date", "venue", "location"]
    bout_compare_cols = [
        "event_id",
        "fighter_red_id",
        "fighter_blue_id",
        "weight_class",
        "is_title_fight",
        "scheduled_rounds",
        "status",
        "card_position",
        "rounds_confirmed",
    ]

    event_diffs = _diff_snapshots(events_before, events_after, "id", event_compare_cols)
    bout_diffs = _diff_snapshots(bouts_before, bouts_after, "id", bout_compare_cols)

    all_diffs = [f"events: {d}" for d in event_diffs] + [
        f"bouts: {d}" for d in bout_diffs
    ]

    passed = len(all_diffs) == 0
    if passed:
        details = (
            f"Rerunning the ingestion pipeline produced identical results: "
            f"{len(events_before)} Wikipedia-origin event(s), "
            f"{len(bouts_before)} bout(s), nothing changed."
        )
    else:
        lines = [f"  - {d}" for d in all_diffs]
        details = (
            f"Found {len(all_diffs)} difference(s) after rerunning the "
            f"pipeline. If nothing changed on Wikipedia in between, this "
            f"is a real bug (e.g. a rounds_confirmed guard failure, same "
            f"class as bug #6 from the Wikipedia-ingestion session):\n"
            + "\n".join(lines)
        )

    return QualityCheckResult(
        check_name="upcoming_events_idempotency", passed=passed, details=details
    )


# All checks in one place, so the runner (and __main__ below) can loop
# over them instead of listing each call by hand. A dedicated runner
# script with proper CLI output comes in a later step — this is just
# enough structure to run everything with one command in the meantime.
ALL_CHECKS = [
    check_duplicate_events,
    check_duplicate_bouts,
    check_bout_fk_integrity,
    check_no_leaked_results_on_scheduled_bouts,
    check_title_fights_have_five_rounds,
    check_orphaned_fighter_aliases,
]


if __name__ == "__main__":
    # Quick manual run: `python -m data.ingestion.quality_checks`
    # (Swap in however you already build your engine elsewhere in the
    # project — e.g. an existing `get_engine()` helper — rather than
    # hardcoding a connection string here.)
    from data.ingestion.loaders import _get_engine

    engine = _get_engine()
    all_passed = True

    for check_fn in ALL_CHECKS:
        result = check_fn(engine)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.check_name}")
        print(result.details)
        print()
        all_passed = all_passed and result.passed

    if not all_passed:
        raise SystemExit(1)  # non-zero exit so this can gate CI later
