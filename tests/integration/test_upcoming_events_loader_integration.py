"""
tests/integration/test_upcoming_events_loader_integration.py

Integration tests for load_bout() (data/ingestion/upcoming_events_loader.py)
against a real, disposable Postgres test database.

ELI5: the unit tests in tests/test_upcoming_events_loader_unit.py check
the DECISION ("this bout should be 5 rounds"). These tests check the
WRITE -- did that decision actually land correctly in the database, and
does it survive being asked to write the same thing twice? That second
part genuinely needs a real database, because it depends on real SQL
behavior (does this become an UPDATE or a second INSERT? does the
rounds_confirmed guard actually stop an overwrite?) that no amount of
pure-Python testing can substitute for.

Requires TEST_DATABASE_URL to be set (see conftest.py in this folder).
Every test here is automatically skipped, not failed, if it isn't --
so a plain `pytest` run from the repo root stays green on a machine
that hasn't set up a test database yet.

NOTE ON VALIDATION: unlike every other test file in this project's
suite, this one has NOT been run against a real Postgres database as
part of writing it -- there's no live database connection available in
the environment these tests were written in. It's been checked
carefully against the actual bouts/events/fighters schema and
load_bout()'s real logic, and it compiles cleanly, but treat the FIRST
real run of this file as the actual verification step, not this
write-up.
"""

from sqlalchemy import text

from data.ingestion.upcoming_events_loader import load_bout


def _bout(weight_class="Lightweight", card_tier="Main card",
          red_champ=False, blue_champ=False, notes=""):
    """
    Minimal bout dict with every key load_bout()/infer_bout_details()
    actually reads. Defaults describe a plain, non-title, non-main-event
    bout -- the "nothing special" case each test starts from.
    """
    return {
        "weight_class": weight_class,
        "card_tier": card_tier,
        "fighter_red_is_champion": red_champ,
        "fighter_blue_is_champion": blue_champ,
        "notes": notes,
    }


def _fetch_scheduled_bout(db_engine, event_id, fighter_red_id, fighter_blue_id):
    """The one active ('scheduled') bout for this exact pairing, or None."""
    with db_engine.connect() as conn:
        return conn.execute(
            text("""
                SELECT * FROM bouts
                WHERE event_id = :event_id
                  AND fighter_red_id = :red AND fighter_blue_id = :blue
                  AND status = 'scheduled'
            """),
            {"event_id": event_id, "red": fighter_red_id, "blue": fighter_blue_id},
        ).fetchone()


class TestLoadBoutInsertAndUpdate:
    def test_first_call_inserts_a_scheduled_bout(self, db_engine, sample_event, sample_fighters):
        red_id, blue_id, _ = sample_fighters

        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)

        row = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)
        assert row is not None
        assert row.status == "scheduled"
        assert row.scheduled_rounds == 3  # non-title, non-main-event -> 3 by default

    def test_rerun_with_no_changes_updates_the_same_row_in_place(
        self, db_engine, sample_event, sample_fighters
    ):
        red_id, blue_id, _ = sample_fighters

        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)
        first_row = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)

        # Rerun with the exact same pairing and details -- this is the
        # idempotency property check_upcoming_events_idempotency (in
        # quality_checks.py) verifies at the whole-pipeline level; this
        # test verifies the same property one function at a time.
        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)
        second_row = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)

        # Same bout id -- an UPDATE happened, not a second INSERT.
        assert second_row.id == first_row.id

    def test_manual_round_correction_survives_a_rerun(
        self, db_engine, sample_event, sample_fighters
    ):
        # Locks in the fix for bug #6: a human manually corrects
        # scheduled_rounds and flips rounds_confirmed=true. The next
        # ingest rerun of the SAME bout (still not flagged main-event
        # or title fight in the source data) must not silently revert
        # that correction back to the inferred default.
        red_id, blue_id, _ = sample_fighters

        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)
        row = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)
        assert row.scheduled_rounds == 3  # sanity check before the manual fix

        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE bouts SET scheduled_rounds = 5, rounds_confirmed = true "
                    "WHERE id = :id"
                ),
                {"id": row.id},
            )

        # Rerun the exact same ingest call again -- nothing about the
        # source data changed.
        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)
        row_after = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)

        assert row_after.scheduled_rounds == 5  # correction preserved, not reverted


class TestLoadBoutFighterSwap:
    def test_fighter_swap_cancels_old_row_and_inserts_new_one(
        self, db_engine, sample_event, sample_fighters
    ):
        # This is ADR-010's cancel-and-reinsert design, exercised
        # end-to-end: a late replacement should never delete the
        # original booking (it stays, marked cancelled, for the
        # historical record) and should always land as a fresh row for
        # the new pairing.
        red_id, blue_id, swap_id = sample_fighters

        load_bout(db_engine, sample_event, _bout(), red_id, blue_id, is_main_event=False)
        original_row = _fetch_scheduled_bout(db_engine, sample_event, red_id, blue_id)

        # Red corner fighter gets swapped out for a late replacement.
        load_bout(db_engine, sample_event, _bout(), swap_id, blue_id, is_main_event=False)

        with db_engine.connect() as conn:
            original_status = conn.execute(
                text("SELECT status FROM bouts WHERE id = :id"), {"id": original_row.id}
            ).scalar_one()

        new_row = _fetch_scheduled_bout(db_engine, sample_event, swap_id, blue_id)

        assert original_status == "cancelled"
        assert new_row is not None
        assert new_row.status == "scheduled"
        assert new_row.id != original_row.id