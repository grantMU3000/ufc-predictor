"""
tests/test_upcoming_events_loader_unit.py

Offline tests for the pure-logic pieces of
data/ingestion/upcoming_events_loader.py -- specifically
infer_bout_details(), which decides is_title_fight and
scheduled_rounds for a bout Wikipedia doesn't state either of directly.

ELI5: infer_bout_details() is a simple rule-checker, not a database
call -- like a referee who just looks at two pieces of paper (is there
a champion marker? is this the first fight listed?) and writes down
"5 rounds" or "3 rounds." Because it doesn't touch the database or the
network, it's fully testable by just handing it fake paper and checking
what it writes down.

This is the piece directly behind bug #5 from the ingestion session
(main-event bouts defaulting to 3 rounds because only is_title_fight
was checked, not card position) -- these tests lock that fix in.

Note: this only tests infer_bout_details() itself, the pure decision
logic. Whether load_bout() correctly WRITES that decision to the
database -- including the rounds_confirmed guard that stops a manual
correction from being silently reverted (bug #6) -- needs a real
database and is covered separately in tests/integration/.
"""

from data.ingestion.upcoming_events_loader import infer_bout_details


def _bout(is_red_champ=False, is_blue_champ=False, notes=""):
    """Minimal bout dict -- only the three keys infer_bout_details reads."""
    return {
        "fighter_red_is_champion": is_red_champ,
        "fighter_blue_is_champion": is_blue_champ,
        "notes": notes,
    }


class TestInferBoutDetails:
    def test_title_fight_via_champion_marker_gets_five_rounds(self):
        result = infer_bout_details(_bout(is_red_champ=True), is_main_event=False)
        assert result["is_title_fight"] is True
        assert result["scheduled_rounds"] == 5

    def test_non_title_main_event_gets_five_rounds(self):
        # This is the exact case bug #5 got wrong: a non-title
        # headliner still gets 5 rounds by UFC convention, purely
        # because of card position, not the title-fight flag.
        result = infer_bout_details(_bout(), is_main_event=True)
        assert result["is_title_fight"] is False
        assert result["scheduled_rounds"] == 5

    def test_title_fight_detected_from_notes_text(self):
        # Interim titles etc. sometimes show up only in freeform notes
        # text, not a champion marker on either fighter.
        result = infer_bout_details(_bout(notes="Interim title bout"), is_main_event=False)
        assert result["is_title_fight"] is True
        assert result["scheduled_rounds"] == 5

    def test_regular_bout_gets_three_rounds(self):
        result = infer_bout_details(_bout(), is_main_event=False)
        assert result["is_title_fight"] is False
        assert result["scheduled_rounds"] == 3

    def test_original_bout_fields_are_preserved(self):
        # infer_bout_details returns the ORIGINAL bout dict plus two new
        # keys -- it shouldn't drop or mutate anything else that was on
        # the bout (e.g. fighter names, weight_class).
        original = _bout()
        original["weight_class"] = "Welterweight"
        result = infer_bout_details(original, is_main_event=False)
        assert result["weight_class"] == "Welterweight"