"""
tests/test_wiki_parsers.py

Offline, fixture-driven tests for data/scraping/wiki_parsers.py.

ELI5: wiki_parsers.py's job is to take Wikipedia's raw wikitext (the
"source code" behind a Wikipedia page, before it gets rendered into a
webpage) and turn it into clean Python dicts your database can use. This
file feeds it a small, fake-but-realistically-shaped wikitext sample and
checks the dicts that come out are exactly right — no network call, no
real Wikipedia page, no database. That's what makes these tests fast
and safe to run on every commit.

Why fixtures instead of hitting live Wikipedia in tests (like
_manual_test_ufc330.py used to do): a fixture is frozen in time. If
Wikipedia's real UFC 330 page changes tomorrow (a fight gets added, a
title changes hands), a test built on the live page starts failing for
a reason that has nothing to do with your code being broken. A saved
fixture only changes when you change it on purpose.
"""

from pathlib import Path

import pytest

from data.scraping.wiki_parsers import parse_fight_card, parse_scheduled_events

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fight_card_wikitext() -> str:
    return (FIXTURES_DIR / "fight_card_sample.wikitext").read_text(encoding="utf-8")


@pytest.fixture
def scheduled_events_wikitext() -> str:
    return (FIXTURES_DIR / "scheduled_events_sample.wikitext").read_text(
        encoding="utf-8"
    )


class TestParseFightCard:
    """
    Covers the exact bug class from bug #5 in the ingestion session:
    main-card vs. preliminary-card tracking, and champion/"(c)"
    detection — both feed directly into the scheduled_rounds /
    is_title_fight inference in upcoming_events_loader.py.
    """

    def test_parses_expected_number_of_bouts(self, fight_card_wikitext):
        bouts = parse_fight_card(fight_card_wikitext)
        assert len(bouts) == 3

    def test_preserves_document_order(self, fight_card_wikitext):
        # Document order matters: upcoming_events_loader.py treats the
        # FIRST bout in this list as the main event (see its docstring
        # on infer_bout_details). If parsing ever silently reordered
        # bouts, every main-event round-count inference downstream
        # would go with it -- quietly.
        bouts = parse_fight_card(fight_card_wikitext)
        assert bouts[0]["fighter_red"] == "Test Fighter Alpha"
        assert bouts[-1]["fighter_red"] == "Test Fighter Epsilon"

    def test_tracks_card_tier_across_bouts(self, fight_card_wikitext):
        bouts = parse_fight_card(fight_card_wikitext)
        assert bouts[0]["card_tier"] == "Main card"
        assert bouts[1]["card_tier"] == "Main card"
        assert bouts[2]["card_tier"] == "Preliminary card"

    def test_detects_champion_marker(self, fight_card_wikitext):
        bouts = parse_fight_card(fight_card_wikitext)
        assert bouts[0]["fighter_red_is_champion"] is True
        assert bouts[0]["fighter_blue_is_champion"] is False
        # The "(c)" marker must be stripped out of the display name --
        # otherwise it ends up looking like part of the fighter's name.
        assert "(c)" not in bouts[0]["fighter_red"]

    def test_extracts_link_target_separately_from_display_text(
        self, fight_card_wikitext
    ):
        # This is bug #7 from the session: a disambiguated Wikipedia
        # link target like "Test Fighter Delta (fighter)" needs to
        # survive separately from the plain display text "Test Fighter
        # Delta", since fighter_resolution.py checks the link target
        # FIRST as the stronger identity signal.
        bouts = parse_fight_card(fight_card_wikitext)
        second_bout = bouts[1]
        assert second_bout["fighter_blue"] == "Test Fighter Delta"
        assert second_bout["fighter_blue_link_target"] == "Test Fighter Delta (fighter)"

    def test_unlinked_fighter_has_no_link_target(self, fight_card_wikitext):
        bouts = parse_fight_card(fight_card_wikitext)
        second_bout = bouts[1]
        assert second_bout["fighter_red"] == "Test Fighter Gamma"
        assert second_bout["fighter_red_link_target"] is None

    def test_extracts_weight_class(self, fight_card_wikitext):
        bouts = parse_fight_card(fight_card_wikitext)
        assert bouts[0]["weight_class"] == "Heavyweight"
        assert bouts[2]["weight_class"] == "Welterweight"

    def test_extracts_result_fields_when_present(self, fight_card_wikitext):
        # Some event pages get result fields filled in same-day by a
        # fast editor, before Greco's daily job confirms anything.
        # ADR-010 says these are never trusted as a DB write, but the
        # parser itself should still extract them faithfully --
        # deciding whether to trust them is upcoming_events_loader's
        # job, not wiki_parsers's.
        bouts = parse_fight_card(fight_card_wikitext)
        third_bout = bouts[2]
        assert third_bout["connector"] == "def."
        assert third_bout["method"] == "KO (punches)"
        assert third_bout["round"] == "2"
        assert third_bout["time"] == "3:14"


class TestParseScheduledEvents:
    """
    Covers the "List of UFC events" schedule table -- the entry point
    that finds which events exist before any single event's fight card
    is even looked at.
    """

    def test_parses_linked_rows_only(self, scheduled_events_wikitext):
        # The fixture has two real linked rows, one unlinked row (no
        # wikilink -> skip), and one malformed/short colspan row
        # (< 4 cells -> skip). Only the two real ones should come out.
        events = parse_scheduled_events(scheduled_events_wikitext)
        assert len(events) == 2

    def test_uses_link_target_not_display_text_for_event_title(
        self, scheduled_events_wikitext
    ):
        # event_title must be the wikilink TARGET (used for the pageid
        # lookup in wiki_api.py), which can differ from what's shown on
        # the page -- the exact "UFC 331" vs. "UFC 331: Ankalaev vs
        # Pereira 2" situation from the duplicate-events check.
        events = parse_scheduled_events(scheduled_events_wikitext)
        beta = next(e for e in events if e["event_title"] == "Test Event Beta")
        assert beta["event_display_name"] == "UFC Test Beta: Someone vs Someone"

    def test_parses_dts_template_date(self, scheduled_events_wikitext):
        events = parse_scheduled_events(scheduled_events_wikitext)
        alpha = next(e for e in events if e["event_title"] == "Test Event Alpha")
        assert alpha["date"].isoformat() == "2026-09-12"

    def test_raises_on_unexpected_headers(self):
        # Guards against silently reading the wrong columns if
        # Wikipedia ever renames or reorders this table's headers --
        # exactly the class of bug that would otherwise fail silently
        # (wrong data, no error) rather than loudly.
        bad_wikitext = (
            "{|\n"
            '! scope="col" | Name\n'
            '! scope="col" | When\n'
            "|-\n"
            "| [[Some Event]]\n"
            "| 2026-01-01\n"
            "|}"
        )
        with pytest.raises(ValueError, match="Unexpected table columns"):
            parse_scheduled_events(bad_wikitext)
