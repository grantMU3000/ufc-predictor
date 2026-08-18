"""
tests/test_wiki_api.py

Offline tests for data/scraping/wiki_api.py, with the network call
(fetch()) replaced by canned JSON responses shaped like the real
MediaWiki API.

ELI5: this file pretends to be Wikipedia's API for a second. Instead of
actually asking Wikipedia a question over the internet, it hands
wiki_api.py a fake answer that's shaped exactly like a real one, and
checks wiki_api.py reads that answer correctly. This is exactly the
kind of test that would have caught bug #1 from the ingestion session
before a live run did: get_page_info() was originally built using the
wrong API action/param/response-key combination
(action=parse/page=/["parse"] instead of action=query/titles=/
["query"]["pages"]), which would raise a KeyError on every single real
call. A test built against a correctly-shaped fake response catches
that mismatch immediately, offline, without spending a live API call
to find out.
"""

import json
from unittest.mock import patch

import pytest
import requests

from data.scraping import wiki_api


class TestGetPageInfo:
    def test_parses_pageid_and_title(self):
        fake_response = json.dumps(
            {"query": {"pages": {"12345": {"pageid": 12345, "title": "UFC 330"}}}}
        )
        with patch.object(wiki_api, "fetch", return_value=fake_response):
            info = wiki_api.get_page_info("UFC 330")

        assert info == {"pageid": 12345, "title": "UFC 330"}

    def test_requests_correct_api_action(self):
        # Locks in the fix for bug #1: this MUST be action=query with
        # titles=, not action=parse with page=. If this ever regresses,
        # this test fails immediately instead of every real page lookup
        # raising a KeyError in production.
        fake_response = json.dumps(
            {"query": {"pages": {"1": {"pageid": 1, "title": "Some Page"}}}}
        )
        with patch.object(wiki_api, "fetch", return_value=fake_response) as mock_fetch:
            wiki_api.get_page_info("Some Page")

        _, kwargs = mock_fetch.call_args
        assert kwargs["params"]["action"] == "query"
        assert kwargs["params"]["titles"] == "Some Page"

    def test_follows_redirect_to_resolved_title(self):
        # A renamed page (e.g. a Fight Night retitled after its
        # headliner changed) should surface the CURRENT title, not the
        # one that was searched for -- this is the whole reason ADR-011
        # uses pageid instead of title as the stable key.
        fake_response = json.dumps(
            {
                "query": {
                    "pages": {
                        "999": {"pageid": 999, "title": "UFC Fight Night: New Title"}
                    }
                }
            }
        )
        with patch.object(wiki_api, "fetch", return_value=fake_response):
            info = wiki_api.get_page_info("UFC Fight Night: Old Title")

        assert info["title"] == "UFC Fight Night: New Title"


class TestGetSectionIndex:
    FAKE_SECTIONS = json.dumps(
        {
            "parse": {
                "sections": [
                    {"line": "Background", "index": "1"},
                    {"line": "Fight card", "index": "3"},
                ]
            }
        }
    )

    def test_finds_matching_section(self):
        with patch.object(wiki_api, "fetch", return_value=self.FAKE_SECTIONS):
            index = wiki_api.get_section_index("UFC 330", "Fight card")
        assert index == "3"

    def test_raises_with_available_sections_when_not_found(self):
        # The error message should list what sections DO exist -- this
        # is what makes a real failure (e.g. Wikipedia renaming "Fight
        # card" to "Card") debuggable from the exception text alone.
        with (
            patch.object(wiki_api, "fetch", return_value=self.FAKE_SECTIONS),
            pytest.raises(
                wiki_api.SectionNotFoundError, match="Background.*Fight card"
            ),
        ):
            wiki_api.get_section_index("UFC 330", "Nonexistent Section")


class TestGetSectionWikitext:
    def test_returns_raw_wikitext(self):
        fake_response = json.dumps(
            {"parse": {"wikitext": {"*": "{{MMAevent card|Main card}}"}}}
        )
        with patch.object(wiki_api, "fetch", return_value=fake_response):
            text = wiki_api.get_section_wikitext("UFC 330", "3")
        assert text == "{{MMAevent card|Main card}}"

    def test_threads_use_cache_through_to_fetch(self):
        # Bug #3 from the session: use_cache was silently hardcoded to
        # False inside this function, ignoring whatever the caller
        # passed in. This test locks in that the parameter actually
        # reaches fetch().
        fake_response = json.dumps({"parse": {"wikitext": {"*": "text"}}})
        with patch.object(wiki_api, "fetch", return_value=fake_response) as mock_fetch:
            wiki_api.get_section_wikitext("UFC 330", "3", use_cache=False)

        _, kwargs = mock_fetch.call_args
        assert kwargs["use_cache"] is False


class TestResolveEventPageids:
    def test_enriches_events_with_pageid_and_resolved_title(self):
        fake_response = json.dumps(
            {"query": {"pages": {"1": {"pageid": 1, "title": "UFC 330"}}}}
        )
        with patch.object(wiki_api, "fetch", return_value=fake_response):
            events = wiki_api.resolve_event_pageids([{"event_title": "UFC 330"}])

        assert events[0]["pageid"] == 1
        assert events[0]["resolved_title"] == "UFC 330"

    def test_one_failed_lookup_does_not_abort_the_whole_batch(self):
        # This is the actual resilience behavior resolve_event_pageids
        # promises in its own docstring: one bad row (a deleted page, a
        # transient network blip) shouldn't take down every other
        # event in the same batch.
        def flaky_fetch(url, params=None, use_cache=True):
            if params["titles"] == "Good Event":
                return json.dumps(
                    {"query": {"pages": {"1": {"pageid": 1, "title": "Good Event"}}}}
                )
            raise requests.exceptions.ConnectionError("simulated network failure")

        with patch.object(wiki_api, "fetch", side_effect=flaky_fetch):
            events = wiki_api.resolve_event_pageids(
                [{"event_title": "Good Event"}, {"event_title": "Bad Event"}]
            )

        assert events[0]["pageid"] == 1
        assert events[1]["pageid"] is None
        assert events[1]["resolved_title"] is None
