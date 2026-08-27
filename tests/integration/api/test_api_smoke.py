"""
tests/integration/api/test_api_smoke.py

Contract tests for the Week 4 Monday FastAPI skeleton.

These assert on STATUS CODES and RESPONSE SHAPE, never on specific
fighter names or event names. The upcoming-events window changes every
week; a test that asserts "Volkanovski is on the card" is a test that
fails next Tuesday for no useful reason.

Requires TEST_DATABASE_URL (see conftest.py in this folder). Skipped
automatically otherwise.
"""

import pytest
from sqlalchemy import text


class TestHealth:
    def test_health_returns_ok_with_a_model_version(self, api_client):
        # If lifespan wired correctly, both dependencies are up and the
        # active model version is reported. A 503 here means the app
        # booted but something it needs is unreachable.
        response = api_client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["database_reachable"] is True
        assert body["model_loaded"] is True
        assert body["active_model_version"]


class TestUpcomingEvents:
    def test_default_window_returns_a_valid_envelope(self, api_client):
        response = api_client.get("/events/upcoming")

        assert response.status_code == 200
        body = response.json()
        assert body["weeks"] == 4
        assert body["event_count"] == len(body["events"])
        assert body["window_start"] <= body["window_end"]

    def test_every_returned_bout_is_scheduled_and_inside_the_window(self, api_client):
        # The endpoint promises "upcoming". Two things must hold: nothing
        # already finished leaks in, and nothing outside the requested
        # date range does either.
        response = api_client.get("/events/upcoming?weeks=4")
        body = response.json()

        for event in body["events"]:
            assert body["window_start"] <= event["event_date"] <= body["window_end"]
            for bout in event["bouts"]:
                assert bout["status"] == "scheduled"

    @pytest.mark.parametrize("weeks", [0, -1, 13, 999])
    def test_out_of_range_weeks_is_rejected(self, api_client, weeks):
        # Bounded by Query(ge=1, le=12) so no request can ask Postgres
        # for an unbounded date scan. FastAPI rejects before the handler
        # body ever runs.
        assert api_client.get(f"/events/upcoming?weeks={weeks}").status_code == 422

    def test_non_numeric_weeks_is_rejected(self, api_client):
        assert api_client.get("/events/upcoming?weeks=four").status_code == 422


class TestPredictionHistory:
    def test_returns_a_paginated_envelope(self, api_client):
        response = api_client.get("/predictions/history?limit=5")

        assert response.status_code == 200
        body = response.json()
        assert body["limit"] == 5
        assert body["returned"] == len(body["items"])
        assert body["returned"] <= 5

    def test_unsettled_predictions_are_included(self, api_client):
        # A prediction with no prediction_results row yet must still be
        # returned, with settled=False. Filtering those out would turn
        # the public track record into marketing.
        body = api_client.get("/predictions/history?limit=50").json()

        for item in body["items"]:
            assert isinstance(item["settled"], bool)
            if not item["settled"]:
                assert item["correct"] is None
                assert item["settled_at"] is None

    @pytest.mark.parametrize("limit", [0, 201, 5000])
    def test_out_of_range_limit_is_rejected(self, api_client, limit):
        assert api_client.get(f"/predictions/history?limit={limit}").status_code == 422

    def test_negative_offset_is_rejected(self, api_client):
        assert api_client.get("/predictions/history?offset=-1").status_code == 422


class TestFightPrediction:
    def test_unknown_bout_returns_404(self, api_client):
        response = api_client.get("/fights/999999999/prediction")

        assert response.status_code == 404
        assert "detail" in response.json()

    def test_bout_with_no_prediction_returns_501(
        self, api_client, sample_event, sample_fighters
    ):
        # A real bout that has never been predicted must return 501, not
        # a placeholder probability. This is the guard against a fake
        # 0.5 quietly shipping to production.
        red_id, blue_id, _ = sample_fighters
        # Reuse the integration suite's throwaway event/fighters rather
        # than inventing new ones -- cleanup is already handled there.
        with api_client.app.state.engine.begin() as conn:
            bout_id = conn.execute(
                text("""
                    INSERT INTO bouts (event_id, fighter_red_id, fighter_blue_id,
                                       weight_class, scheduled_rounds, status)
                    VALUES (:event_id, :red, :blue, 'Lightweight', 3, 'scheduled')
                    RETURNING id
                """),
                {"event_id": sample_event, "red": red_id, "blue": blue_id},
            ).scalar_one()

        response = api_client.get(f"/fights/{bout_id}/prediction")

        assert response.status_code == 501


class TestOpenAPI:
    def test_every_planned_route_is_registered(self, api_client):
        # Cheap regression guard: if a router stops being included in
        # create_app(), this catches it immediately.
        paths = api_client.get("/openapi.json").json()["paths"]

        assert "/health" in paths
        assert "/events/upcoming" in paths
        assert "/fights/{bout_id}/prediction" in paths
        assert "/predictions/history" in paths
        assert "/model/performance" in paths


class TestModelPerformance:
    def test_serves_only_the_shipping_artifact(self, api_client):
        response = api_client.get("/model/performance")

        assert response.status_code == 200
        body = response.json()
        assert body["shipping_artifact"] == "B"
        assert body["feature_count"] == 32
        assert len(body["test_metrics"]) > 0
        for row in body["test_metrics"]:
            assert row["who"] in {"model", "market"}