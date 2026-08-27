"""
tests/test_api_metrics.py

Offline unit tests for api/services/metrics.py.

ELI5: model_registry.metrics is a big pile of scorecards. Some of them
are for the model we actually shipped ("artifact B") and some are for a
diagnostic variant we deliberately do NOT publish ("artifact A"). This
file checks that the API hands out only the shipped scorecards, in a
sensible order, and does not fall over when there are no scorecards at
all.

No database, no FastAPI app, no model file — this tests a pure function,
so it runs anywhere in milliseconds. The endpoint that uses it is
covered separately in tests/integration/api/.
"""

from api.services.metrics import SHIPPING_ARTIFACT, select_shipping_metrics


def _row(artifact: str, slice_name: str, who: str) -> dict:
    """A minimal metrics entry with only the fields the filter reads."""
    return {"artifact": artifact, "slice": slice_name, "who": who}


class TestSelectShippingMetrics:
    def test_drops_the_diagnostic_artifact(self):
        # Artifact A exists in the frozen metadata as a diagnostic only
        # (ADR-020). It must never reach the API response, or someone
        # will eventually quote whichever number looks better.
        metrics = [
            _row("A", "full", "model"),
            _row("B", "full", "model"),
        ]

        result = select_shipping_metrics(metrics)

        assert len(result) == 1
        assert result[0]["artifact"] == SHIPPING_ARTIFACT

    def test_keeps_market_rows_alongside_model_rows(self):
        # The market baseline travels WITH the model's numbers. A model
        # number without its baseline is not a claim, it's a decoration.
        metrics = [
            _row("B", "odds_covered", "market"),
            _row("B", "odds_covered", "model"),
        ]

        result = select_shipping_metrics(metrics)

        assert {r["who"] for r in result} == {"model", "market"}

    def test_orders_slices_full_then_odds_covered_then_close(self):
        metrics = [
            _row("B", "close", "model"),
            _row("B", "full", "model"),
            _row("B", "odds_covered", "model"),
        ]

        result = select_shipping_metrics(metrics)

        assert [r["slice"] for r in result] == ["full", "odds_covered", "close"]

    def test_model_sorts_before_market_within_a_slice(self):
        metrics = [
            _row("B", "close", "market"),
            _row("B", "close", "model"),
        ]

        result = select_shipping_metrics(metrics)

        assert [r["who"] for r in result] == ["model", "market"]

    def test_handles_none_and_empty(self):
        # model_registry.metrics is nullable. A freshly registered model
        # with no metrics yet must not 500 the /model/performance route.
        assert select_shipping_metrics(None) == []
        assert select_shipping_metrics([]) == []

    def test_does_not_mutate_the_input(self):
        metrics = [_row("B", "close", "model"), _row("A", "full", "model")]
        original = [dict(m) for m in metrics]

        select_shipping_metrics(metrics)

        assert metrics == original

    def test_unknown_slice_sorts_last_instead_of_raising(self):
        # If a future model version adds a new slice name, the endpoint
        # should still serve it rather than crash on an unknown key.
        metrics = [_row("B", "brand_new_slice", "model"), _row("B", "full", "model")]

        result = select_shipping_metrics(metrics)

        assert [r["slice"] for r in result] == ["full", "brand_new_slice"]