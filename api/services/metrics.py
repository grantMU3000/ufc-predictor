"""
Reshaping frozen model metrics for the API — Week 4 Monday.

Live, since-deployment metrics live in api/services/live_metrics.py (ADR-026).

model_registry.metrics holds the raw JSON list written by
models/freeze.py: one entry per (artifact, slice, who) combination.
That includes artifact "A", which ADR-020 kept only as a DIAGNOSTIC and
which docs/MODEL_CARD.md explicitly does not headline.

Serving both artifacts through the API would invite exactly the mistake
the model card exists to prevent — someone quoting whichever number
looks best. So the API exposes the shipping artifact only, and the
filtering lives in this pure function where it can be unit-tested
without a database.
"""

from typing import Any

SHIPPING_ARTIFACT = "B"

# Order the frontend will display them in. Anything not listed sorts last.
_SLICE_ORDER = {"full": 0, "odds_covered": 1, "close": 2}
_WHO_ORDER = {"model": 0, "market": 1}


def select_shipping_metrics(
    metrics: list[dict[str, Any]] | None,
    artifact: str = SHIPPING_ARTIFACT,
) -> list[dict[str, Any]]:
    """
    Keep only the rows for the shipping artifact, sorted for display.

    Market rows are kept alongside model rows on purpose: the honest
    framing of this project is "here is our number AND here is the
    closing line's number on the same fights". Shipping one without the
    other would misrepresent the result.

    Parameters
    ----------
    metrics : the raw list from model_registry.metrics, or None.
    artifact : which artifact to keep. Defaults to the shipping one.

    Returns
    -------
    A new sorted list. The input is never mutated.
    """
    if not metrics:
        return []

    kept = [m for m in metrics if m.get("artifact") == artifact]
    return sorted(
        kept,
        key=lambda m: (
            _SLICE_ORDER.get(str(m.get("slice")), 99),
            _WHO_ORDER.get(str(m.get("who")), 99),
        ),
    )