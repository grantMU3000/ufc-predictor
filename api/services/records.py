"""
Flattens a BoutPrediction into a JSON-serializable dict — shared by
scripts/score_upcoming.py (the JSON dump) and api/services/ledger.py
(the feature_snapshot envelope).

WHY ITS OWN FILE: to_record() used to live inside
scripts/score_upcoming.py on the theory that "the ledger writer needs
the same flattening." True, but the consequence wasn't handled:
api/services/ledger.py importing FROM scripts/score_upcoming.py,
while score_upcoming.py's --write path needs to import FROM
api/services/ledger.py, is a circular import. Moving the shared piece
to a leaf module that depends on neither side is the fix.
"""

from dataclasses import asdict
from datetime import UTC, datetime

from api.services.inference import BoutPrediction


def to_record(prediction: BoutPrediction) -> dict:
    """
    Flatten a BoutPrediction into a JSON-serializable dict.

    One shape, defined once — so the JSON file the script writes and
    the feature_snapshot the ledger inserts can never quietly drift
    into describing the same prediction two different ways.
    """
    record = asdict(prediction)
    record["coverage"]["fraction"] = prediction.coverage.fraction
    record["scored_at"] = datetime.now(UTC).isoformat()
    return record