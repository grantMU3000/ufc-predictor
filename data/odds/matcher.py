"""
Resolves Odds API fighter name strings against the fighters table.

Unlike the internal Greco matching in data/ingestion/transform.py (which
leans on a shared source_url), the Odds API shares no identifier scheme
at all -- every name is a freeform string ("Conor McGregor") with nothing
to key against except the string itself. Fuzzy matching is the only
option here, not a fallback for edge cases.
"""
from rapidfuzz import fuzz, process
from datetime import date, datetime, timedelta, timezone

# Below this score (0-100), a fuzzy match is not auto-accepted -- logged
# for manual review instead of guessed. Kept conservative: fighter names
# are short (2-4 words), so a wrong high-scoring match is easy to produce
# by accident, and a bad auto-match here would silently attach the wrong
# fighter's odds to a bout.
FUZZY_MATCH_THRESHOLD = 90

def resolve_fighter_name(
    name: str,
    real_name_lookup: dict[str, int | list[int]],
    alias_lookup: dict[str, int],
) -> tuple[int | list[int] | None, str]:
    """
    Resolve a raw Odds API fighter name string to a fighter_id.

    Returns (result, method):
    - 'alias'           -> result is a single int (confirmed alias match)
    - 'exact'           -> result is a single int (unambiguous real_name match)
    - 'exact_ambiguous' -> result is a list[int]; name is shared by >1 real
                           fighter (e.g. two "Bruno Silva"s). Caller must
                           disambiguate via bout-matching, and must NEVER
                           write this raw name as a new alias -- the string
                           itself is permanently ambiguous, not a one-off.
    - 'fuzzy'           -> result is a single int, safe to confirm as an alias
    - 'fuzzy_ambiguous' -> result is a list[int]; the fuzzy match landed on
                           an ambiguous real_name. Same caution as
                           exact_ambiguous -- a variant spelling of an
                           ambiguous name inherits that same ambiguity.
    - 'unresolved'      -> result is None
    """
    name = name.strip()

    if name in alias_lookup:
        return alias_lookup[name], "alias"

    if name in real_name_lookup:
        value = real_name_lookup[name]
        if isinstance(value, list):
            return value, "exact_ambiguous"
        return value, "exact"

    match = process.extractOne(name, real_name_lookup.keys(), scorer=fuzz.WRatio)
    if match is not None:
        matched_name, score, _ = match
        if score >= FUZZY_MATCH_THRESHOLD:
            value = real_name_lookup[matched_name]
            if isinstance(value, list):
                return value, "fuzzy_ambiguous"
            return value, "fuzzy"

    return None, "unresolved"

def resolve_odds_entries(
    odds_entries: list[dict],
    real_name_lookup: dict[str, int],
    alias_lookup: dict[str, int],
) -> list[dict]:
    """
    Resolve every home_team/away_team across a list of odds API entries.
    One row per (entry, side) -- doesn't touch bout-matching, that's a
    separate step once fighter_ids are resolved.
    """
    results = []
    for entry in odds_entries:
        for side, name in [("home", entry["home_team"]), ("away", entry["away_team"])]:
            fighter_id, method = resolve_fighter_name(name, real_name_lookup, alias_lookup)
            results.append({
                "commence_time": entry["commence_time"],
                "side": side,
                "raw_name": name,
                "fighter_id": fighter_id,
                "match_method": method,
            })
    return results

def build_bout_lookup(bouts_events_rows: list[dict]) -> dict[frozenset, list[dict]]:
    """
    bouts_events_rows: [{bout_id, fighter_red_id, fighter_blue_id, event_date}, ...]
    pulled from a bouts JOIN events query.

    Keyed on frozenset({fighter_a, fighter_b}) since the Odds API's
    home/away order has no relationship to red/blue assignment.
    Value is a list, not a single bout, to hold multiple entries for
    fighter pairs who've rematched.
    """
    lookup: dict[frozenset, list[dict]] = {}
    for row in bouts_events_rows:
        key = frozenset({row["fighter_red_id"], row["fighter_blue_id"]})
        lookup.setdefault(key, []).append(row)
    return lookup


def match_to_bout(
    fighter_a_id: int, fighter_b_id: int, commence_time: str,
    bout_lookup: dict[frozenset, list[dict]], max_days_diff: int = 2,
) -> int | None:
    """
    Resolve a resolved fighter pair + commence_time to a real bout_id.
    Returns None if no bout exists for this pair at all (catches
    coincidental name matches like Cejudo/Dvalishvili -- real roster
    names, but not an actual logged bout together), or if the closest
    candidate is outside max_days_diff (catches a wrong rematch pick).
    """
    key = frozenset({fighter_a_id, fighter_b_id})
    candidates = bout_lookup.get(key)
    if not candidates:
        return None

    commence_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
    best = min(candidates, key=lambda c: abs((c["event_date"] - commence_date).days))

    if abs((best["event_date"] - commence_date).days) > max_days_diff:
        return None
    return best["bout_id"]
