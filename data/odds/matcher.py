"""
Resolves Odds API fighter name strings against the fighters table.

Unlike the internal Greco matching in data/ingestion/transform.py (which
leans on a shared source_url), the Odds API shares no identifier scheme
at all -- every name is a freeform string ("Conor McGregor") with nothing
to key against except the string itself. Fuzzy matching is the only
option here, not a fallback for edge cases.
"""
from rapidfuzz import fuzz, process

# Below this score (0-100), a fuzzy match is not auto-accepted -- logged
# for manual review instead of guessed. Kept conservative: fighter names
# are short (2-4 words), so a wrong high-scoring match is easy to produce
# by accident, and a bad auto-match here would silently attach the wrong
# fighter's odds to a bout.
FUZZY_MATCH_THRESHOLD = 90

def resolve_fighter_name(
    name: str,
    real_name_lookup: dict[str, int],
    alias_lookup: dict[str, int],
) -> tuple[int | None, str]:
    """
    Resolve a raw Odds API fighter name string to a fighter_id.

    Returns (fighter_id_or_None, method):
    - 'alias'      -> matched a previously-confirmed alias (no fuzzy work needed)
    - 'exact'      -> matches fighters.real_name exactly
    - 'fuzzy'      -> matched above threshold, NOT yet a confirmed alias --
                      still routed to manual review before being trusted,
                      same discipline as the Greco name-collision work
    - 'unresolved' -> below threshold or no candidates at all
    """
    name = name.strip()

    if name in alias_lookup:
        return alias_lookup[name], "alias"

    if name in real_name_lookup:
        return real_name_lookup[name], "exact"

    match = process.extractOne(name, real_name_lookup.keys(), scorer=fuzz.WRatio)
    if match is not None:
        matched_name, score, _ = match
        if score >= FUZZY_MATCH_THRESHOLD:
            return real_name_lookup[matched_name], "fuzzy"

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


