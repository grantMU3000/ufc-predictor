"""
Fighter identity resolution for Wikipedia-sourced upcoming events.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import text

FUZZY_MATCH_THRESHOLD = 90
LOG_DIR = Path("data/ingestion/logs")


def _normalize(name: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace — for comparison only, never stored."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _blocking_key(normalized_name: str) -> str:
    """First letter of the last name-token — narrows fuzzy-match candidates before scoring."""
    tokens = normalized_name.split()
    return tokens[-1][0] if tokens else ""


@dataclass
class FighterRoster:
    """
    Fighters table + fighter_aliases loaded once per ingestion run, not
    once per fighter — the whole roster is ~4,500 rows, small enough to
    hold in memory and cheap enough to load once, versus a DB round-trip
    per name on a card of ~20 fighters.
    """

    by_id: dict[int, str]  # fighter_id -> real_name
    name_index: dict[str, list[int]]  # normalized real_name -> [fighter_id, ...]
    alias_index: dict[str, int]  # normalized alias -> fighter_id

    @classmethod
    def load(cls, engine) -> "FighterRoster":
        """Loaded via pandas, matching how the rest of the pipeline reads from Postgres."""
        fighters_df = pd.read_sql("SELECT id, real_name FROM fighters", engine)
        aliases_df = pd.read_sql(
            "SELECT fighter_id, alias_name FROM fighter_aliases", engine
        )

        by_id = dict(zip(fighters_df["id"], fighters_df["real_name"]))
        name_index: dict[str, list[int]] = {}
        for fid, name in by_id.items():
            name_index.setdefault(_normalize(name), []).append(fid)

        alias_index: dict[str, int] = {}
        for alias_name, fighter_id in zip(
            aliases_df["alias_name"], aliases_df["fighter_id"]
        ):
            norm = _normalize(alias_name)
            if norm in alias_index and alias_index[norm] != fighter_id:
                _log(
                    "alias_collisions.jsonl",
                    {
                        "alias_name": alias_name,
                        "existing_fighter_id": alias_index[norm],
                        "conflicting_fighter_id": fighter_id,
                    },
                )
                continue  # keep the first-seen mapping, don't silently overwrite
            alias_index[norm] = fighter_id

        return cls(by_id=by_id, name_index=name_index, alias_index=alias_index)


@dataclass
class ResolvedFighter:
    fighter_id: int | None
    match_type: str  # "alias" | "exact" | "fuzzy" | "collision" | "unresolved"


def _log(filename: str, record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _add_alias(engine, fighter_id: int, alias_name: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO fighter_aliases (fighter_id, alias_name)
                VALUES (:fighter_id, :alias_name)
                ON CONFLICT (fighter_id, alias_name) DO NOTHING
            """),
            {"fighter_id": fighter_id, "alias_name": alias_name},
        )


def resolve_fighter(
    engine,
    roster: FighterRoster,
    display_name: str,
    wikipedia_link_target: str | None = None,
) -> ResolvedFighter:
    """
    Resolve a Wikipedia-sourced fighter name to a fighter_id.

    display_name: the text shown on the page (e.g. "Islam Makhachev")
    wikipedia_link_target: the wikilink target if the name was linked
        (e.g. "Islam Makhachev", or a disambiguated title like
        "Bruno Silva (welterweight)") — checked first since it's a
        stronger identity signal than plain display text.
    """
    # 1. Alias lookup — check the link target first (stronger signal), then display name
    for candidate_key in filter(None, [wikipedia_link_target, display_name]):
        norm = _normalize(candidate_key)
        if norm in roster.alias_index:
            return ResolvedFighter(roster.alias_index[norm], "alias")

    # 2. Exact match on real_name
    norm_display = _normalize(display_name)
    exact_matches = roster.name_index.get(norm_display, [])

    if len(exact_matches) == 1:
        fighter_id = exact_matches[0]
        if wikipedia_link_target:
            _add_alias(engine, fighter_id, wikipedia_link_target)
        return ResolvedFighter(fighter_id, "exact")

    if len(exact_matches) > 1:
        # Real collision (e.g. two "Bruno Silva"s) and no wikilink to disambiguate —
        # don't guess. Route to the same manual-review process as Greco's collisions.
        _log(
            "fighter_collisions.jsonl",
            {
                "display_name": display_name,
                "wikipedia_link_target": wikipedia_link_target,
                "candidate_fighter_ids": exact_matches,
            },
        )
        return ResolvedFighter(None, "collision")

    # 3. Blocked fuzzy match
    block_key = _blocking_key(norm_display)
    candidate_pool = {
        fid: _normalize(name)
        for fid, name in roster.by_id.items()
        if _blocking_key(_normalize(name)) == block_key
    }
    if candidate_pool:
        match = process.extractOne(
            norm_display,
            candidate_pool,
            scorer=fuzz.WRatio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if match is not None:
            _, _, fighter_id = match
            _add_alias(engine, fighter_id, display_name)
            if wikipedia_link_target:
                _add_alias(engine, fighter_id, wikipedia_link_target)
            return ResolvedFighter(fighter_id, "fuzzy")

    # 4. Nothing found — log for manual review, caller decides whether to stub
    _log(
        "unresolved_fighters.jsonl",
        {
            "display_name": display_name,
            "wikipedia_link_target": wikipedia_link_target,
        },
    )
    return ResolvedFighter(None, "unresolved")


def create_stub_fighter(engine, roster: FighterRoster, display_name: str) -> int:
    """
    Insert a minimal fighter row for a name with no match — only real_name
    is known. source_url is intentionally left NULL here: per the earlier
    decision, an unverified ufcstats guess is worse than no link at all,
    since it would silently misattribute a future Greco fighter's real
    stats to the wrong row. Verified links get added manually later via
    the same override-dict process used for Greco's name collisions.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO fighters (real_name) VALUES (:name) RETURNING id"),
            {"name": display_name},
        )
        fighter_id = result.scalar_one()

    _add_alias(engine, fighter_id, display_name)

    # Keep the in-memory roster in sync so a second fighter with the same
    # stub name later in the same run is treated as a duplicate name
    # (routed to collision handling), not silently given a second stub.
    roster.by_id[fighter_id] = display_name
    roster.name_index.setdefault(_normalize(display_name), []).append(fighter_id)

    return fighter_id
