# data/ingestion/parsers.py
import pandas as pd
from pathlib import Path
import re

RAW_DATA_DIR = Path("data/raw")


def read_events() -> pd.DataFrame:
    """
    Read and lightly clean ufc_events.csv.

    Column names/logic below are a placeholder until confirmed against the
    real file — don't trust the specifics yet, the structure (read, clean
    column names, strip strings) is what matters at this stage.
    """
    path = RAW_DATA_DIR / "ufc_events.csv"
    df = pd.read_csv(path)

    # Rename directly to your schema's column names now, rather than doing
    # a generic snake_case pass — since we already know the real headers,
    # there's no reason to add a translation step later in transform.py.
    df = df.rename(columns={
        "EVENT": "name",
        "URL": "source_url",
        "DATE": "event_date",
        "LOCATION": "location",
    })

    # Stripping whitespace to avoid future mismatches
    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    # Parse "August 01, 2026" into an actual date type. 
    df["event_date"] = pd.to_datetime(
        df["event_date"], format="%B %d, %Y"
    ).dt.date

    return df


def _parse_height_to_cm(value: str) -> float | None:
    """Convert "5' 11\"" to centimeters. Returns None for missing ('--')."""
    if not isinstance(value, str) or value.strip() == "--":
        return None
    match = re.match(r"(\d+)'\s*(\d+)\"", value.strip())
    if not match:
        return None
    feet, inches = int(match.group(1)), int(match.group(2))
    total_inches = feet * 12 + inches
    return round(total_inches * 2.54, 1)


def _parse_reach_to_cm(value: str) -> float | None:
    """Convert '66"' to centimeters. Returns None for missing ('--')."""
    if not isinstance(value, str) or value.strip() == "--":
        return None
    match = re.match(r'(\d+)"', value.strip())
    if not match:
        return None
    inches = int(match.group(1))
    return round(inches * 2.54, 1)


def _parse_stance(value) -> str | None:
    """Lowercase and clean stance; leaves genuinely missing values as None."""
    if not isinstance(value, str) or value.strip() == "":
        return None
    return value.strip().lower()


def read_fighter_details() -> pd.DataFrame:
    """
    Read and clean ufc_fighter_details.csv.

    Source columns: FIRST, LAST, NICKNAME, URL
    """
    path = RAW_DATA_DIR / "ufc_fighter_details.csv"
    df = pd.read_csv(path)

    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    # Combine first/last into the single real_name field schema expects.
    df["real_name"] = df["FIRST"].fillna("") + " " + df["LAST"].fillna("")
    df["real_name"] = df["real_name"].str.strip()

    df = df.rename(columns={"URL": "source_url", "NICKNAME": "nickname"})

    return df[["source_url", "real_name", "nickname"]]


def read_fighter_tott() -> pd.DataFrame:
    """
    Read and clean ufc_fighter_tott.csv ("tale of the tape" — physicals).

    Source columns: FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL
    WEIGHT is intentionally dropped — per ADR, weight isn't stored on
    `fighters` (it's fight-specific, not a fixed fighter attribute), and
    this source doesn't provide per-bout weigh-ins either, so there's no
    home for it in the current schema.
    """
    path = RAW_DATA_DIR / "ufc_fighter_tott.csv"
    df = pd.read_csv(path)

    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    df["height_cm"] = df["HEIGHT"].apply(_parse_height_to_cm)
    df["reach_cm"] = df["REACH"].apply(_parse_reach_to_cm)
    df["stance"] = df["STANCE"].apply(_parse_stance)
    df["dob"] = pd.to_datetime(df["DOB"], format="%b %d, %Y", errors="coerce").dt.date

    df = df.rename(columns={"URL": "source_url"})

    return df[["source_url", "height_cm", "reach_cm", "stance", "dob"]]


def read_fighters() -> pd.DataFrame:
    """
    Join fighter_details + fighter_tott on fighter_url into one fighters-
    shaped DataFrame.

    Note: `nationality` and `ufc_debut_date` are NOT populated here.
    Neither source CSV provides nationality. `ufc_debut_date` can't be
    derived from fighter data alone — it requires finding each fighter's
    earliest bout, which means it has to be computed later in transform.py
    once the bouts table is assembled, not in this read step.
    """
    details = read_fighter_details()
    tott = read_fighter_tott()

    fighters = details.merge(tott, on="fighter_url", how="left")

    return fighters

def _parse_landed_attempted(value: str) -> tuple[int | None, int | None]:
    """Convert '11 of 14' into (11, 14). Returns (None, None) if unparseable."""
    if not isinstance(value, str):
        return (None, None)
    match = re.match(r"(\d+)\s+of\s+(\d+)", value.strip())
    if not match:
        return (None, None)
    return int(match.group(1)), int(match.group(2))


def _parse_control_time_to_seconds(value: str) -> int | None:
    """Convert '0:04' (mm:ss) into total seconds."""
    if not isinstance(value, str) or ":" not in value:
        return None
    minutes, seconds = value.strip().split(":")
    return int(minutes) * 60 + int(seconds)


def _parse_round_number(value: str) -> int | None:
    """Convert 'Round 1' into 1."""
    if not isinstance(value, str):
        return None
    match = re.match(r"Round\s+(\d+)", value.strip())
    return int(match.group(1)) if match else None


# Every column in the source that follows the "X of Y" landed/attempted
# format, mapped to the (landed_col, attempted_col) names your bout_stats
# schema expects.
LANDED_ATTEMPTED_COLUMNS = {
    "SIG.STR.": ("sig_strikes_landed", "sig_strikes_attempted"),
    "TOTAL STR.": ("total_strikes_landed", "total_strikes_attempted"),
    "TD": ("takedowns_landed", "takedowns_attempted"),
    "HEAD": ("head_strikes_landed", "head_strikes_attempted"),
    "BODY": ("body_strikes_landed", "body_strikes_attempted"),
    "LEG": ("leg_strikes_landed", "leg_strikes_attempted"),
    "DISTANCE": ("distance_strikes_landed", "distance_strikes_attempted"),
    "CLINCH": ("clinch_strikes_landed", "clinch_strikes_attempted"),
    "GROUND": ("ground_strikes_landed", "ground_strikes_attempted"),
}


def read_fight_stats() -> pd.DataFrame:
    """
    Read and clean ufc_fight_stats.csv — round-by-round per-fighter stats.

    Source columns: EVENT, BOUT, ROUND, FIGHTER, KD, SIG.STR., SIG.STR. %,
    TOTAL STR., TD, TD %, SUB.ATT, REV., CTRL, HEAD, BODY, LEG, DISTANCE,
    CLINCH, GROUND

    Dropped: SIG.STR. % and TD % (redundant once landed/attempted are
    parsed out), REV. (not present in the bout_stats schema).

    IMPORTANT: this file has no fighter or bout URL — only free-text EVENT,
    BOUT ("Fighter A vs Fighter B"), and FIGHTER (name string). Resolving
    these rows to fighter_id/bout_id requires matching on real_name and on
    (event, bout matchup) in transform.py — this function only cleans the
    raw values, it does not resolve identity.
    """
    path = RAW_DATA_DIR / "ufc_fight_stats.csv"
    df = pd.read_csv(path)

    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    # Drop rows with no real content (the 42 EVENT/BOUT-only rows) —
    # flagged for a manual look, but excluded here since they carry no
    # usable stats.
    df = df.dropna(subset=["FIGHTER", "ROUND"])

    df["round_number"] = df["ROUND"].apply(_parse_round_number)
    df["control_time_seconds"] = df["CTRL"].apply(_parse_control_time_to_seconds)
    df["knockdowns"] = df["KD"].astype("Int64")
    df["sub_attempts"] = df["SUB.ATT"].astype("Int64")
    df["reversals"] = df["REV."].astype("Int64")

    for source_col, (landed_col, attempted_col) in LANDED_ATTEMPTED_COLUMNS.items():
        parsed = df[source_col].apply(_parse_landed_attempted)
        df[landed_col] = parsed.apply(lambda t: t[0])
        df[attempted_col] = parsed.apply(lambda t: t[1])

    df = df.rename(columns={
        "EVENT": "event_name",
        "BOUT": "bout_matchup",
        "FIGHTER": "fighter_name",
    })

    keep_cols = [
        "event_name", "bout_matchup", "round_number", "fighter_name",
        "knockdowns", "sub_attempts", "reversals", "control_time_seconds",
    ]
    keep_cols += [col for pair in LANDED_ATTEMPTED_COLUMNS.values() for col in pair]

    return df[keep_cols]

def _parse_scheduled_rounds(value: str) -> int | None:
    """
    Extract the base round count from strings like '3 Rnd (5-5-5)' or
    '3 Rnd + OT (5-5-5-5)' -> 3. Returns None for 'No Time Limit' (no
    concept of a round count in that format).
    """
    if not isinstance(value, str):
        return None
    match = re.match(r"(\d+)\s*Rnd", value.strip())
    return int(match.group(1)) if match else None

def _parse_weight_class(value: str) -> str:
    """Strip the trailing 'Bout' / tournament naming noise, keep it simple."""
    if not isinstance(value, str):
        return value
    return value.strip()

def _is_title_fight(weightclass_raw: str) -> bool:
    """Detect title/championship bouts across historical naming variants."""
    if not isinstance(weightclass_raw, str):
        return False
    return bool(re.search(r"title|championship", weightclass_raw, re.IGNORECASE))

VALID_SCHEDULED_ROUNDS = {3, 5}


def read_fight_results() -> pd.DataFrame:
    """
    Read and clean ufc_fight_results.csv — this is the sole source for
    bout-level data; ufc_fight_details.csv is redundant (identical URL
    set, strict column subset) and is intentionally not read.

    Source columns: EVENT, BOUT, OUTCOME, WEIGHTCLASS, METHOD, ROUND,
    TIME, TIME FORMAT, REFEREE, DETAILS, URL

    IMPORTANT: like fight_stats, this file has no fighter URLs — BOUT is
    free text ("Fighter A vs Fighter B"). Resolving to fighter_id happens
    in transform.py via name matching against the fighters table.

    Rows with scheduled_rounds outside {3, 5} (pre-Unified-Rules-era
    fights: 1/2-round or no-time-limit formats) are filtered out here,
    per ck_bouts_scheduled_rounds. This means fight_stats.csv will still
    contain round-by-round rows for some excluded bouts — transform.py
    MUST inner-join bout_stats to this filtered bouts table (on
    event_name + bout matchup), not left-join, so those orphaned stats
    rows are dropped rather than causing an FK violation at insert time.

    Every row here has status='completed', since this file only contains
    fights that have already happened.
    """
    path = RAW_DATA_DIR / "ufc_fight_results.csv"
    df = pd.read_csv(path)

    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

    split_names = df["BOUT"].str.split(" vs. ", n=1, expand=True)
    df["fighter_red_name"] = split_names[0].str.strip()
    df["fighter_blue_name"] = split_names[1].str.strip()

    def _winner_side(outcome: str) -> str | None:
        first, second = outcome.split("/")
        if first == "W":
            return "red"
        elif second == "W":
            return "blue"
        return None  # NC or D — no winner

    df["winner_side"] = df["OUTCOME"].apply(_winner_side)

    df["event_name"] = df["EVENT"]
    df["weight_class"] = df["WEIGHTCLASS"].apply(_parse_weight_class)
    df["is_title_fight"] = df["WEIGHTCLASS"].apply(_is_title_fight)
    df["method"] = df["METHOD"]
    df["method_detail"] = df["DETAILS"]
    df["ending_round"] = df["ROUND"].astype("Int64")
    df["ending_time_seconds"] = df["TIME"].apply(_parse_control_time_to_seconds)
    df["scheduled_rounds"] = df["TIME FORMAT"].apply(_parse_scheduled_rounds)
    df["source_url"] = df["URL"]
    df["status"] = "completed"

    # Filter out pre-Unified-Rules-era formats that violate
    # ck_bouts_scheduled_rounds. Logged loudly, not dropped silently —
    # matches the project's "fail/report loud" data-quality discipline.
    before_count = len(df)
    df = df[df["scheduled_rounds"].isin(VALID_SCHEDULED_ROUNDS)]
    dropped_count = before_count - len(df)
    print(f"read_fight_results: dropped {dropped_count} rows with "
          f"scheduled_rounds outside {VALID_SCHEDULED_ROUNDS} "
          f"({before_count} -> {len(df)})")

    df["bout_matchup"] = df["BOUT"]
    keep_cols = [
        "event_name", "fighter_red_name", "fighter_blue_name", "winner_side",
        "weight_class", "is_title_fight", "method", "method_detail",
        "ending_round", "ending_time_seconds", "scheduled_rounds",
        "source_url", "status", "bout_matchup"
    ]

    return df[keep_cols]

read_fight_results()