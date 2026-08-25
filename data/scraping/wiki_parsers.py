import re
from datetime import date, datetime

import mwparserfromhell

EXPECTED_HEADERS = ["Event", "Date", "Venue", "Location", "Ref."]

# A cell's attribute prefix (e.g. `rowspan=2 ` or `style="background:#fcc" `)
# is plain formatting text — no wikilinks, no templates. If the text before
# the first top-level pipe doesn't look like this, it's content, not attributes.
_ATTRIBUTE_TEXT_RE = re.compile(r"""^[\w\s\-=:;%#.,'"()/]*$""")
_ATTRIBUTE_PAIR_RE = re.compile(r"""(\w+)\s*=\s*"?([^"\s|]+)"?""")

def _first_wikilink(cell_wikicode) -> dict | None:
    """Return {'target': ..., 'display': ...} for the first wikilink in a cell, or None."""
    links = cell_wikicode.filter_wikilinks()
    if not links:
        return None
    link = links[0]
    target = str(link.title).strip()
    display = str(link.text).strip() if link.text is not None else target
    return {"target": target, "display": display}

def _parse_dts(cell_wikicode) -> date | None:
    """Extract a date from a {{dts|YYYY|Mon|DD}} template."""
    for template in cell_wikicode.filter_templates():
        if template.name.strip().lower() != "dts":
            continue
        params = [str(p.value).strip() for p in template.params if not p.showkey]
        if len(params) < 3:
            continue
        year, month, day = params[:3]
        for fmt in ("%Y %b %d", "%Y %B %d", "%Y %m %d"):
            try:
                # Calendar date only, no time component -- tzinfo doesn't apply.
                return datetime.strptime(f"{year} {month} {day}", fmt).date()  # noqa: DTZ007
            except ValueError:
                continue
    return None

def _top_level_pipes(text: str) -> list[tuple[int, bool]]:
    """
    Find pipe positions that belong to the *table* rather than to a wikilink
    or template nested inside a cell.

    `[[Las Vegas|Nevada]]` and `{{dts|2026|Nov|07}}` both contain pipes that
    are not cell boundaries. Scanning with a nesting depth counter is the
    difference between splitting a cell correctly and shredding a piped link.

    Returns (index, is_double) pairs; `||` separates cells on one line,
    a single `|` separates a cell's attributes from its content.
    """
    depth = 0
    i = 0
    found: list[tuple[int, bool]] = []
    while i < len(text):
        pair = text[i : i + 2]
        if pair in ("[[", "{{", "{|"):
            depth += 1
            i += 2
            continue
        if pair in ("]]", "}}", "|}"):
            depth = max(depth - 1, 0)
            i += 2
            continue
        if text[i] == "|" and depth == 0:
            is_double = text[i + 1 : i + 2] == "|"
            found.append((i, is_double))
            i += 2 if is_double else 1
            continue
        i += 1
    return found


def _split_cell(cell_text: str) -> tuple[dict[str, str], str]:
    """
    Split one cell into ({attribute: value}, content).

    MediaWiki's rule: if a cell contains a top-level single pipe, everything
    before it is HTML attributes. Without this step, `rowspan=2 | Meta Apex`
    gets stored as a venue name, which is exactly what happened on the
    UFC Fight Night 292/293 Apex pair.
    """
    single_pipes = [
        pos for pos, is_double in _top_level_pipes(cell_text) if not is_double
    ]
    if not single_pipes:
        return {}, cell_text.strip()

    split_at = single_pipes[0]
    attribute_text, content = cell_text[:split_at], cell_text[split_at + 1 :]

    # Safety net: only treat the prefix as attributes if it actually looks
    # like `name=value` formatting. Otherwise keep the cell intact rather
    # than silently truncating real content.
    if "=" not in attribute_text or not _ATTRIBUTE_TEXT_RE.match(attribute_text):
        return {}, cell_text.strip()

    attributes = {
        name.lower(): value
        for name, value in _ATTRIBUTE_PAIR_RE.findall(attribute_text)
    }
    return attributes, content.strip()


def _row_cells(row_block: str) -> list[tuple[dict[str, str], str]]:
    """
    Extract the cells physically present in one row block.

    Handles both wikitext cell styles: one cell per line (`| Foo`) and
    several cells on one line (`| Foo || Bar || Baz`).
    """
    cells: list[tuple[dict[str, str], str]] = []
    for raw_line in row_block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith(("|}", "|-", "|+")):
            continue
        body = line[1:]

        boundaries = [pos for pos, is_double in _top_level_pipes(body) if is_double]
        start = 0
        chunks = []
        for pos in boundaries:
            chunks.append(body[start:pos])
            start = pos + 2
        chunks.append(body[start:])

        cells.extend(_split_cell(chunk) for chunk in chunks)
    return cells


def _expand_spans(
    rows: list[list[tuple[dict[str, str], str]]], n_cols: int
) -> list[list[str]]:
    """
    Turn physically-present cells into a full rectangular grid by carrying
    rowspan/colspan cells forward.

    A cell with `rowspan=2` in column 2 fills column 2 of the next row too.
    Tracking the *column index* (not just "the last value seen") matters:
    the row below supplies its own Event and Date cells, so the carried
    Venue has to slot into position 2, not be appended at the end.
    """
    pending: dict[int, list] = {}
    expanded: list[list[str]] = []

    for cells in rows:
        remaining = list(cells)
        row: list[str] = []
        col = 0
        while col < n_cols:
            carried = pending.get(col)
            if carried is not None:
                row.append(carried[0])
                carried[1] -= 1
                if carried[1] == 0:
                    del pending[col]
                col += 1
                continue

            if not remaining:
                break

            attributes, content = remaining.pop(0)
            try:
                row_span = int(attributes.get("rowspan", 1))
                col_span = int(attributes.get("colspan", 1))
            except ValueError:
                row_span = col_span = 1

            for offset in range(max(col_span, 1)):
                if col + offset >= n_cols:
                    break
                if row_span > 1:
                    pending[col + offset] = [content, row_span - 1]
                row.append(content)
            col += max(col_span, 1)

        expanded.append(row)

    return expanded


def parse_scheduled_events(wikitext: str) -> list[dict]:
    """
    Parse the "Scheduled events" wikitable (List_of_UFC_events) into a list
    of event dicts: event_title (page title — use for pageid lookups),
    event_display_name, date, venue, location.

    Rows are expanded through _expand_spans() first, so back-to-back Apex
    cards that share a merged Venue/Location cell each get their own
    complete row. A row that still can't be reconstructed raises rather
    than being skipped — a silently dropped event is a missing prediction,
    and there's nothing downstream that would catch it.
    """
    table_match = re.search(r"\{\|.*?\n\|\}", wikitext, re.DOTALL)
    if not table_match:
        raise ValueError("No wikitable found in 'Scheduled events' wikitext")
    table_text = table_match.group(0)

    header_block, *row_blocks = re.split(r"\n\|-\s*\n?", table_text)
    header_cells = []
    for line in re.findall(r"^!\s*(.+)$", header_block, re.MULTILINE):
        header_cells.extend(line.split("!!"))
    headers = [_split_cell(cell)[1] for cell in header_cells]
    if headers != EXPECTED_HEADERS:
        raise ValueError(f"Unexpected table columns: {headers}")

    raw_rows = [_row_cells(block) for block in row_blocks]
    raw_rows = [cells for cells in raw_rows if cells]
    grid = _expand_spans(raw_rows, len(EXPECTED_HEADERS))

    events = []
    for row_number, row in enumerate(grid, start=1):
        if len(row) < 4:
            raise ValueError(
                f"Scheduled-events row {row_number} reconstructed to {len(row)} "
                f"columns, expected at least 4. Row content: {row!r}"
            )

        event_link = _first_wikilink(mwparserfromhell.parse(row[0]))
        if event_link is None:
            print(f"Skipping scheduled-events row {row_number}: no linked event page ({row[0]!r})")
            continue

        events.append(
            {
                "event_title": event_link["target"],
                "event_display_name": event_link["display"],
                "date": _parse_dts(mwparserfromhell.parse(row[1])),
                "venue": mwparserfromhell.parse(row[2]).strip_code().strip(),
                "location": mwparserfromhell.parse(row[3]).strip_code().strip(),
            }
        )

    return events


def _get_param(template, index: int) -> str:
    """Extract and clean positional param `index` (1-based) from a template, or '' if absent."""
    key = str(index)
    if template.has(key):
        value = str(template.get(key).value)
        return mwparserfromhell.parse(value).strip_code().strip()
    return ""


def _get_param_and_link(template, index: int) -> tuple[str, str | None]:
    """
    Extract positional param `index` (1-based) as (display_text, link_target).
    link_target is the wikilink's target page title if the param is linked
    (e.g. "Islam Makhachev", or a disambiguated title like
    "Bruno Silva (welterweight)"), or None if the fighter's name appears
    as plain unlinked text (e.g. no Wikipedia article yet).
    """
    key = str(index)
    if not template.has(key):
        return "", None

    value_wikicode = template.get(key).value
    links = value_wikicode.filter_wikilinks()
    link_target = str(links[0].title).strip() if links else None

    display_text = mwparserfromhell.parse(str(value_wikicode)).strip_code().strip()
    return display_text, link_target


def parse_fight_card(wikitext: str) -> list[dict]:
    """
    Parses a UFC event page's "Fight card" section wikitext into a list of
    bout dicts, in document order (main card, then prelims, then early prelims).
    """
    parsed = mwparserfromhell.parse(wikitext)
    bouts = []
    current_tier = None

    for template in parsed.filter_templates():
        name = template.name.strip()

        if name == "MMAevent card":
            current_tier = _get_param(template, 1)
            continue

        if name != "MMAevent bout":
            continue

        fighter_red_raw, fighter_red_link = _get_param_and_link(template, 2)
        fighter_blue_raw, fighter_blue_link = _get_param_and_link(template, 4)

        bouts.append(
            {
                "card_tier": current_tier,
                "weight_class": _get_param(template, 1),
                "fighter_red": fighter_red_raw.replace("(c)", "").strip(),
                "fighter_red_link_target": fighter_red_link,
                "fighter_red_is_champion": "(c)" in fighter_red_raw,
                "connector": _get_param(
                    template, 3
                ),  # "vs." = not yet fought, "def." = result recorded
                "fighter_blue": fighter_blue_raw.replace("(c)", "").strip(),
                "fighter_blue_link_target": fighter_blue_link,
                "fighter_blue_is_champion": "(c)" in fighter_blue_raw,
                "method": _get_param(template, 5),
                "round": _get_param(template, 6),
                "time": _get_param(template, 7),
                "notes": _get_param(template, 8),
            }
        )

    return bouts
