from datetime import UTC, date, datetime, timedelta


def filter_by_event_date(
    odds_data: list[dict],
    event_date: date,
    hours_before: int = 6,
    hours_after: int = 36,
) -> list[dict]:
    """
    Narrow a historical odds snapshot down to fights actually happening
    around a specific UFC event, discarding futures/outright lines for
    unrelated future cards.

    The window isn't a strict same-calendar-day match on purpose: a real
    card's commence_time often spans midnight UTC (UFC 329's July 11
    event ran 21:00Z July 11 -> 03:20Z July 12), and event_date's exact
    recorded convention (US Eastern vs. event-local vs. UTC) isn't
    guaranteed. hours_before/hours_after give slack in both directions
    rather than assuming exact alignment.

    Note: this filter only handles the "wrong date" category of noise
    (future/unrelated cards). It does NOT filter out same-day content
    from a different sport or promotion (e.g. an RAF wrestling match
    sharing the same sport_key) -- that's the fighter-name matcher's job,
    applied as a separate step after this one.
    """
    window_start = datetime.combine(
        event_date, datetime.min.time(), tzinfo=UTC
    ) - timedelta(hours=hours_before)
    window_end = datetime.combine(
        event_date, datetime.min.time(), tzinfo=UTC
    ) + timedelta(hours=hours_after)

    def _in_window(entry: dict) -> bool:
        commence = datetime.fromisoformat(entry["commence_time"])
        return window_start <= commence <= window_end

    return [entry for entry in odds_data if _in_window(entry)]
