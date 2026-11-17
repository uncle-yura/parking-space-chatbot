from langchain.tools import ToolRuntime, tool

from app.graphs.utils import RagGraphContextSchema


@tool
def get_free_parking_lots(
    runtime: ToolRuntime[RagGraphContextSchema],
    period_start: str | None = None,
    period_end: str | None = None,
) -> str:
    """Count spaces free now, or throughout an ISO-8601 start/end interval."""
    if (period_start is None) != (period_end is None):
        return "Provide both period_start and period_end, or neither."
    lots = runtime.context.sql_service.get_free_parking_lots(
        start_at=period_start, end_at=period_end
    )
    if not lots:
        return "No parking lots found."
    return f"Available parking lots for the requested period: {lots[0]['free_lots']}"


@tool
def get_prices(
    runtime: ToolRuntime[RagGraphContextSchema], as_of: str | None = None
) -> str:
    """Return the price effective now, or on the requested ISO-8601 date/time."""
    prices = runtime.context.sql_service.get_prices(as_of=as_of)
    if not prices:
        return "No pricing data found."

    blocks = []
    for p in prices:
        cap = f", daily cap: ${p['price_per_day']}" if p["price_per_day"] else ""
        limit = f", max stay: {p['max_stay_hours']}h" if p["max_stay_hours"] else ""
        blocks.append(f"  PRICING: ${p['price_per_hour']}/hr{cap}{limit}")
    return "\n".join(blocks)


@tool
def get_working_hours(runtime: ToolRuntime[RagGraphContextSchema]) -> str:
    """Return working hours (recurring + holiday overrides). Use when asked about open/close times."""

    rows = runtime.context.sql_service.get_working_hours()
    if not rows:
        return "No working hours data found."
    blocks = []
    for r in rows:
        day_label = (
            f"day {r['day_of_week']}"
            if r["day_of_week"] is not None
            else f"on {r['date']} ({r['reason']})"
        )
        blocks.append(f"  PARKING: {day_label} — {r['start_time']} to {r['end_time']}")
    return "\n".join(blocks)
