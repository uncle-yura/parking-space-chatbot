import json
from datetime import datetime
from uuid import UUID

from langchain.tools import ToolRuntime, tool

from app.graphs.utils import AdministratorGraphContextSchema, RagGraphContextSchema


@tool
def get_free_parking_lots(
    runtime: ToolRuntime[RagGraphContextSchema],
    period_start: str | None = None,
    period_end: str | None = None,
) -> str:
    """Count spaces free now, or throughout an ISO-8601 start/end interval."""
    if (period_start is None) != (period_end is None):
        return "Provide both period_start and period_end, or neither."

    if period_start is not None and period_end is not None:
        try:
            start_at = datetime.fromisoformat(period_start)
            end_at = datetime.fromisoformat(period_end)
        except (TypeError, ValueError):
            return "Use valid ISO-8601 datetimes for period_start and period_end."
        if end_at <= start_at:
            return "period_end must be later than period_start."
    else:
        start_at = end_at = None

    lots = runtime.context.sql_service.get_free_parking_lots(
        start_at=start_at, end_at=end_at
    )
    if not lots:
        return "No parking lots found."
    return f"Available parking lots for the requested period: {lots[0]['free_lots']}"


@tool
def get_prices(
    runtime: ToolRuntime[RagGraphContextSchema], as_of: str | None = None
) -> str:
    """Return the price effective now, or on the requested ISO-8601 date/time."""
    parsed_as_of = None
    if as_of is not None:
        try:
            parsed_as_of = datetime.fromisoformat(as_of)
        except (TypeError, ValueError):
            return "Use a valid ISO-8601 datetime for as_of."

    prices = runtime.context.sql_service.get_prices(as_of=parsed_as_of)
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


@tool
def approve_reservation(runtime: ToolRuntime[AdministratorGraphContextSchema]) -> str:
    """Approve this valid reservation if an available parking space can be assigned."""
    booking = runtime.state.get("booking")
    if not booking:
        raise ValueError("Approval tool requires a booking in graph state.")

    name = booking.get("customer_name")
    car_number = booking.get("car_number")
    start_at = booking.get("start_at")
    end_at = booking.get("end_at")
    invalid = (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(car_number, str)
        or not car_number.strip()
        or not isinstance(start_at, datetime)
        or not isinstance(end_at, datetime)
        or end_at <= start_at
    )
    sql_service = runtime.context.sql_service
    booking_id = UUID(str(booking["id"]))
    if invalid:
        result = sql_service.reject_pending_booking(booking_id)
        return json.dumps(result or {"status": "unchanged"}, default=str)

    result = sql_service.process_pending_booking(booking_id)
    return json.dumps(result or {"status": "unchanged"}, default=str)


@tool
def reject_reservation(runtime: ToolRuntime[AdministratorGraphContextSchema]) -> str:
    """Reject this pending reservation when its request data is invalid."""
    booking = runtime.state.get("booking")
    if not booking:
        raise ValueError("Rejection tool requires a booking in graph state.")
    result = runtime.context.sql_service.reject_pending_booking(
        UUID(str(booking["id"]))
    )
    return json.dumps(result or {"status": "unchanged"}, default=str)
