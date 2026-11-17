from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.graphs.utils import RagGraphContextSchema


class BookingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    full_name: str
    car_number: str
    reservation_start: datetime
    reservation_end: datetime
    booking_approved: bool
    cancelled: bool
    error: str | None


def _request_input(field: str, message: str, error: str | None):
    response = interrupt(
        {
            "type": "input",
            "field": field,
            "message": message,
            "error_message": error,
            "options": ["[cancel]"],
        }
    )
    return response, response == "[cancel]"


def start_booking(state: BookingState):
    return {"messages": [AIMessage(content="Please enter your full name.")]}


def get_full_name(state: BookingState):
    value, cancelled = _request_input(
        "full_name", "Please enter your full name", state.get("error")
    )
    if cancelled:
        return {"cancelled": cancelled}

    if len(value) > 250 or len(value) < 3:
        return {"error": "Please ensure that you have entered the correct name."}

    return {
        "full_name": value,
        "error": None,
        "messages": [AIMessage(content="Please enter your car number")],
    }


def get_car_number(state: BookingState):
    value, cancelled = _request_input(
        "car_number", "Please enter your car number", state.get("error")
    )
    if cancelled:
        return {"cancelled": cancelled}

    if len(value) < 6 or len(value) >= 10:
        return {"error": "Please ensure that you have entered the correct car number."}

    return {
        "car_number": value,
        "error": None,
        "messages": [
            AIMessage(
                content="Please enter the booking start time "
                "in the format YYYY-MM-DD HH:mm (e.g., 2026-01-31 13:00)"
            )
        ],
    }


def get_reservation_start(state: BookingState):
    value, cancelled = _request_input(
        "reservation_start",
        "Please enter the booking start time "
        "in the format YYYY-MM-DD HH:mm (e.g., 2026-01-31 13:00)",
        state.get("error"),
    )
    if cancelled:
        return {"cancelled": cancelled}

    try:
        rs_datetime = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return {
            "error": "Please use the format YYYY-MM-DD HH:mm (for example, 2026-01-31 13:00)"
        }

    return {
        "reservation_start": rs_datetime,
        "error": None,
        "messages": [
            AIMessage(
                content="Please enter the booking end time "
                "in the format YYYY-MM-DD HH:mm (e.g., 2026-01-31 13:00)"
            )
        ],
    }


def get_reservation_end(state: BookingState):
    value, cancelled = _request_input(
        "reservation_end",
        "Please enter the booking end time "
        "in the format YYYY-MM-DD HH:mm (e.g., 2026-01-31 13:00)",
        state.get("error"),
    )
    if cancelled:
        return {"cancelled": cancelled}

    try:
        re_datetime = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return {
            "error": "Please use the format YYYY-MM-DD HH:mm (for example, 2026-01-31 13:00)"
        }

    return {
        "reservation_end": re_datetime,
        "error": None,
    }


def date_range_error(state: BookingState):
    return {
        "messages": [
            AIMessage(
                content="Please ensure that the end date is later than the start date."
            )
        ]
    }


def validate_reservation_end(
    state: BookingState,
) -> Literal["confirm_booking", "date_range_error", "get_reservation_end", "__end__"]:
    if state.get("cancelled"):
        return "__end__"

    if state.get("error"):
        return "get_reservation_end"

    if state["reservation_end"] <= state["reservation_start"]:
        return "date_range_error"

    return "confirm_booking"


def confirm_booking(state: BookingState):
    decision = interrupt(
        {
            "type": "approval",
            "message": "Please confirm your booking details",
            "details": {
                "full_name": state["full_name"],
                "car_number": state["car_number"],
                "reservation_start": state["reservation_start"],
                "reservation_end": state["reservation_end"],
            },
            "options": ["[approve]", "[reject]"],
        }
    )

    return {"booking_approved": decision == "[approve]"}


def route_node_input(
    state: BookingState,
) -> Literal["continue", "retry", "cancel_booking"]:
    if state.get("cancelled"):
        return "cancel_booking"

    if state.get("error"):
        return "retry"

    return "continue"


def route_booking_decision(
    state: BookingState,
) -> Literal["finish_booking", "cancel_booking"]:
    if state["booking_approved"]:
        return "finish_booking"

    return "cancel_booking"


def cancel_booking(state: BookingState):
    return {"messages": [AIMessage(content="Your booking has been cancelled.")]}


def finish_booking(
    state: BookingState,
    runtime: Runtime[RagGraphContextSchema],
):
    runtime.context.sql_service.create_booking(
        customer_name=state["full_name"],
        car_number=state["car_number"],
        start_at=state["reservation_start"],
        end_at=state["reservation_end"],
    )
    return {
        "messages": [
            AIMessage(
                content="Thank you! Your application has been submitted. Please wait for the result."
            )
        ]
    }


def get_booking_graph():
    booking_graph = StateGraph(BookingState, context_schema=RagGraphContextSchema)

    booking_graph.add_node("start_booking", start_booking)
    booking_graph.add_node("get_full_name", get_full_name)
    booking_graph.add_node("get_car_number", get_car_number)
    booking_graph.add_node("get_reservation_start", get_reservation_start)
    booking_graph.add_node("get_reservation_end", get_reservation_end)
    booking_graph.add_node("date_range_error", date_range_error)
    booking_graph.add_node("confirm_booking", confirm_booking)
    booking_graph.add_node("cancel_booking", cancel_booking)
    booking_graph.add_node("finish_booking", finish_booking)

    booking_graph.add_edge(START, "start_booking")
    booking_graph.add_edge("start_booking", "get_full_name")
    for node_name in ["get_full_name", "get_car_number", "get_reservation_start"]:
        booking_graph.add_conditional_edges(
            node_name,
            route_node_input,
            {
                "cancel_booking": "cancel_booking",
                "continue": {
                    "get_full_name": "get_car_number",
                    "get_car_number": "get_reservation_start",
                    "get_reservation_start": "get_reservation_end",
                }[node_name],
                "retry": node_name,
            },
        )
    booking_graph.add_conditional_edges("confirm_booking", route_booking_decision)
    booking_graph.add_conditional_edges("get_reservation_end", validate_reservation_end)
    booking_graph.add_edge("date_range_error", "get_reservation_start")
    booking_graph.add_edge("finish_booking", END)
    booking_graph.add_edge("cancel_booking", END)

    return booking_graph
