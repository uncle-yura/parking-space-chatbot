import json
import operator
from typing import Annotated, Literal
from uuid import UUID

from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from app.graphs.utils import AdministratorGraphContextSchema, extract_last_message
from app.services.mcp_adapter import get_mcp_adapter
from app.tools.sql_tools import approve_reservation, reject_reservation


class BookingReviewState(MessagesState, total=False):
    booking: dict
    processed: Annotated[list[dict], operator.add]


TOOLS = [approve_reservation, reject_reservation]


async def review_booking(
    state: BookingReviewState,
    runtime: Runtime[AdministratorGraphContextSchema],
    mcp_tools: list[BaseTool],
):
    booking = state.get("booking", {})
    messages = state.get("messages", [])

    system_message = SystemMessage(
        content=(
            "If the SQL approval tool has already returned status approved, call "
            "save_approved_reservation now and do nothing else. Otherwise review "
            "the request and call exactly one SQL decision tool. Reject only if "
            "required data is missing or invalid: customer name and car number "
            "must be non-empty, timestamps valid, and end time later than start. "
            "Otherwise call approve_reservation, which checks availability. Do not "
            "invent data or decide without a tool.\n"
            f"Request: {json.dumps(booking, default=str)}"
        )
    )
    if booking.get("status", "pending") == "approved":
        bound_llm = runtime.context.llm.bind_tools(
            mcp_tools,
            tool_choice="save_approved_reservation",
        )
    else:
        bound_llm = runtime.context.llm.bind_tools(
            [approve_reservation, reject_reservation]
        )
    response = await bound_llm.ainvoke([system_message, *messages])
    if hasattr(response, "tool_calls") and len(getattr(response, "tool_calls", [])) > 0:
        response = response.model_copy(update={"tool_calls": response.tool_calls[:1]})
    return {"messages": [response]}


def tools_condition(
    state: BookingReviewState,
    messages_key: str = "messages",
) -> Literal["booking_tools", "finish_review"]:
    message = extract_last_message(state, messages_key)
    if hasattr(message, "tool_calls") and len(getattr(message, "tool_calls", [])) > 0:
        return "booking_tools"
    return "finish_review"


def update_state(
    state: BookingReviewState,
    messages_key: str = "messages",
):
    message = extract_last_message(state, messages_key)
    if isinstance(message, ToolMessage) and message.name == "approve_reservation":
        result = json.loads(str(message.content))
        return {"booking": {**state.get("booking", {}), "status": result.get("status")}}

    if isinstance(message, ToolMessage) and message.name == "save_approved_reservation":
        return {"booking": {**state.get("booking", {}), "saved": True}}

    return {}


def status_condition(
    state: BookingReviewState,
) -> Literal["review", "finish_review"]:
    if state.get("booking", {}).get("saved", False):
        return "finish_review"

    if state.get("booking", {}).get("status", "pending") == "approved":
        return "review"

    return "finish_review"


def finish_review(
    state: BookingReviewState,
    runtime: Runtime[AdministratorGraphContextSchema],
):
    booking = state.get("booking", {})
    if booking.get("status", "pending") == "pending":
        runtime.context.sql_service.release_booking_claim(UUID(str(booking["id"])))
    return {"processed": [booking]}


async def build_booking_review_graph():
    graph = StateGraph(
        BookingReviewState, context_schema=AdministratorGraphContextSchema
    )

    mcp_tools = await get_mcp_adapter().list_tools()

    async def review_with_mcp(
        state: BookingReviewState, runtime: Runtime[AdministratorGraphContextSchema]
    ):
        return await review_booking(state, runtime, mcp_tools)

    graph.add_node("review", review_with_mcp)
    graph.add_node("booking_tools", ToolNode(TOOLS + mcp_tools))
    graph.add_node("update_state", update_state)
    graph.add_node("finish_review", finish_review)
    graph.add_edge(START, "review")
    graph.add_conditional_edges("review", tools_condition)
    graph.add_conditional_edges("update_state", status_condition)
    graph.add_edge("booking_tools", "update_state")
    graph.add_edge("finish_review", END)

    return graph
