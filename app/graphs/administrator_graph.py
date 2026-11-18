import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from app.graphs.booking_review_graph import build_booking_review_graph
from app.graphs.utils import AdministratorGraphContextSchema


class AdministratorState(MessagesState, total=False):
    pending: list[dict]
    processed: Annotated[list[dict], operator.add]


def gather_pending_bookings(
    state: AdministratorState, runtime: Runtime[AdministratorGraphContextSchema]
):
    return {"pending": runtime.context.sql_service.claim_pending_bookings()}


def dispatch_booking_agents(
    state: AdministratorState,
) -> list[Send] | Literal["__end__"]:
    pending = state.get("pending", [])
    if not pending:
        return "__end__"
    return [Send("review_booking", {"booking": booking}) for booking in pending]


async def build_administrator_graph():
    graph = StateGraph(
        AdministratorState, context_schema=AdministratorGraphContextSchema
    )

    graph.add_node("gather_pending_bookings", gather_pending_bookings)
    graph.add_node("review_booking", (await build_booking_review_graph()).compile())
    graph.add_edge(START, "gather_pending_bookings")
    graph.add_conditional_edges("gather_pending_bookings", dispatch_booking_agents)
    graph.add_edge("review_booking", END)

    return graph
