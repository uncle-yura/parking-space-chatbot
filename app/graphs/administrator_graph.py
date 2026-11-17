from typing import TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.graphs.utils import AdministratorGraphContextSchema


class AdministratorState(TypedDict, total=False):
    processed: list[dict]
    messages: list[AIMessage]


def process_pending_bookings(
    state: AdministratorState, runtime: Runtime[AdministratorGraphContextSchema]
):
    sql_service = runtime.context.sql_service
    pending = sql_service.get_pending_bookings()
    processed = []

    for booking in pending:
        result = sql_service.process_pending_booking(booking["id"])
        if result is not None:
            processed.append({**booking, **result})

    if not processed:
        message = "There are no pending reservation requests."
    else:
        outcomes = []
        for booking in processed:
            if booking["status"] == "approved":
                outcomes.append(
                    f"Reservation {booking['id']} approved; assigned {booking['lot_identifier']}."
                )
            else:
                outcomes.append(
                    f"Reservation {booking['id']} rejected; no parking space is available."
                )
        message = "\n".join(outcomes)

    return {"processed": processed, "messages": [AIMessage(content=message)]}


def get_administrator_graph():
    graph = StateGraph(AdministratorState, context_schema=AdministratorGraphContextSchema)
    graph.add_node("process_pending_bookings", process_pending_bookings)
    graph.add_edge(START, "process_pending_bookings")
    graph.add_edge("process_pending_bookings", END)
    return graph
