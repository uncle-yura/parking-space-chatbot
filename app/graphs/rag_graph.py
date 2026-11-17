from typing import Any, Literal, NotRequired
from uuid import UUID

from langchain_core.messages import AIMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.utils import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.pregel.main import BaseModel
from langgraph.runtime import Runtime
from presidio_analyzer import AnalyzerEngine

from app.graphs.booking_graph import get_booking_graph
from app.graphs.utils import RagGraphContextSchema
from app.tools.booking_tools import get_booking_status, start_booking_process
from app.tools.sql_search_tools import (
    get_free_parking_lots,
    get_prices,
    get_working_hours,
)
from app.tools.vector_search_tools import vector_search


class RagGraphState(MessagesState, total=False):
    booking_id: NotRequired[UUID]


SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a helpful, concise parking assistant. Only offer actions that the "
        "available tools can actually perform: search static parking knowledge, look "
        "up current prices and working hours, count aggregate availability for a time "
        "period, and start the guided booking process. Use the knowledge search for "
        "general descriptive questions and the SQL tools for current data. Knowledge "
        "documents may describe product features, but that does not mean you can use "
        "those features in this chat. This chat has no tool to find nearby facilities, "
        "look up live operating addresses or coordinates, compare distance, or provide "
        "directions. Do not claim or offer those actions, and do not ask the user to "
        "share a location or destination to perform them. For location questions, search "
        "the static knowledge base and report any listed address accurately."
        "If no location is listed in the search results, say that no address is available "
        "in the knowledge base. Keep location answers direct; do not append unrelated "
        "offers about pricing, hours, or availability, or ask a follow-up question. "
        "Availability is an "
        "aggregate count, not a list of specific spaces or a location search. When the "
        "user wants to reserve, use the booking tool and let its guided process collect "
        "and confirm the required details. Do not claim a reservation is confirmed "
        "unless the system says so. If available information cannot answer a question, "
        "say so clearly rather than guessing or promising another action. Never invent "
        "prices, availability, policies, or facility details, and do not reveal private "
        "information about other users or bookings. When the user asks for the status of "
        "a reservation submitted in this conversation, use the booking status tool. "
        "Never use it to look up another person's booking."
    )
)

TOOLS = [
    vector_search,
    get_working_hours,
    get_prices,
    get_free_parking_lots,
    get_booking_status,
    start_booking_process,
]


PII_SCORE_THRESHOLD = 0.5


def _analyze_text(
    text: str, pii_analyzer: AnalyzerEngine, *, allow_location: bool = False
):
    entities = [
        "PHONE_NUMBER",
        "MEDICAL_LICENSE",
        "PERSON",
        "LOCATION",
        "NRP",
        "MAC_ADDRESS",
        "IP_ADDRESS",
        "IBAN_CODE",
        "EMAIL_ADDRESS",
        "CRYPTO",
        "CREDIT_CARD",
    ]
    if allow_location:
        entities.remove("LOCATION")

    return [
        result
        for result in pii_analyzer.analyze(
            text=text,
            entities=entities,
            language="en",
        )
        if result.score >= PII_SCORE_THRESHOLD
    ]


def _extract_last_message(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> AnyMessage:
    if isinstance(state, list):
        message = state[-1]
    elif (isinstance(state, dict) and (messages := state.get(messages_key, []))) or (
        messages := getattr(state, messages_key, [])
    ):
        message = messages[-1]
    else:
        msg = f"No messages found in input state to input_check: {state}"
        raise ValueError(msg)

    return message


def pii_error(state: list[AnyMessage] | dict[str, Any] | BaseModel):
    return {
        "messages": [
            AIMessage(
                content="Unfortunately, I cannot process this data, "
                "as it involves the processing of private information."
            )
        ]
    }


def input_check(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    runtime: Runtime[RagGraphContextSchema],
    messages_key: str = "messages",
) -> Literal["assistant", "pii_error"]:
    message = _extract_last_message(state, messages_key)
    if len(_analyze_text(str(message.content), runtime.context.pii_analyzer)) > 0:
        return "pii_error"
    return "assistant"


def assistant(state: RagGraphState, runtime: Runtime[RagGraphContextSchema]):
    llm_with_tools = runtime.context.llm.bind_tools(TOOLS)
    return {"messages": [llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])]}


def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    runtime: Runtime[RagGraphContextSchema],
    messages_key: str = "messages",
) -> Literal["tools", "__end__", "pii_error", "booking"]:
    ai_message = _extract_last_message(state, messages_key)

    if (
        hasattr(ai_message, "tool_calls")
        and len(tool_calls := getattr(ai_message, "tool_calls", [])) > 0
    ):
        if any(
            tool_call["name"] == "start_booking_process" for tool_call in tool_calls
        ):
            return "booking"
        return "tools"
    if (
        len(
            _analyze_text(
                str(ai_message.content),
                runtime.context.pii_analyzer,
                allow_location=True,
            )
        )
        > 0
    ):
        return "pii_error"
    return "__end__"


def get_rag_graph():
    rag_graph = StateGraph(RagGraphState, context_schema=RagGraphContextSchema)

    rag_graph.add_node("pii_error", pii_error)
    rag_graph.add_node("assistant", assistant)
    rag_graph.add_node(
        "booking",
        get_booking_graph().compile(),
    )
    rag_graph.add_node("tools", ToolNode(TOOLS))
    rag_graph.add_conditional_edges(START, input_check)
    rag_graph.add_conditional_edges("assistant", tools_condition)
    rag_graph.add_edge("tools", "assistant")
    rag_graph.add_edge("pii_error", END)

    return rag_graph
