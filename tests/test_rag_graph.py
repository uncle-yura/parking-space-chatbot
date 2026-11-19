import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from presidio_analyzer import AnalyzerEngine

from app.graphs import rag_graph
from app.graphs.utils import RagGraphContextSchema
from app.services.embeddings import EmbeddingsService
from app.services.llm_chat import LLMService
from app.services.sql_service import SqlService


class FakeSqlService(SqlService):
    def __init__(self):
        self.created = []

    def create_booking(
        self,
        customer_name: str,
        car_number: str,
        start_at,
        end_at,
        lot_id: int | None = None,
        customer_id: int | None = None,
    ) -> dict:
        booking = {
            "id": uuid4(),
            "status": "pending",
            "customer_name": customer_name,
            "car_number": car_number,
            "start_at": start_at,
            "end_at": end_at,
        }
        self.created.append(booking)
        return booking


def _invoke(graph, graph_input, **kwargs):
    return asyncio.run(graph.ainvoke(graph_input, **kwargs))


@pytest.fixture(autouse=True)
def stub_pii_detection(monkeypatch):
    monkeypatch.setattr(
        rag_graph,
        "_analyze_text",
        lambda text, analyzer, **kwargs: [object()] if "SECRET-PII-123" in text else [],
    )


def _context(thread_id):
    embeddings = EmbeddingsService().embeddings
    return RagGraphContextSchema(
        llm=LLMService().llm,
        sql_service=FakeSqlService(),
        pii_analyzer=AnalyzerEngine.__new__(AnalyzerEngine),
        embeddings=embeddings,
        vectorstore=InMemoryVectorStore(embeddings),
        thread_id=thread_id,
    )


async def _fake_assistant(state, runtime):
    return {"messages": [_assistant_response(state)]}


def _assistant_response(state):
    last_user_message = next(
        message
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )
    if "book" in str(last_user_message.content).lower():
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "start_booking_process",
                    "args": {},
                    "id": "booking-start-call",
                    "type": "tool_call",
                }
            ],
        )
    return AIMessage(content="Parking information response.")


def _mock_admin_notification(monkeypatch):
    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        "app.graphs.booking_graph.urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )


def test_assistant_refreshes_current_utc_date_each_call(monkeypatch):
    times = iter(
        [
            datetime(2030, 12, 31, 23, 59, tzinfo=UTC),
            datetime(2031, 1, 1, 0, 1, tzinfo=UTC),
        ]
    )

    class FakeDatetime:
        @staticmethod
        def now(tz):
            assert tz is UTC
            return next(times)

    class CapturingLLM:
        def __init__(self):
            self.calls = []

        def bind_tools(self, tools):
            assert tools == rag_graph.TOOLS
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content="Please specify your parking interval.")

    monkeypatch.setattr(rag_graph, "datetime", FakeDatetime)
    llm = CapturingLLM()
    runtime = cast(Any, SimpleNamespace(context=SimpleNamespace(llm=llm)))
    state = cast(
        rag_graph.RagGraphState,
        {"messages": [HumanMessage(content="Any spaces tomorrow?")]},
    )

    for _ in range(2):
        result = asyncio.run(rag_graph.assistant(state, runtime))
        assert result["messages"][0].content == "Please specify your parking interval."

    for messages, expected in zip(
        llm.calls,
        ["2030-12-31T23:59:00+00:00", "2031-01-01T00:01:00+00:00"],
        strict=True,
    ):
        content = str(messages[0].content)
        assert content == rag_graph.SYSTEM_MESSAGE.format(
            current_date=expected[:10], current_datetime=expected
        )
        assert f"Current date (UTC): {expected[:10]}" in content
        assert f"Current datetime (UTC): {expected}" in content
        assert messages[1:] == state["messages"]
    assert "Current date (UTC): {current_date}" in rag_graph.SYSTEM_MESSAGE
    assert "Current datetime (UTC): {current_datetime}" in rag_graph.SYSTEM_MESSAGE


def test_parent_graph_removes_rejected_pii_before_later_turn(monkeypatch):
    monkeypatch.setattr(rag_graph, "assistant", _fake_assistant)
    context = _context("privacy-regression")
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    config = cast(Any, {"configurable": {"thread_id": "privacy-regression"}})

    rejected = _invoke(
        graph,
        {"messages": [HumanMessage(content="My private value is SECRET-PII-123")]},
        config=config,
        context=context,
    )

    assert "private information" in rejected["messages"][-1].content.lower()
    assert all(
        "SECRET-PII-123" not in str(message.content) for message in rejected["messages"]
    )

    later = _invoke(
        graph,
        {"messages": [HumanMessage(content="What are your opening hours?")]},
        config=config,
        context=context,
    )

    assert later["messages"][-1].content == "Parking information response."
    assert all(
        "SECRET-PII-123" not in str(message.content) for message in later["messages"]
    )


@pytest.mark.parametrize(
    "cancel_booking", [False, True], ids=["completed", "cancelled"]
)
def test_parent_graph_continues_after_booking_completion_or_cancellation(
    monkeypatch, cancel_booking
):
    _mock_admin_notification(monkeypatch)
    thread_id = f"booking-followup-{cancel_booking}"
    context = _context(thread_id)
    monkeypatch.setattr(rag_graph, "assistant", _fake_assistant)
    graph = rag_graph.build_rag_graph().compile(checkpointer=InMemorySaver())
    config = cast(Any, {"configurable": {"thread_id": thread_id}})
    state = _invoke(
        graph,
        {"messages": [HumanMessage(content="I want to book a parking spot.")]},
        config=config,
        context=context,
    )
    responses = {
        "full_name": "[cancel]" if cancel_booking else "Taylor Example",
        "car_number": "ABC1234",
        "reservation_start": "2026-12-01 10:00",
        "reservation_end": "2026-12-01 12:00",
        "approval": "[approve]",
    }

    while interrupts := state.get("__interrupt__"):
        prompt = interrupts[0].value
        state = _invoke(
            graph,
            Command(resume=responses[prompt.get("field", "approval")]),
            config=config,
            context=context,
        )

    start_call = next(
        message
        for message in state["messages"]
        if isinstance(message, AIMessage)
        and any(call["name"] == "start_booking_process" for call in message.tool_calls)
    )
    matching_tool_reply = next(
        message
        for message in state["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id == start_call.tool_calls[0]["id"]
    )
    assert matching_tool_reply.content == "Booking process started."

    later = _invoke(
        graph,
        {"messages": [HumanMessage(content="What are your opening hours?")]},
        config=config,
        context=context,
    )

    assert later["messages"][-1].content == "Parking information response."
    if cancel_booking:
        assert cast(FakeSqlService, context.sql_service).created == []
    else:
        assert len(cast(FakeSqlService, context.sql_service).created) == 1
