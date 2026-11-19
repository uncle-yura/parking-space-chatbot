import asyncio
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.graphs import administrator_graph, booking_review_graph
from app.graphs.administrator_graph import (
    dispatch_booking_agents,
    gather_pending_bookings,
)
from app.graphs.booking_review_graph import (
    finish_review,
    review_booking,
    tools_condition,
    update_state,
)
from app.graphs.utils import AdministratorGraphContextSchema
from app.services import reservation_mcp


class FakeLLM:
    def __init__(self, response):
        self.responses = response if isinstance(response, list) else [response]
        self.bound_tools = None
        self.bind_options = {}
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.responses.pop(0)


def _decision_call(name, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _review_context(llm):
    return AdministratorGraphContextSchema.model_construct(
        llm=llm,
        sql_service=SimpleNamespace(
            mark_reservation_persisted=lambda booking_id: True,
            release_booking_claim=lambda booking_id: None,
        ),
    )


def _mcp_save_call():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "save_approved_reservation",
                "args": {
                    "name": "Taylor Example",
                    "car_number": "ABC1234",
                    "reservation_start": "2026-12-01T10:00:00",
                    "reservation_end": "2026-12-01T12:00:00",
                },
                "id": "mcp-save-call",
                "type": "tool_call",
            }
        ],
    )


def _fake_save_tool(saved_records):
    @tool
    def save_approved_reservation(
        name: str,
        car_number: str,
        reservation_start: str,
        reservation_end: str,
    ) -> str:
        """Fake MCP persistence tool for administrator graph tests."""
        saved_records.append((name, car_number, reservation_start, reservation_end))
        return "Approved reservation saved."

    return save_approved_reservation


def _booking(status, booking_id=None):
    return {
        "id": str(booking_id if booking_id is not None else uuid4()),
        "customer_name": "Taylor Example",
        "car_number": "ABC1234",
        "start_at": datetime(2026, 12, 1, 10),
        "end_at": datetime(2026, 12, 1, 12),
        "status": status,
    }


def test_compiled_review_graph_calls_fake_mcp_tool(monkeypatch):
    saved_records = []

    @tool
    def save_approved_reservation(
        name: str,
        car_number: str,
        reservation_start: str,
        reservation_end: str,
    ) -> str:
        """Fake MCP persistence tool used by the workflow test."""
        saved_records.append((name, car_number, reservation_start, reservation_end))
        return "Approved reservation saved."

    class FakeAdapter:
        async def list_tools(self):
            return [save_approved_reservation]

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    llm = FakeLLM(_mcp_save_call())
    context = _review_context(llm)
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile()
    booking = _booking("approved")
    result = asyncio.run(
        graph.ainvoke({"booking": booking, "messages": []}, context=context)
    )
    assert saved_records == [
        (
            "Taylor Example",
            "ABC1234",
            "2026-12-01T10:00:00",
            "2026-12-01T12:00:00",
        )
    ]
    assert result["booking"]["saved"] is True
    assert result["processed"][0]["status"] == "approved"


def test_pending_sql_approval_recovers_after_mcp_write_then_failure(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "approved.txt"))
    booking_id = uuid4()
    booking = _booking("pending", booking_id)
    record_attempts = []
    tool_attempts = 0

    @tool
    def save_approved_reservation(
        name: str,
        car_number: str,
        reservation_start: str,
        reservation_end: str,
    ) -> str:
        """Persist the reservation, simulating a lost response on first attempt."""
        nonlocal tool_attempts
        tool_attempts += 1
        result = reservation_mcp.save_approved_reservation(
            name=name,
            car_number=car_number,
            reservation_start=reservation_start,
            reservation_end=reservation_end,
        )
        record_attempts.append(result)
        if tool_attempts == 1:
            raise RuntimeError("MCP response lost after file write")
        return result

    class FakeAdapter:
        async def list_tools(self):
            return [save_approved_reservation]

    class FakeSqlService:
        def __init__(self):
            self.booking = dict(booking)
            self.claimed = False
            self.marked_saved = False

        def claim_pending_bookings(self):
            eligible = self.booking["status"] == "pending" or (
                self.booking["status"] == "approved" and not self.marked_saved
            )
            if not eligible or self.claimed:
                return []
            self.claimed = True
            return [dict(self.booking)]

        def process_pending_booking(self, requested_id):
            assert requested_id == booking_id
            assert self.claimed
            self.booking["status"] = "approved"
            self.claimed = False
            return {"status": "approved", "lot_id": 1, "lot_identifier": "TEST-LOT"}

        def mark_reservation_persisted(self, requested_id):
            assert requested_id == booking_id
            if self.marked_saved:
                return False
            self.marked_saved = True
            return True

        def release_booking_claim(self, requested_id):
            assert requested_id == booking_id
            self.claimed = False

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    sql_service = FakeSqlService()
    llm = FakeLLM(
        [
            _decision_call("approve_reservation", "approve-pending-booking"),
            _mcp_save_call(),
            _mcp_save_call(),
        ]
    )
    context = AdministratorGraphContextSchema.model_construct(
        llm=llm, sql_service=sql_service
    )

    async def run_administrator_processing():
        graph = await administrator_graph.build_administrator_graph()
        return await graph.compile().ainvoke(
            {"messages": [], "pending": [], "processed": []},
            context=context,
        )

    first_run = asyncio.run(run_administrator_processing())
    record_path = Path(os.environ["RESERVATIONS_FILE"])
    assert sql_service.booking["status"] == "approved"
    assert not sql_service.marked_saved
    assert "save_error" in first_run["processed"][0]
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 1

    second_run = asyncio.run(run_administrator_processing())
    assert sql_service.marked_saved
    assert tool_attempts == 2
    assert record_attempts == [
        "Approved reservation saved.",
        "Approved reservation already saved.",
    ]
    assert second_run["processed"][0]["saved"] is True
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 1
    assert llm.calls == 3


def test_pending_approval_runs_sql_tool_then_calls_mcp(monkeypatch):
    saved_records = []
    save_tool = _fake_save_tool(saved_records)

    class FakeAdapter:
        async def list_tools(self):
            return [save_tool]

    class FakeSqlService:
        def __init__(self):
            self.approved = []

        def process_pending_booking(self, booking_id):
            self.approved.append(booking_id)
            return {"status": "approved", "lot_id": 1, "lot_identifier": "A"}

        def mark_reservation_persisted(self, booking_id):
            return True

        def release_booking_claim(self, booking_id):
            return None

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    sql_service = FakeSqlService()
    llm = FakeLLM(
        [_decision_call("approve_reservation", "approve-call"), _mcp_save_call()]
    )
    context = AdministratorGraphContextSchema.model_construct(
        llm=llm, sql_service=sql_service
    )
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile()
    booking_id = uuid4()
    booking = _booking("pending", booking_id)
    result = asyncio.run(
        graph.ainvoke({"booking": booking, "messages": []}, context=context)
    )
    assert sql_service.approved == [booking_id]
    assert saved_records == [
        (
            "Taylor Example",
            "ABC1234",
            "2026-12-01T10:00:00",
            "2026-12-01T12:00:00",
        )
    ]
    assert result["booking"]["status"] == "approved"
    assert result["booking"]["saved"] is True


def test_pending_rejection_uses_reject_tool_and_never_calls_mcp(monkeypatch):
    saved_records = []
    save_tool = _fake_save_tool(saved_records)

    class FakeAdapter:
        async def list_tools(self):
            return [save_tool]

    class FakeSqlService:
        def __init__(self):
            self.rejected = []

        def reject_pending_booking(self, booking_id):
            self.rejected.append(booking_id)
            return {"status": "rejected", "lot_id": None, "lot_identifier": None}

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    sql_service = FakeSqlService()
    llm = FakeLLM(_decision_call("reject_reservation", "reject-call"))
    context = AdministratorGraphContextSchema.model_construct(
        llm=llm, sql_service=sql_service
    )
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile()
    booking_id = uuid4()
    booking = _booking("pending", booking_id)
    result = asyncio.run(
        graph.ainvoke({"booking": booking, "messages": []}, context=context)
    )
    assert sql_service.rejected == [booking_id]
    assert saved_records == []
    assert result["booking"]["status"] == "rejected"
    assert llm.calls == 1


def test_mcp_tool_exception_is_recorded_without_marking_saved(monkeypatch):
    @tool
    def save_approved_reservation(
        name: str,
        car_number: str,
        reservation_start: str,
        reservation_end: str,
    ) -> str:
        """Fake MCP tool that fails to persist the reservation."""
        calls.append(name)
        raise RuntimeError("MCP storage unavailable")

    calls = []

    class FakeAdapter:
        async def list_tools(self):
            return [save_approved_reservation]

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    llm = FakeLLM(_mcp_save_call())
    released_claims = []
    context = AdministratorGraphContextSchema.model_construct(
        llm=llm,
        sql_service=SimpleNamespace(
            release_booking_claim=lambda booking_id: released_claims.append(booking_id),
            mark_reservation_persisted=lambda booking_id: pytest.fail(
                "failed persistence must not be marked saved"
            ),
        ),
    )
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile()
    booking = _booking("approved")
    result = asyncio.run(
        graph.ainvoke({"booking": booking, "messages": []}, context=context)
    )
    assert llm.calls == 1
    assert calls == ["Taylor Example"]
    assert result["booking"].get("saved") is not True
    assert "MCP storage unavailable" in result["booking"]["save_error"]
    assert released_claims == [UUID(booking["id"])]


def test_interrupt_before_mcp_tool_does_not_mark_saved_and_can_resume(monkeypatch):
    saved_records = []
    save_tool = _fake_save_tool(saved_records)

    class FakeAdapter:
        async def list_tools(self):
            return [save_tool]

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    llm = FakeLLM(_mcp_save_call())
    context = AdministratorGraphContextSchema.model_construct(
        llm=llm,
        sql_service=SimpleNamespace(
            mark_reservation_persisted=lambda booking_id: True,
            release_booking_claim=lambda booking_id: None,
        ),
    )
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["booking_tools"],
    )
    config = cast(Any, {"configurable": {"thread_id": "mcp-interrupt-test"}})
    booking = _booking("approved")
    paused = asyncio.run(
        graph.ainvoke(
            {"booking": booking, "messages": []},
            config=config,
            context=context,
        )
    )
    assert saved_records == []
    assert paused["booking"].get("saved") is not True
    resumed = asyncio.run(graph.ainvoke(None, config=config, context=context))
    assert len(saved_records) == 1
    assert resumed["booking"]["saved"] is True


def test_graph_builders_register_expected_nodes(monkeypatch):
    async def fake_review_graph():
        graph = StateGraph(MessagesState)
        graph.add_node("fake_review", lambda state: {})
        graph.add_edge(START, "fake_review")
        graph.add_edge("fake_review", END)
        return graph

    class FakeAdapter:
        async def list_tools(self):
            return []

    monkeypatch.setattr(
        administrator_graph, "build_booking_review_graph", fake_review_graph
    )
    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    admin_graph = asyncio.run(administrator_graph.build_administrator_graph())
    review_graph = asyncio.run(booking_review_graph.build_booking_review_graph())
    assert "review_booking" in admin_graph.nodes
    assert "review" in review_graph.nodes
    assert "booking_tools" in review_graph.nodes


def test_administrator_dispatch_handles_empty_and_pending_queues():
    empty_runtime = SimpleNamespace(
        context=SimpleNamespace(
            sql_service=SimpleNamespace(claim_pending_bookings=list),
        )
    )
    assert gather_pending_bookings(cast(Any, {}), cast(Any, empty_runtime)) == {
        "pending": []
    }
    assert dispatch_booking_agents(cast(Any, {"pending": []})) == "__end__"
    pending = [{"id": str(uuid4())}, {"id": str(uuid4())}]
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            sql_service=SimpleNamespace(claim_pending_bookings=lambda: pending),
        )
    )
    state = gather_pending_bookings(cast(Any, {}), cast(Any, runtime))
    sends = dispatch_booking_agents(cast(Any, state))
    assert state["pending"] == pending
    assert isinstance(sends, list)
    assert [send.node for send in sends] == ["review_booking", "review_booking"]
    assert [send.arg for send in sends] == [{"booking": item} for item in pending]


def test_review_agent_routes_tool_calls_and_forces_single_decision():
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "approve_reservation",
                "args": {},
                "id": "one",
                "type": "tool_call",
            },
            {
                "name": "reject_reservation",
                "args": {},
                "id": "two",
                "type": "tool_call",
            },
        ],
    )
    llm = FakeLLM(response)
    runtime = SimpleNamespace(context=SimpleNamespace(llm=llm))
    state = {"booking": {"id": "booking-id", "status": "pending"}, "messages": []}
    result = asyncio.run(review_booking(cast(Any, state), cast(Any, runtime), []))
    assert len(result["messages"][0].tool_calls) == 1
    assert llm.bound_tools
    assert llm.bind_options == {}
    assert (
        tools_condition(cast(Any, {"messages": result["messages"]})) == "booking_tools"
    )


def test_approved_booking_requires_mcp_tool_and_non_tool_response_finishes():
    llm = FakeLLM(AIMessage(content="done"))
    runtime = SimpleNamespace(context=SimpleNamespace(llm=llm))
    save_tool = object()
    state = {"booking": {"id": "booking-id", "status": "approved"}, "messages": []}
    result = asyncio.run(
        review_booking(
            cast(Any, state),
            cast(Any, runtime),
            cast(Any, [save_tool]),
        )
    )
    assert llm.bound_tools == [save_tool]
    assert llm.bind_options == {"tool_choice": "save_approved_reservation"}
    assert (
        tools_condition(cast(Any, {"messages": result["messages"]})) == "finish_review"
    )


def test_compiled_graph_marks_approved_booking_incomplete_when_model_skips_mcp(
    monkeypatch,
):
    save_calls = []

    @tool
    def save_approved_reservation(
        name: str,
        car_number: str,
        reservation_start: str,
        reservation_end: str,
    ) -> str:
        """Track whether the compiled workflow attempts MCP persistence."""
        save_calls.append(name)
        return "Approved reservation saved."

    class FakeAdapter:
        async def list_tools(self):
            return [save_approved_reservation]

    monkeypatch.setattr(booking_review_graph, "get_mcp_adapter", lambda: FakeAdapter())
    released_claims = []
    context = AdministratorGraphContextSchema.model_construct(
        llm=FakeLLM(AIMessage(content="done")),
        sql_service=SimpleNamespace(
            release_booking_claim=released_claims.append,
            mark_reservation_persisted=lambda booking_id: pytest.fail(
                "MCP was not called, so persistence cannot be marked"
            ),
        ),
    )
    graph = asyncio.run(booking_review_graph.build_booking_review_graph()).compile()
    booking_id = uuid4()
    booking = _booking("approved", booking_id)
    result = asyncio.run(
        graph.ainvoke({"booking": booking, "messages": []}, context=context)
    )
    assert save_calls == []
    assert result["processed"][0]["status"] == "approved"
    assert result["processed"][0].get("saved") is not True
    assert (
        result["processed"][0]["save_error"]
        == "Approved reservation was not persisted."
    )
    assert released_claims == [booking_id]


def test_review_state_handles_tool_errors_without_marking_saved():
    booking = {"id": str(uuid4()), "status": "approved"}
    state = update_state(
        cast(
            Any,
            {
                "booking": booking,
                "messages": [
                    ToolMessage(
                        content="MCP storage unavailable",
                        tool_call_id="failed-save",
                        name="save_approved_reservation",
                        status="error",
                    )
                ],
            },
        )
    )
    assert state["booking"].get("saved") is not True
    assert state["booking"]["save_error"] == "MCP storage unavailable"


def test_review_state_handles_unrecognized_message_and_saved_status():
    assert update_state({"messages": [HumanMessage(content="review")]}) == {}
    assert (
        booking_review_graph.status_condition(
            cast(
                Any,
                {
                    "booking": {"status": "approved", "saved": True},
                },
            )
        )
        == "finish_review"
    )


def test_update_state_marks_mcp_save_and_finish_releases_pending_claim():
    booking = {
        "id": str(uuid4()),
        "status": "pending",
        "start_at": datetime(2026, 12, 1, 10),
    }
    saved_state = update_state(
        {
            "booking": booking,
            "messages": [
                ToolMessage(
                    content="saved",
                    tool_call_id="save-call",
                    name="save_approved_reservation",
                )
            ],
        }
    )
    assert saved_state["booking"]["saved"] is True
    released = []
    runtime = SimpleNamespace(
        context=SimpleNamespace(
            sql_service=SimpleNamespace(release_booking_claim=released.append),
        )
    )
    result = finish_review(cast(Any, {"booking": booking}), cast(Any, runtime))
    assert result["processed"] == [booking]
    assert released == [UUID(booking["id"])]
    approved = {"id": booking["id"], "status": "approved"}
    finished = finish_review(cast(Any, {"booking": approved}), cast(Any, runtime))
    assert finished["processed"] == [
        {**approved, "save_error": "Approved reservation was not persisted."}
    ]
