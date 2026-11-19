from datetime import UTC, datetime
from typing import Any, cast
from urllib.error import URLError
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from presidio_analyzer import AnalyzerEngine

from app.admin_server import app as admin_app
from app.graphs.booking_graph import _notify_administrator, build_booking_graph
from app.graphs.booking_review_graph import status_condition, update_state
from app.graphs.utils import RagGraphContextSchema
from app.services import reservation_mcp
from app.services.embeddings import EmbeddingsService
from app.services.llm_chat import LLMService
from app.services.sql_service import SqlService


class FakeSqlService(SqlService):
    def __init__(self):
        self.created = None

    def create_booking(
        self,
        customer_name: str,
        car_number: str,
        start_at,
        end_at,
        lot_id: int | None = None,
        customer_id: int | None = None,
    ) -> dict:
        self.created = {
            "customer_name": customer_name,
            "car_number": car_number,
            "start_at": start_at,
            "end_at": end_at,
        }
        return {"id": uuid4(), "status": "pending"}


def _booking_setup(thread_id):
    sql_service = FakeSqlService()
    embeddings = EmbeddingsService().embeddings
    context = RagGraphContextSchema(
        llm=LLMService().llm,
        sql_service=sql_service,
        pii_analyzer=AnalyzerEngine.__new__(AnalyzerEngine),
        embeddings=embeddings,
        vectorstore=InMemoryVectorStore(embeddings),
        thread_id=thread_id,
    )
    graph = build_booking_graph().compile(checkpointer=InMemorySaver())
    config = cast(Any, {"configurable": {"thread_id": thread_id}})
    return sql_service, context, graph, config


def test_compiled_booking_graph_resumes_and_submits_booking(monkeypatch):
    sql_service, context, graph, config = _booking_setup("booking-test")
    monkeypatch.delenv("ADMIN_SERVER_URL", raising=False)
    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")
    notifications = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        notifications.append(
            (
                request.full_url,
                request.get_method(),
                request.get_header("Authorization"),
                timeout,
            )
        )
        return FakeResponse()

    monkeypatch.setattr("app.graphs.booking_graph.urlopen", fake_urlopen)
    state = graph.invoke(cast(Any, {"messages": []}), config=config, context=context)
    responses = {
        "full_name": "Taylor Example",
        "car_number": "ABC1234",
        "reservation_start": "2026-12-01 10:00",
        "reservation_end": "2026-12-01 12:00",
        "approval": "[approve]",
    }
    while interrupts := state.get("__interrupt__"):
        interrupt_data = interrupts[0].value
        resume = responses[interrupt_data.get("field", "approval")]
        state = graph.invoke(Command(resume=resume), config=config, context=context)
    assert sql_service.created is not None
    assert sql_service.created["customer_name"] == "Taylor Example"
    assert sql_service.created["car_number"] == "ABC1234"
    assert state["booking_id"]
    assert notifications == [
        ("http://localhost:8000/booking-processing", "POST", "Bearer test-token", 3)
    ]


def test_compiled_booking_graph_rejection_does_not_create_booking(monkeypatch):
    sql_service, context, graph, config = _booking_setup("booking-reject-test")
    state = graph.invoke(cast(Any, {"messages": []}), config=config, context=context)
    responses = {
        "full_name": "Taylor Example",
        "car_number": "ABC1234",
        "reservation_start": "2026-12-01 10:00",
        "reservation_end": "2026-12-01 12:00",
        "approval": "[reject]",
    }
    while interrupts := state.get("__interrupt__"):
        interrupt_data = interrupts[0].value
        state = graph.invoke(
            Command(resume=responses[interrupt_data.get("field", "approval")]),
            config=config,
            context=context,
        )
    assert sql_service.created is None
    assert "cancelled" in state["messages"][-1].content.lower()


@pytest.mark.parametrize(
    "cancel_field", ["car_number", "reservation_start", "reservation_end"]
)
def test_booking_graph_cancellation_at_each_remaining_prompt(cancel_field):
    sql_service, context, graph, config = _booking_setup(
        f"booking-cancel-{cancel_field}"
    )
    responses = {
        "full_name": "Taylor Example",
        "car_number": "ABC1234",
        "reservation_start": "2026-12-01 10:00",
        "reservation_end": "2026-12-01 12:00",
    }
    state = graph.invoke(cast(Any, {"messages": []}), config=config, context=context)
    while interrupts := state.get("__interrupt__"):
        field = interrupts[0].value["field"]
        response = "[cancel]" if field == cancel_field else responses[field]
        state = graph.invoke(Command(resume=response), config=config, context=context)
        if field == cancel_field:
            break
    assert state.get("cancelled") is True
    assert sql_service.created is None
    assert "cancelled" in state["messages"][-1].content.lower()


def test_booking_graph_retries_invalid_fields_and_date_range(monkeypatch):
    sql_service, context, graph, config = _booking_setup("booking-validation-test")
    state = graph.invoke(cast(Any, {"messages": []}), config=config, context=context)
    responses = {
        "full_name": ["ab", "Taylor Example"],
        "car_number": ["123", "ABC1234"],
        "reservation_start": ["bad date", "2026-12-01 10:00", "2026-12-01 10:00"],
        "reservation_end": ["bad date", "2026-12-01 09:00", "2026-12-01 12:00"],
        "approval": ["[reject]"],
    }
    while interrupts := state.get("__interrupt__"):
        interrupt_data = interrupts[0].value
        field = interrupt_data.get("field", "approval")
        state = graph.invoke(
            Command(resume=responses[field].pop(0)),
            config=config,
            context=context,
        )
    assert sql_service.created is None
    assert "Please ensure that the end date is later than the start date." in [
        message.content for message in state["messages"]
    ]
    assert len(responses["full_name"]) == 0
    assert len(responses["car_number"]) == 0


def test_booking_graph_cancel_at_first_prompt_does_not_create_booking():
    sql_service, context, graph, config = _booking_setup("booking-cancel-test")
    state = graph.invoke(cast(Any, {"messages": []}), config=config, context=context)
    state = graph.invoke(Command(resume="[cancel]"), config=config, context=context)
    assert sql_service.created is None
    assert "cancelled" in state["messages"][-1].content.lower()


def test_admin_endpoint_authenticates_and_runs_processor(monkeypatch):
    processor_calls = []
    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")
    monkeypatch.setattr(
        "app.admin_server._run_processor",
        lambda: processor_calls.append("processed"),
    )
    with TestClient(admin_app) as client:
        unauthorized = client.post("/booking-processing")
        accepted = client.post(
            "/booking-processing",
            headers={"Authorization": "Bearer test-token"},
        )
    assert unauthorized.status_code == 401
    assert accepted.status_code == 202
    assert processor_calls == ["processed"]


def test_admin_review_approval_routes_to_mcp_persistence(monkeypatch, tmp_path):
    target = tmp_path / "approved.txt"
    target.write_text("pre-existing record\n", encoding="utf-8")
    monkeypatch.setenv("RESERVATIONS_FILE", str(target))
    booking = {
        "id": str(uuid4()),
        "customer_name": "Taylor Example",
        "car_number": "ABC1234",
        "start_at": datetime(2026, 12, 1, 10),
        "end_at": datetime(2026, 12, 1, 12),
        "status": "pending",
    }
    decision = update_state(
        {
            "booking": booking,
            "messages": [
                ToolMessage(
                    content='{"status": "approved"}',
                    tool_call_id="decision-call",
                    name="approve_reservation",
                )
            ],
        }
    )
    approved_booking = decision["booking"]
    assert approved_booking["status"] == "approved"
    assert status_condition(cast(Any, {"booking": approved_booking})) == "review"
    for _ in range(2):
        saved = reservation_mcp.save_approved_reservation(
            name=approved_booking["customer_name"],
            car_number=approved_booking["car_number"],
            reservation_start=approved_booking["start_at"].isoformat(),
            reservation_end=approved_booking["end_at"].isoformat(),
        )
        assert saved == (
            "Approved reservation saved."
            if _ == 0
            else "Approved reservation already saved."
        )
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "pre-existing record"
    assert len(lines) == 2
    for line in lines[1:]:
        fields = [field.strip() for field in line.split("|")]
        assert len(fields) == 4
        assert fields[:3] == [
            "Taylor Example",
            "ABC1234",
            "2026-12-01T10:00:00 - 2026-12-01T12:00:00",
        ]
        approval_time = datetime.fromisoformat(fields[3])
        assert approval_time.tzinfo == UTC


def test_rejected_admin_decision_does_not_persist_to_mcp(monkeypatch, tmp_path):
    target = tmp_path / "approved.txt"
    monkeypatch.setenv("RESERVATIONS_FILE", str(target))
    booking = {"id": str(uuid4()), "status": "pending"}
    decision = update_state(
        {
            "booking": booking,
            "messages": [
                ToolMessage(
                    content='{"status": "rejected"}',
                    tool_call_id="decision-call",
                    name="approve_reservation",
                )
            ],
        }
    )
    assert decision["booking"]["status"] == "rejected"
    assert (
        status_condition(cast(Any, {"booking": decision["booking"]})) == "finish_review"
    )
    assert not target.exists()


def test_admin_notification_requires_token(monkeypatch):
    monkeypatch.delenv("ADMIN_SERVER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ADMIN_SERVER_TOKEN is not configured"):
        _notify_administrator()


@pytest.mark.parametrize("failure", [URLError("unavailable"), TimeoutError("timeout")])
def test_admin_notification_propagates_network_failure(monkeypatch, failure):
    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")
    monkeypatch.setattr(
        "app.graphs.booking_graph.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(type(failure)):
        _notify_administrator()


def test_mcp_rejects_record_delimiter_in_reservation_data(tmp_path, monkeypatch):
    monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "approved.txt"))
    with pytest.raises(ValueError, match="pipes or newlines"):
        reservation_mcp.save_approved_reservation(
            name="Taylor | Example",
            car_number="ABC1234",
            reservation_start="2026-12-01T10:00:00",
            reservation_end="2026-12-01T12:00:00",
        )
