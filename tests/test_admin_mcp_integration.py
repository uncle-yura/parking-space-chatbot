import asyncio
import json
import socket
import threading
import time
from datetime import datetime
from uuid import uuid4

import uvicorn
from langchain_core.messages import AIMessage

from app.graphs.administrator_graph import build_administrator_graph
from app.graphs.utils import AdministratorGraphContextSchema
from app.mcp_server import app as mcp_app


class BookingAwareLLM:
    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        request = json.loads(str(messages[0].content).split("Request: ", 1)[1])
        if request["status"] == "pending":
            call = {"name": "approve_reservation", "args": {}}
        else:
            call = {
                "name": "save_approved_reservation",
                "args": {
                    "name": request["customer_name"],
                    "car_number": request["car_number"],
                    "reservation_start": request["start_at"],
                    "reservation_end": request["end_at"],
                },
            }
        return AIMessage(
            content="",
            tool_calls=[{**call, "id": str(uuid4()), "type": "tool_call"}],
        )


class FakeSqlService:
    def __init__(self, bookings):
        self.bookings = {str(item["id"]): dict(item) for item in bookings}
        self.persisted = set()

    def claim_pending_bookings(self):
        return [dict(booking) for booking in self.bookings.values()]

    def process_pending_booking(self, booking_id):
        booking = self.bookings[str(booking_id)]
        booking["status"] = "approved"
        return {"status": "approved", "lot_id": 1, "lot_identifier": "TEST"}

    def mark_reservation_persisted(self, booking_id):
        self.persisted.add(str(booking_id))
        return True

    def release_booking_claim(self, booking_id):
        return None


def test_admin_graph_persists_each_booking_through_production_mcp_adapter(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MCP_SERVER_TOKEN", "integration-token")
    record_file = tmp_path / "approved.txt"
    monkeypatch.setenv("RESERVATIONS_FILE", str(record_file))

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(mcp_app, host="127.0.0.1", port=port, log_level="error")
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "MCP server did not start"
    monkeypatch.setenv("RESERVATION_MCP_URL", f"http://127.0.0.1:{port}/mcp")

    bookings = [
        {
            "id": str(uuid4()),
            "customer_name": "Taylor Example",
            "car_number": "ABC1234",
            "start_at": datetime(2026, 12, 1, 10),
            "end_at": datetime(2026, 12, 1, 12),
            "status": "pending",
        },
        {
            "id": str(uuid4()),
            "customer_name": "Morgan Sample",
            "car_number": "XYZ9876",
            "start_at": datetime(2026, 12, 2, 14),
            "end_at": datetime(2026, 12, 2, 16),
            "status": "pending",
        },
    ]
    sql_service = FakeSqlService(bookings)
    context = AdministratorGraphContextSchema.model_construct(
        llm=BookingAwareLLM(), sql_service=sql_service
    )

    try:
        graph = asyncio.run(build_administrator_graph())
        result = asyncio.run(
            graph.compile().ainvoke(
                {"messages": [], "pending": [], "processed": []}, context=context
            )
        )
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)

    lines = record_file.read_text(encoding="utf-8").splitlines()
    records = {
        fields[0].strip(): [field.strip() for field in fields[:3]]
        for fields in (line.split("|") for line in lines)
    }
    assert len(lines) == 2
    assert records == {
        "Taylor Example": [
            "Taylor Example",
            "ABC1234",
            "2026-12-01 10:00:00 - 2026-12-01 12:00:00",
        ],
        "Morgan Sample": [
            "Morgan Sample",
            "XYZ9876",
            "2026-12-02 14:00:00 - 2026-12-02 16:00:00",
        ],
    }
    assert sql_service.persisted == {str(item["id"]) for item in bookings}
    assert {item["customer_name"] for item in result["processed"]} == {
        "Taylor Example",
        "Morgan Sample",
    }
    assert all(item["saved"] is True for item in result["processed"])
