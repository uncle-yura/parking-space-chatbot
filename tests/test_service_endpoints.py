import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import admin_server, mcp_server


def test_admin_health_and_unconfigured_auth(monkeypatch):
    monkeypatch.delenv("ADMIN_SERVER_TOKEN", raising=False)
    with TestClient(admin_server.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/booking-processing")
    assert response.status_code == 503


def test_admin_booking_status_returns_not_found(monkeypatch):
    class EmptySqlService:
        def get_booking_status(self, booking_id):
            return None

    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")
    monkeypatch.setattr(admin_server, "SqlService", EmptySqlService)
    with TestClient(admin_server.app) as client:
        wrong_token = client.get(
            "/booking-status/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer wrong-token"},
        )
        response = client.get(
            "/booking-status/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer test-token"},
        )
    assert wrong_token.status_code == 401
    assert response.status_code == 404


def test_admin_processor_handles_success_and_failure(monkeypatch, caplog):
    class FakeGraph:
        def compile(self):
            return self

        async def ainvoke(self, *args, **kwargs):
            return {"processed": []}

    async def build_graph():
        return FakeGraph()

    monkeypatch.setattr(admin_server, "build_administrator_graph", build_graph)
    monkeypatch.setattr(admin_server, "SqlService", lambda: object())
    monkeypatch.setattr(
        admin_server,
        "LLMService",
        lambda: SimpleNamespace(llm=object()),
    )
    monkeypatch.setattr(
        admin_server,
        "AdministratorGraphContextSchema",
        lambda **kwargs: kwargs,
    )
    asyncio.run(admin_server._run_processor())

    async def fail_to_build():
        raise RuntimeError("processor failed")

    monkeypatch.setattr(admin_server, "build_administrator_graph", fail_to_build)
    asyncio.run(admin_server._run_processor())
    assert "Background reservation processing failed" in caplog.text


def test_admin_booking_status_returns_booking(monkeypatch):
    expected = {"status": "approved", "lot_identifier": "Lot A"}

    class FoundSqlService:
        def get_booking_status(self, booking_id):
            return expected

    monkeypatch.setenv("ADMIN_SERVER_TOKEN", "test-token")
    monkeypatch.setattr(admin_server, "SqlService", FoundSqlService)
    with TestClient(admin_server.app) as client:
        response = client.get(
            "/booking-status/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    assert response.json() == expected


def test_mcp_health_and_authentication(monkeypatch):
    monkeypatch.setenv("MCP_SERVER_TOKEN", "mcp-test-token")
    with TestClient(mcp_server.app) as client:
        health = client.get("/health")
        unauthorized = client.post("/mcp")
        wrong_token = client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert unauthorized.status_code == 401
    assert wrong_token.status_code == 401


def test_authenticated_mcp_call_persists_reservation(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_SERVER_TOKEN", "mcp-test-token")
    target = tmp_path / "reservations.txt"
    monkeypatch.setenv("RESERVATIONS_FILE", str(target))
    headers = {
        "Authorization": "Bearer mcp-test-token",
        "Accept": "application/json, text/event-stream",
    }
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "integration-test", "version": "1.0"},
        },
    }
    call_tool = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "save_approved_reservation",
            "arguments": {
                "name": "Taylor Example",
                "car_number": "ABC1234",
                "reservation_start": "2026-12-01T10:00:00",
                "reservation_end": "2026-12-01T12:00:00",
            },
        },
    }
    with TestClient(mcp_server.app) as client:
        initialized = client.post("/mcp/", headers=headers, json=initialize)
        assert initialized.status_code == 200, initialized.text
        session_id = initialized.headers.get("mcp-session-id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        response = client.post("/mcp/", headers=headers, json=call_tool)
    assert response.status_code == 200, response.text
    result_body = (
        response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else json.loads(
            next(
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            )
        )
    )
    assert result_body["result"]["isError"] is False
    fields = [
        field.strip() for field in target.read_text(encoding="utf-8").strip().split("|")
    ]
    assert fields[:3] == [
        "Taylor Example",
        "ABC1234",
        "2026-12-01T10:00:00 - 2026-12-01T12:00:00",
    ]
    assert len(fields) == 4
