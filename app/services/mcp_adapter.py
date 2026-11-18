import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter


def get_mcp_adapter() -> MCPAdapter:
    token = os.environ.get("MCP_SERVER_TOKEN")
    if not token:
        raise RuntimeError("MCP_SERVER_TOKEN is not configured")

    url = os.environ.get("RESERVATION_MCP_URL", "http://localhost:8001/mcp")
    client = Client(
        StreamableHttpTransport(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
    )
    return MCPAdapter(client)
