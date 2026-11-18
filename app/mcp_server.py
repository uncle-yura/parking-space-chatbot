import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.services.reservation_mcp import mcp

mcp_app = mcp.http_app(path="/", transport="streamable-http")


@asynccontextmanager
async def lifespan(app):
    async with mcp_app.router.lifespan_context(mcp_app):
        yield


app = FastAPI(title="Parking Reservation MCP Server", lifespan=lifespan)
app.mount("/mcp", mcp_app)


@app.middleware("http")
async def require_bearer_token(request, call_next):
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    expected = os.environ.get("MCP_SERVER_TOKEN")
    if not request.url.path.startswith("/health") and (
        not expected
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(token, expected)
    ):
        return JSONResponse(
            {"detail": "Invalid or missing bearer token."},
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}
