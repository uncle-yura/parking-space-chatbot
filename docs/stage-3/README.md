# Stage 3 — LLM review and MCP persistence

Snapshot: `8f3e6e3` — “Process confirmed reservation by using MCP server”.
[Project README](../../README.md)

## Objective

Review pending requests with automated administrator agents and persist approved reservation details through a separate authenticated MCP service.

## Implemented changes and key files

- `app/graphs/administrator_graph.py`: claims requests using database leases and fans out per-booking reviews with LangGraph `Send`.
- `app/graphs/booking_review_graph.py`: an **automated LLM reviewer** chooses SQL approval/rejection tools, then requests MCP persistence after approval. SQL remains responsible for availability and atomic assignment.
- `app/tools/sql_tools.py`: state-scoped reservation decision tools.
- `app/admin_server.py`: renamed administrator HTTP service, now running the asynchronous review graph.
- `app/services/mcp_adapter.py`: authenticated Streamable HTTP client/adapter; this is the actual path, not `reservation_mcp_client.py`.
- `app/mcp_server.py` and `app/services/reservation_mcp.py`: expose `save_approved_reservation` and append name, registration, reservation period and UTC write timestamp to `data/approved_reservations.txt` under Compose.

## Run/check

From the repository root **at this commit**, copy `.env.example` to `.env`. Replace both token examples with independent secrets: `ADMIN_SERVER_TOKEN` and `MCP_SERVER_TOKEN`. Retain Compose service URLs, configure container-reachable Ollama (macOS: `http://host.docker.internal:11434`), and pull the configured models.

**SQL seeding resets the configured database; use disposable data.**

```sh
docker compose --file docker-compose-dev.yml up -d --build
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_sql
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_vectorstore
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.agents.customer_agent
```

Submit/confirm a synthetic request; check its CLI status and the approved-reservations file. Existing databases need `bookings.review_claimed_at` (see this commit's schema).

## Outcome/limitations

No human administrator decision UI exists; customer confirmation is distinct from automated review. SQL approval and file persistence are separate, non-atomic operations. This snapshot lacks Stage 4's persisted-write marker/recovery and file deduplication; an MCP failure can leave an approved booking unsaved.
