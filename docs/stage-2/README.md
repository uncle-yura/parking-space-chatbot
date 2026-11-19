# Stage 2 — Automated administrator processing

Snapshot: `4e91d5b` — “Implementation of Human-in-the-Loop Agent”.
[Project README](../../README.md)

## Objective

Extend Stage 1's pending-request workflow with asynchronous processing and customer-visible status. The commit title is aspirational: this snapshot does not implement a human administrator decision UI.

## Implemented changes and key files

- `app/graphs/administrator_graph.py`: a deterministic processor, **not an LLM reviewer**, iterating pending reservations.
- `app/services/sql_service.py`: UUID bookings, status queries, and transaction/advisory-lock-protected availability checks and lot assignment; approve if a space is free, otherwise reject.
- `app/server.py`: FastAPI bearer-authenticated `POST /booking-processing` (202/background processing) and `GET /booking-status/{booking_id}`; public `/health`.
- `app/graphs/booking_graph.py`: save first, then notify the administrator service with retries.
- `app/agents/customer_agent.py`: asynchronous CLI, displayed confirmation details/options, and five-second background status polling.
- `app/tools/booking_tools.py`, `app/tools/sql_search_tools.py`, `app/tools/vector_search_tools.py`: actual pluralized paths at this stage.

## Run/check

Run from the repository root **at this commit**. Copy `.env.example` to `.env`; replace `ADMIN_SERVER_TOKEN` with a unique secret. Configure Docker service hostnames and a container-reachable Ollama URL (macOS: `http://host.docker.internal:11434`); pull the configured models.

**SQL seeding destroys the configured database; use disposable data.**

```sh
docker compose --file docker-compose-dev.yml up -d --build
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_sql
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_vectorstore
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.agents.customer_agent
```

Submit a request, confirm with `[approve]`, and observe the eventual status/assigned lot in the CLI.

## Outcome/limitations

The customer confirms details; SQL decides approval automatically. No human administrator approval/rejection interface or LLM administrator exists here. Presidio still uses recognizers plus pretrained spaCy NLP, not rules alone. Background-task acceptance is not completed approval; notification failure can leave a saved request pending. MCP persistence arrives in Stage 3.
