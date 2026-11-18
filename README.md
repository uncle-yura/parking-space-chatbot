# Parking Space Chatbot

An AI-powered parking space assistant built with LangGraph, Ollama, and PostgreSQL. It provides RAG-based Q&A over parking knowledge and supports automated booking workflows.

## Quick Start

```bash
# Start all services (LangGraph API, administrator + MCP servers, PostgreSQL, Weaviate)
docker compose --file docker-compose-dev.yml up -d

# Seed the relational database with schema & sample data
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_sql

# Build vector embeddings and populate the vector store
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_vectorstore

# Run the customer agent CLI directly (outside Docker)
python -m app.agents.customer_agent
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [docker compose](https://docs.docker.com/compose/) v2+
- [Ollama](https://ollama.ai/) running locally (or adjust `OLLAMA_BASE_URL` in `.env`)

## Environment Variables

Copy `.env.example` to `.env`, then generate separate unique secrets for the administrator API and MCP server:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Generate a token with `python -c 'import secrets; print(secrets.token_urlsafe(32))'` and set it as `ADMIN_SERVER_TOKEN`; generate another independently and set it as `MCP_SERVER_TOKEN`. Do not reuse these tokens.

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRES_DB` | `parking_chatbot` | Target database name |
| `ADMIN_SERVER_URL` | `http://administrator-server:8000` | Internal administrator service URL |
| `ADMIN_SERVER_TOKEN` | (required) | Bearer token for RAG-to-administrator requests |
| `MCP_SERVER_TOKEN` | (required) | Separate bearer token for administrator-to-MCP requests |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_LLM_MODEL` | `qwen3.6:35b` | Model used for chat/inference |
| `OLLAMA_EMBEDDINGS_MODEL` | `qwen3-embedding:8b` | Model used for embedding generation |
| `WEAVIATE_HOST` | `localhost` | Weaviate hostname |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |

## Project Structure

```
.
├── app/
│   ├── admin_server.py              # Authenticated administrator HTTP service
│   ├── mcp_server.py                # Standalone authenticated reservation MCP service
│   ├── agents/
│   │   └── customer_agent.py       # CLI entry-point for the customer agent
│   ├── graphs/
│   │   ├── rag_graph.py             # LangGraph state machine: RAG Q&A workflow
│   │   ├── booking_graph.py         # LangGraph state machine: booking workflow
│   │   ├── administrator_graph.py   # Dispatches pending bookings to review agents
│   │   ├── booking_review_graph.py  # Per-booking review, SQL tools, and MCP persistence
│   │   └── utils.py                 # Shared graph utilities
│   ├── scripts/
│   │   ├── seed_sql.py              # Initializes DB schema + seeds relational data
│   │   └── seed_vectorstore.py      # Generates embeddings & populates Weaviate
│   ├── services/
│   │   ├── llm_chat.py              # Ollama chat service wrapper
│   │   ├── embeddings.py            # Ollama embedding generation service
│   │   ├── sql_service.py           # PostgreSQL admin + query helpers
│   │   ├── reservation_mcp.py       # MCP tool and approved-reservation file storage
│   │   ├── reservation_mcp_client.py # Administrator-to-MCP client
│   │   └── weaviate_vectorstore.py  # Weaviate vector store client
│   └── tools/
│       ├── booking_tool.py          # Booking creation/retrieval LangChain tool
│       ├── sql_tools.py             # SQL query + reservation review tools
│       └── vectorstore_tools.py     # Semantic search tool (exposed to LLM)
├── data/
│   ├── schema.sql                   # PostgreSQL DDL (tables & constraints)
│   ├── seed.sql                     # Sample parking data (lots, hours, prices)
│   └── parking_knowledge.json       # Knowledge-base documents for RAG
├── langgraph.json                   # LangGraph CLI config (graphs, env)
├── docker-compose-dev.yml           # Dev stack: API + PostgreSQL + Weaviate
├── Dockerfile                       # App image definition
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
└── .gitignore
```

## Development

### Running the Dev Server

The `langgraph dev` server auto-reloads code changes (hot-reload via volume mount). After starting:

1. The LangGraph REST API is available at [http://localhost:8123](http://localhost:8123)
2. Health check endpoint: `GET http://localhost:8123/ok`
3. Use the [LangGraph Studio](https://langchain-ai.github.io/langgraph/cloud/) or any LangGraph client to invoke graphs.
4. The administrator trigger is available at `POST http://localhost:8000/booking-processing`; it returns `202 Accepted` and processes pending reservations in the background. The customer booking flow sends this trigger asynchronously after saving the request.
5. The standalone MCP Streamable HTTP server is available at `http://localhost:8001/mcp`; it exposes `save_approved_reservation` and requires the separate `MCP_SERVER_TOKEN` bearer token. The administrator service calls it after approval. Approved bookings are appended to `data/approved_reservations.txt` as `Name | Car Number | Reservation Period | Approval Time` (UTC ISO 8601). Configure `RESERVATION_MCP_URL` on the administrator and `RESERVATIONS_FILE` on the MCP service to change their defaults.

### Available LangGraph Graphs

| Graph | Entry Point | Purpose |
|---|---|---|
| `rag_graph` | `app/graphs/rag_graph.py:get_rag_graph` | Answer parking questions using RAG (semantic search + SQL) |
| `booking_graph` | `app/graphs/booking_graph.py:get_booking_graph` | Automated booking creation |
| `administrator_graph` | `app/graphs/administrator_graph.py:get_administrator_graph` | Fan out one LLM review agent per pending request; agents can call SQL approval/rejection tools, and approved reservations are persisted through MCP |

## Database Schema

The PostgreSQL database uses the following tables:

- **parking_lots** — Parking lot identifiers
- **prices** — Hourly/daily pricing with validity windows
- **working_hours** — Recurring weekly schedules + holiday overrides
- **bookings** — Customer parking reservations with UUID primary keys (status: pending → approved/rejected), including a lease timestamp for concurrent administrator claims

See [data/schema.sql](data/schema.sql) and [data/seed.sql](data/seed.sql) for full definitions. For an existing database, apply `ALTER TABLE bookings ADD COLUMN IF NOT EXISTS review_claimed_at TIMESTAMPTZ;` before deploying the claim-based administrator workflow.

## Stopping Services

```bash
docker compose --file docker-compose-dev.yml down
```
