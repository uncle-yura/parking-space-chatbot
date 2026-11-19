# Parking Space Chatbot

An AI-powered parking space assistant built with LangGraph, Ollama, and PostgreSQL. It provides RAG-based Q&A over parking knowledge and supports automated booking workflows.

## Stage Summaries and Presentation

Each summary describes the implementation at its corresponding commit:

| Stage | Commit | Summary |
|---|---|---|
| 1 — RAG and reservation requests | `cb55403` | [Stage 1 README](docs/stage-1/README.md) |
| 2 — Administrator processing | `4e91d5b` | [Stage 2 README](docs/stage-2/README.md) |
| 3 — MCP persistence | `8f3e6e3` | [Stage 3 README](docs/stage-3/README.md) |
| 4 — Integrated orchestration and checks | `91e2a65` | [Stage 4 README](docs/stage-4/README.md) |

[PowerPoint presentation](docs/presentation/parking-space-chatbot.pptx) — 13 editable slides with speaker notes. See the [presentation README](docs/presentation/README.md) for contents and rebuild instructions.

**Implementation note:** administrator decisions are currently automated (deterministic in Stage 2, LLM-assisted from Stage 3), not made by a human approval interface. Existing scenario evaluations are not concurrent load tests.

## Quick Start

```bash
# Start all services (LangGraph API, administrator + MCP servers, PostgreSQL, Weaviate)
docker compose --file docker-compose-dev.yml up -d

# Seed the relational database with schema & sample data
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_sql

# Build vector embeddings and populate the vector store
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_vectorstore

# Run the customer agent CLI directly
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.agents.customer_agent
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [docker compose](https://docs.docker.com/compose/) v2+
- [Ollama](https://ollama.ai/) running locally, or an OpenAI API key

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
| `LLM_PROVIDER` | `ollama` | Chat provider: `ollama` or `openai` |
| `LLM_TEMPERATURE` | (unset) | Optional numeric chat temperature; omitted when unset or empty. Set only for models that support it. |
| `EMBEDDINGS_PROVIDER` | `ollama` | Embeddings provider, selected independently |
| `OPENAI_API_KEY` | (required for OpenAI) | OpenAI API key |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDINGS_MODEL` | `text-embedding-3-small` | OpenAI embeddings model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_LLM_MODEL` | `qwen3.6:35b` | Model used for chat/inference |
| `OLLAMA_EMBEDDINGS_MODEL` | `qwen3-embedding:8b` | Model used for embedding generation |
| `WEAVIATE_HOST` | `localhost` | Weaviate hostname |
| `WEAVIATE_LOCATION` | auto | Optional trace tag override: `local` or `external` |
| `WEAVIATE_PORT` | `8080` | Weaviate HTTP port |
| `WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |

To use OpenAI, set in `.env`:

```dotenv
LLM_PROVIDER=openai
EMBEDDINGS_PROVIDER=openai
OPENAI_API_KEY=your-key
```

Providers can be mixed. OpenAI requests are paid and send chat/tool data or
embedding inputs to OpenAI. After changing the embeddings provider/model,
rebuild the vector index with `app.scripts.seed_vectorstore` before querying it;
stored vectors must use the same model as query vectors. Rebuild Docker images
after updating dependencies.

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

### Tests and Evaluations

```bash
# Build once; repeat only after changing dependencies or Dockerfile
docker compose --file docker-compose-dev.yml build graph-tests

# All tests + retrieval eval + graph/MCP scenarios; stop test databases on exit
sh test.sh all

# All pytest tests, including PostgreSQL integration tests and coverage
sh test.sh pytest

# One test file or test case
sh test.sh pytest tests/test_booking_workflow.py
sh test.sh pytest -k test_booking_graph_retries

# PostgreSQL integration tests only
sh test.sh pytest -m integration

# Retrieval evaluation only
sh test.sh retrieval

# Graph/MCP scenarios only
sh test.sh graphs --repeat 3
```

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

See [data/schema.sql](data/schema.sql) and [data/seed.sql](data/seed.sql) for full definitions. For an existing database, apply `ALTER TABLE bookings ADD COLUMN IF NOT EXISTS review_claimed_at TIMESTAMPTZ;` and `ALTER TABLE bookings ADD COLUMN IF NOT EXISTS mcp_saved_at TIMESTAMPTZ;` before deploying the claim-based administrator workflow. Approved bookings whose MCP write has not been marked complete are reclaimed by later administrator runs; file writes are idempotent for the same reservation details.

## Stopping Services

```bash
docker compose --file docker-compose-dev.yml down
```
