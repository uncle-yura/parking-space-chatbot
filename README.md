# Parking Space Chatbot

An AI-powered parking space assistant built with LangGraph, Ollama, and PostgreSQL. It provides RAG-based Q&A over parking knowledge and supports automated booking workflows.

## Quick Start

```bash
# Start all services (LangGraph API, PostgreSQL, Weaviate)
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

Copy `.env.example` to `.env` and configure the values below:

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `postgres` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRES_DB` | `parking_chatbot` | Target database name |
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
│   ├── agents/
│   │   └── customer_agent.py       # CLI entry-point for the customer agent
│   ├── graphs/
│   │   ├── rag_graph.py             # LangGraph state machine: RAG Q&A workflow
│   │   ├── booking_graph.py         # LangGraph state machine: booking workflow
│   │   └── utils.py                 # Shared graph utilities
│   ├── scripts/
│   │   ├── seed_sql.py              # Initializes DB schema + seeds relational data
│   │   └── seed_vectorstore.py      # Generates embeddings & populates Weaviate
│   ├── services/
│   │   ├── llm_chat.py              # Ollama chat service wrapper
│   │   ├── embeddings.py            # Ollama embedding generation service
│   │   ├── sql_service.py           # PostgreSQL admin + query helpers
│   │   └── weaviate_vectorstore.py  # Weaviate vector store client
│   └── tools/
│       ├── booking_tool.py          # Booking creation/retrieval LangChain tool
│       ├── sql_search_tool.py       # SQL-query tool (exposed to LLM)
│       └── vector_search_tool.py    # Semantic search tool (exposed to LLM)
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

### Available LangGraph Graphs

| Graph | Entry Point | Purpose |
|---|---|---|
| `rag_graph` | `app/graphs/rag_graph.py:get_rag_graph` | Answer parking questions using RAG (semantic search + SQL) |
| `booking_graph` | `app/graphs/booking_graph.py:get_booking_graph` | Automated booking creation |

## Database Schema

The PostgreSQL database uses the following tables:

- **parking_lots** — Parking lot identifiers
- **prices** — Hourly/daily pricing with validity windows
- **working_hours** — Recurring weekly schedules + holiday overrides
- **bookings** — Customer parking reservations (status: pending → approved/rejected)

See [data/schema.sql](data/schema.sql) and [data/seed.sql](data/seed.sql) for full definitions.

## Stopping Services

```bash
docker compose --file docker-compose-dev.yml down
```
