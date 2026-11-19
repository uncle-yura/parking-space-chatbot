# Stage 1 — RAG and reservation requests

Snapshot: `cb55403` — “Creation of a RAG System and Chatbot”.
[Project README](../../README.md)

## Objective

Answer parking questions using retrieved knowledge and current SQL data, and collect reservation requests through a guided conversation. This describes the historical snapshot, not today's implementation.

## Implemented changes and key files

- `app/graphs/rag_graph.py`: Ollama tool-calling loop over Weaviate knowledge, SQL prices/hours/aggregate availability, and a booking subgraph.
- `app/graphs/booking_graph.py`: interrupt/resume collection of name, registration and dates, validation, cancellation, and customer confirmation.
- `app/agents/customer_agent.py`: synchronous CLI with PostgreSQL-backed checkpoints.
- `app/tools/sql_search_tool.py`, `app/tools/vector_search_tool.py`, and `app/tools/booking_tool.py`: historical tool paths.
- `data/parking_knowledge.json`, `data/rag_questions.json`, and `eval/evaluate_retrieval.py`: knowledge, labeled questions, and Recall@3/Precision@3/MRR/latency evaluation.

## Run/check

From the repository root **at this commit**, copy `.env.example` to `.env`. Configure container-reachable PostgreSQL/Weaviate hosts and Ollama; on macOS Docker, use `http://host.docker.internal:11434`, not container localhost. Pull the configured chat and embedding models first. Docker installs the pretrained spaCy model.

**Use disposable data: SQL seeding drops and recreates the configured database.**

```sh
docker compose --file docker-compose-dev.yml up -d --build
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_sql
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.scripts.seed_vectorstore
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m app.agents.customer_agent
docker compose --file docker-compose-dev.yml run --rm langgraph-api python -m eval.evaluate_retrieval
```

## Outcome/limitations

Customer `[approve]` inserts a pending request only: no administrator processor, assignment, or status polling exists yet. Presidio filtering combines recognizers/rules with pretrained spaCy NLP; it is not purely rule-based or a privacy guarantee. Rejected Q&A messages remain in graph history here. Retrieval scores measure document retrieval, not end-to-end answer correctness.
