# Stage 4 — Integrated orchestration and checks

Snapshot: `e42acf3` — “Orchestrating All Components via LangGraph”.
[Project README](../../README.md)

## Objective

Harden the integrated RAG → customer confirmation → automated administrator review → MCP workflow and make its behavior testable and observable.

## Implemented changes and key files

- `app/graphs/rag_graph.py`: explicit booking tool handoff, UTC-aware prompts, clearer scope/tool-selection rules, and removal of the detected sensitive message from current graph history.
- `app/graphs/booking_review_graph.py`, `app/services/sql_service.py`, and `app/services/reservation_mcp.py`: distinguish SQL approval from successful persistence, track `mcp_saved_at`, reclaim approved-but-unsaved bookings on later administrator runs, and deduplicate file writes by reservation details.
- `app/services/llm_chat.py` and `app/services/embeddings.py`: independently selectable Ollama/OpenAI providers.
- `tests/`, `app/scripts/run_tests.py`, `test.sh`, and `docker-compose-dev.yml`: unit/integration checks, coverage, and isolated test services.
- `eval/exercise_graphs.py`: scripted graph/MCP scenarios with node/tool/model timings, token counts, errors and missing-coverage reporting; expanded retrieval data and saved reports under `eval/`.

## Run/check

From this snapshot's repository root, copy `.env.example` to `.env`. Configure reachable models/providers; use independent administrator/MCP secrets for the application. Docker must reach Ollama (macOS: `http://host.docker.internal:11434`) or OpenAI.

```sh
docker compose --file docker-compose-dev.yml build graph-tests
sh test.sh pytest
sh test.sh retrieval
sh test.sh graphs --repeat 3
# Alternatively, run all three suites:
sh test.sh all
```

The runner resets the isolated test database, seeds test vectors for evaluations, starts local administrator/MCP services for scenarios, and writes reports under `eval/`. The wrapper stops test databases on exit.

## Outcome/limitations

These are sequential scenario exercises, **not a concurrent load test** or throughput/scalability evidence. Administrator reviewers remain automated LLMs; there is no human decision UI. Presidio combines recognizers/rules with pretrained spaCy NLP, not rules alone; message removal does not guarantee erasure from earlier checkpoints/traces. SQL/file persistence remains non-atomic. Existing application databases need `review_claimed_at` and `mcp_saved_at`. OpenAI incurs costs and external data transfer; changing embedding models requires rebuilding the index.
