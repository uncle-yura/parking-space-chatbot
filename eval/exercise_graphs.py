"""Exercise real parking agents and collect node/tool/LLM statistics.

Run with python -m eval.exercise_graphs --help. Services and tracing configuration
are read from the environment, just like the customer CLI.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt
from langgraph.types import Command


class Statistics(BaseCallbackHandler):
    """Collect timings without retaining prompts, personal data, or tool outputs."""

    def __init__(self):
        self.active = {}
        self.rows = defaultdict(list)
        self.requested_tools = set()
        self.lock = Lock()

    def _start(self, kind, name, run_id, metadata):
        metadata = metadata or {}
        key = (metadata.get("exercise_graph", "unknown"), kind, name)
        with self.lock:
            self.active[run_id] = (key, perf_counter())

    def _end(self, run_id, *, error=False, interrupted=False, usage=None):
        with self.lock:
            entry = self.active.pop(run_id, None)
            if entry:
                key, started = entry
                self.rows[key].append(
                    {
                        "duration_ms": (perf_counter() - started) * 1000,
                        "error": error,
                        "interrupted": interrupted,
                        "usage": usage or {},
                    }
                )

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node and kwargs.get("name") == node:
            self._start("node", node, run_id, metadata)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        interrupted = isinstance(error, GraphInterrupt)
        self._end(run_id, error=not interrupted, interrupted=interrupted)

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):
        self._start("tool", serialized.get("name", "unknown"), run_id, metadata)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id, error=getattr(output, "status", None) == "error")

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=True)

    def on_chat_model_start(
        self, serialized, messages, *, run_id, metadata=None, **kwargs
    ):
        name = kwargs.get("name") or (serialized or {}).get("name", "chat_model")
        self._start("llm", name, run_id, metadata)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = defaultdict(int)
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    usage[key] += (getattr(message, "usage_metadata", None) or {}).get(
                        key, 0
                    )
                with self.lock:
                    self.requested_tools.update(
                        call["name"] for call in getattr(message, "tool_calls", [])
                    )
        self._end(run_id, usage=dict(usage))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=True)

    def report(self):
        rows = []
        for (graph, kind, name), samples in sorted(self.rows.items()):
            durations = sorted(sample["duration_ms"] for sample in samples)
            rows.append(
                {
                    "graph": graph,
                    "kind": kind,
                    "name": name,
                    "calls": len(samples),
                    "errors": sum(sample["error"] for sample in samples),
                    "interrupts": sum(sample["interrupted"] for sample in samples),
                    "mean_ms": sum(durations) / len(durations),
                    "p95_ms": durations[(95 * len(durations) + 99) // 100 - 1],
                    "input_tokens": sum(
                        s["usage"].get("input_tokens", 0) for s in samples
                    ),
                    "output_tokens": sum(
                        s["usage"].get("output_tokens", 0) for s in samples
                    ),
                    "total_tokens": sum(
                        s["usage"].get("total_tokens", 0) for s in samples
                    ),
                }
            )
        return rows


async def drive(graph, graph_input, context, config, responses, *, max_resumes=20):
    """Resume by interrupt ID; fail closed on unexpected/exhausted input scripts."""
    responses = {field: iter(values) for field, values in responses.items()}
    for _ in range(max_resumes + 1):
        state = await graph.ainvoke(graph_input, config=config, context=context)
        interrupts = state.get("__interrupt__", ())
        if not interrupts:
            return state
        resume = {}
        for item in interrupts:
            field = item.value.get("field", item.value.get("type"))
            try:
                resume[item.id] = next(responses[field])
            except (KeyError, StopIteration) as exc:
                raise ValueError(f"No scripted response left for {field!r}") from exc
        graph_input = Command(resume=resume)
    raise RuntimeError(f"Exceeded {max_resumes} interrupt resumes")


def check_write_safety(allow_writes):
    if allow_writes:
        database = os.environ.get("POSTGRES_DB", "parking_chatbot")
        if not (database.startswith("test_") or database.endswith("_test")):
            raise ValueError("--allow-writes requires POSTGRES_DB=test_* or *_test")


async def exercise(args, statistics, results, expected_nodes):
    # Delay service imports/initialization until after CLI safety checks.
    from langgraph.checkpoint.memory import InMemorySaver
    from presidio_analyzer import AnalyzerEngine

    from app.graphs.booking_graph import build_booking_graph
    from app.graphs.rag_graph import TOOLS, build_rag_graph
    from app.graphs.utils import (
        AdministratorGraphContextSchema,
        RagGraphContextSchema,
    )
    from app.services.embeddings import EmbeddingsService
    from app.services.llm_chat import LLMService
    from app.services.sql_service import SqlService
    from app.services.weaviate_vectorstore import WeaviateService

    llm = LLMService().llm
    sql_service = SqlService()
    embeddings = EmbeddingsService().embeddings
    vector_service = WeaviateService("ParkingKnowledge", embeddings)
    try:
        context = RagGraphContextSchema(
            llm=llm,
            sql_service=sql_service,
            pii_analyzer=AnalyzerEngine(),
            embeddings=embeddings,
            vectorstore=vector_service.get_vectorstore(),
            thread_id="exercise",
        )
        graphs = {
            "rag_graph": build_rag_graph().compile(checkpointer=InMemorySaver()),
            "booking_graph": build_booking_graph().compile(
                checkpointer=InMemorySaver()
            ),
        }
        admin_context = AdministratorGraphContextSchema(
            llm=llm, sql_service=sql_service
        )
        if args.allow_writes:
            from app.graphs.administrator_graph import build_administrator_graph
            from app.graphs.booking_review_graph import build_booking_review_graph

            graphs["administrator_graph"] = (
                await build_administrator_graph()
            ).compile()
            graphs["booking_review_graph"] = (
                await build_booking_review_graph()
            ).compile()
        for name, graph in graphs.items():
            expected_nodes[name] = sorted(
                set(graph.get_graph().nodes) - {"__start__", "__end__"}
            )

        async def run(name, scenario, graph_input, responses=None, admin=False):
            thread_id = str(uuid4())
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [statistics],
                "tags": ["graph-exercise", scenario],
                "metadata": {"exercise_graph": name, "scenario": scenario},
                "run_name": f"exercise:{name}:{scenario}",
                "recursion_limit": 50,
            }
            started = perf_counter()
            try:
                state = await asyncio.wait_for(
                    drive(
                        graphs[name],
                        graph_input,
                        admin_context
                        if admin
                        else context.model_copy(update={"thread_id": thread_id}),
                        config,
                        responses or {},
                    ),
                    timeout=args.timeout,
                )
                error = None
            except Exception as exc:
                state = {}
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "graph": name,
                    "scenario": scenario,
                    "thread_id": thread_id,
                    "duration_ms": (perf_counter() - started) * 1000,
                    "error": error,
                }
            )
            print(
                f"{name}/{scenario}: {'FAILED' if error else 'completed'}", flush=True
            )
            return state

        start = (datetime.now(UTC) + timedelta(days=7)).replace(
            hour=10, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(hours=2)
        valid = {
            "full_name": ["Synthetic Example"],
            "car_number": ["TEST123"],
            "reservation_start": [start.strftime("%Y-%m-%d %H:%M")],
            "reservation_end": [end.strftime("%Y-%m-%d %H:%M")],
            "approval": ["[reject]"],
        }
        for repetition in range(args.repeat):
            questions = {
                "knowledge": "What parking facilities and security features are available?",
                "prices": "What are the current parking prices?",
                "hours": "What are the parking opening hours?",
                "availability": "How many parking spaces are free right now?",
                "interval": f"How many spaces are free from {start.isoformat()} to {end.isoformat()}?",
                "status": "Check the status of my reservation.",
                "booking-handoff": "I want to reserve a parking space.",
                "privacy": "My email address is synthetic@example.com",
            }
            for scenario, question in questions.items():
                await run(
                    "rag_graph",
                    f"{repetition}:{scenario}",
                    {"messages": [("human", question)]},
                    {"full_name": ["[cancel]"]},
                )
            await run("booking_graph", f"{repetition}:reject", {"messages": []}, valid)
            await run(
                "booking_graph",
                f"{repetition}:validation",
                {"messages": []},
                {
                    **valid,
                    "full_name": ["x", "Synthetic Example"],
                    "car_number": ["x", "TEST123"],
                    "reservation_start": [
                        "invalid",
                        *valid["reservation_start"],
                        *valid["reservation_start"],
                    ],
                    "reservation_end": [
                        "invalid",
                        *valid["reservation_start"],
                        *valid["reservation_end"],
                    ],
                },
            )
            if args.allow_writes:
                await run(
                    "booking_graph",
                    f"{repetition}:submit",
                    {"messages": []},
                    {**valid, "approval": ["[approve]"]},
                )
                # These fixtures belong only to the explicitly selected test database.
                invalid = sql_service.create_booking(
                    customer_name="", car_number="TEST123", start_at=start, end_at=end
                )
                await run(
                    "booking_review_graph",
                    f"{repetition}:invalid",
                    {
                        "booking": {
                            **invalid,
                            "customer_name": "",
                            "car_number": "TEST123",
                            "start_at": start,
                            "end_at": end,
                        },
                        "messages": [],
                    },
                    admin=True,
                )
                sql_service.create_booking(
                    customer_name="Synthetic Example",
                    car_number="TEST456",
                    start_at=start,
                    end_at=end,
                )
                await run(
                    "administrator_graph",
                    f"{repetition}:process",
                    {"messages": []},
                    admin=True,
                )
        return {tool.name for tool in TOOLS} | {
            "approve_reservation",
            "reject_reservation",
            "save_approved_reservation",
        }
    finally:
        vector_service.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=180, help="Maximum seconds per scenario"
    )
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Create/process test reservations and write to MCP",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/graph_statistics.json")
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout <= 0:
        parser.error("--repeat and --timeout must be positive")
    try:
        check_write_safety(args.allow_writes)
    except ValueError as exc:
        parser.error(str(exc))
    statistics, results, expected_nodes = Statistics(), [], {}
    error = None
    expected_tools = set()
    try:
        expected_tools = asyncio.run(
            exercise(args, statistics, results, expected_nodes)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"Exercise failed: {error}")
    rows = statistics.report()
    missing_nodes = {
        graph: sorted(
            set(nodes)
            - {
                row["name"]
                for row in rows
                if row["graph"] == graph and row["kind"] == "node"
            }
        )
        for graph, nodes in expected_nodes.items()
    }
    executed = {row["name"] for row in rows if row["kind"] == "tool"}
    # start_booking_process routes to a subgraph instead of executing via ToolNode.
    covered = executed | (statistics.requested_tools & {"start_booking_process"})
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "allow_writes": args.allow_writes,
        "skipped_graphs": []
        if args.allow_writes
        else ["administrator_graph", "booking_review_graph"],
        "error": error,
        "scenarios": results,
        "statistics": rows,
        "missing_nodes": missing_nodes,
        "requested_tools": sorted(statistics.requested_tools),
        "executed_tools": sorted(executed),
        "missing_tools": sorted(expected_tools - covered),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}")
    print(f"Missing tools: {report['missing_tools']}")
    return int(
        bool(
            error
            or any(result["error"] for result in results)
            or any(row["errors"] for row in rows)
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
