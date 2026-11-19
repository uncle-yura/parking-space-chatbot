import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt

from app.graphs.booking_graph import build_booking_graph
from eval import exercise_graphs
from eval.exercise_graphs import Statistics, check_write_safety, drive


def test_statistics_aggregates_node_timings_and_ignores_nested_chains(monkeypatch):
    statistics = Statistics()
    # Twenty samples distinguish nearest-rank p95 from the maximum.
    times = iter(value for duration in range(1, 21) for value in (0, duration))
    monkeypatch.setattr(exercise_graphs, "perf_counter", lambda: next(times))
    metadata = {"exercise_graph": "booking", "langgraph_node": "collect"}
    for _ in range(20):
        run_id = uuid4()
        statistics.on_chain_start(
            None, {}, run_id=run_id, metadata=metadata, name="collect"
        )
        statistics.on_chain_end({}, run_id=run_id)
        statistics.on_chain_end({}, run_id=run_id)

    for metadata, name in [
        (metadata, "nested_runnable"),
        ({"exercise_graph": "booking"}, "LangGraph"),
        (None, "LangGraph"),
    ]:
        run_id = uuid4()
        statistics.on_chain_start(None, {}, run_id=run_id, metadata=metadata, name=name)
        statistics.on_chain_end({}, run_id=run_id)

    assert statistics.active == {}
    assert statistics.report() == [
        {
            "graph": "booking",
            "kind": "node",
            "name": "collect",
            "calls": 20,
            "errors": 0,
            "interrupts": 0,
            "mean_ms": 10500,
            "p95_ms": 19000,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    ]


def test_statistics_interrupts_are_not_errors_and_graphs_are_separate():
    statistics = Statistics()
    for graph, error in [
        ("booking", GraphInterrupt(())),
        ("booking", RuntimeError("failed")),
        ("rag", None),
    ]:
        run_id = uuid4()
        statistics.on_chain_start(
            None,
            {},
            run_id=run_id,
            metadata={"exercise_graph": graph, "langgraph_node": "collect"},
            name="collect",
        )
        if error is None:
            statistics.on_chain_end({}, run_id=run_id)
        else:
            statistics.on_chain_error(error, run_id=run_id)

    rows = statistics.report()
    assert [
        (row["graph"], row["calls"], row["errors"], row["interrupts"]) for row in rows
    ] == [
        ("booking", 2, 1, 1),
        ("rag", 1, 0, 0),
    ]
    assert statistics.active == {}


def test_statistics_tool_success_error_status_and_exception():
    statistics = Statistics()
    for result in [
        "ok",
        ToolMessage(content="failed", tool_call_id="1", status="error"),
        RuntimeError("failed"),
    ]:
        run_id = uuid4()
        statistics.on_tool_start(
            {"name": "lookup"},
            "private input",
            run_id=run_id,
            metadata={"exercise_graph": "rag"},
        )
        if isinstance(result, Exception):
            statistics.on_tool_error(result, run_id=run_id)
        else:
            statistics.on_tool_end(result, run_id=run_id)

    (row,) = statistics.report()
    assert (row["graph"], row["kind"], row["name"]) == ("rag", "tool", "lookup")
    assert (row["calls"], row["errors"], row["interrupts"]) == (3, 2, 0)
    assert row["total_tokens"] == 0
    assert statistics.requested_tools == set()
    assert statistics.active == {}


def test_statistics_llm_usage_and_requested_tools_across_generations():
    statistics = Statistics()

    def generation(input_tokens, output_tokens, names):
        return ChatGeneration(
            message=AIMessage(
                content="",
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                tool_calls=[
                    {"name": name, "args": {}, "id": str(uuid4())} for name in names
                ],
            )
        )

    for generations in [
        [
            [generation(2, 3, ["start_booking_process", "lookup"])],
            [generation(5, 7, ["lookup"])],
        ],
        [
            [
                generation(11, 13, ["other"]),
                ChatGeneration(message=AIMessage(content="done")),
                Generation(text="plain"),
            ]
        ],
    ]:
        run_id = uuid4()
        statistics.on_chat_model_start(
            {"name": "serialized-model"},
            [],
            run_id=run_id,
            metadata={"exercise_graph": "rag"},
            name="chat",
        )
        statistics.on_llm_end(LLMResult(generations=generations), run_id=run_id)

    run_id = uuid4()
    statistics.on_chat_model_start(
        None, [], run_id=run_id, metadata={"exercise_graph": "rag"}, name="chat"
    )
    statistics.on_llm_error(RuntimeError("failed"), run_id=run_id)
    (row,) = statistics.report()
    assert (row["graph"], row["kind"], row["name"]) == ("rag", "llm", "chat")
    assert (row["calls"], row["errors"], row["interrupts"]) == (3, 1, 0)
    assert (row["input_tokens"], row["output_tokens"], row["total_tokens"]) == (
        18,
        23,
        41,
    )
    assert statistics.requested_tools == {"start_booking_process", "lookup", "other"}
    assert not any(item["kind"] == "tool" for item in statistics.report())
    assert statistics.active == {}


@pytest.mark.parametrize(
    "serialized, expected", [({"name": "model"}, "model"), (None, "chat_model")]
)
def test_statistics_defaults_for_missing_metadata_and_model_name(serialized, expected):
    statistics = Statistics()
    run_id = uuid4()
    statistics.on_chat_model_start(serialized, [], run_id=run_id)
    statistics.on_llm_end(LLMResult(generations=[]), run_id=run_id)
    (row,) = statistics.report()
    assert (row["graph"], row["kind"], row["name"]) == ("unknown", "llm", expected)
    assert row["calls"] == 1
    assert row["total_tokens"] == 0


class ScriptedGraph:
    def __init__(self, *states):
        self.states = iter(states)
        self.calls = []

    async def ainvoke(self, graph_input, *, config, context):
        self.calls.append((graph_input, config, context))
        return next(self.states)


def test_drive_returns_uninterrupted_state_without_resuming():
    final = {"messages": ["done"]}
    graph = ScriptedGraph(final)
    graph_input, context, config = {}, object(), {"configurable": {"thread_id": "test"}}
    assert asyncio.run(drive(graph, graph_input, context, config, {})) is final
    assert graph.calls == [(graph_input, config, context)]


def test_drive_resumes_multiple_interrupts_by_id_and_consumes_scripts_in_order():
    graph = ScriptedGraph(
        {
            "__interrupt__": (
                Interrupt(value={"field": "name", "type": "input"}, id="name-1"),
                Interrupt(value={"type": "approval"}, id="approval-1"),
            )
        },
        {"__interrupt__": (Interrupt(value={"field": "name"}, id="name-2"),)},
        {"done": True},
    )
    context, config, graph_input = object(), {"thread_id": "test"}, {"messages": []}
    responses = {"name": ["bad", "Taylor Example"], "approval": ["[reject]"]}
    assert asyncio.run(
        drive(graph, graph_input, context, config, responses, max_resumes=2)
    ) == {"done": True}
    assert graph.calls[0] == (graph_input, config, context)
    assert all(
        call_config is config and call_context is context
        for _, call_config, call_context in graph.calls
    )
    assert all(isinstance(call[0], Command) for call in graph.calls[1:])
    assert [call[0].resume for call in graph.calls[1:]] == [
        {"name-1": "bad", "approval-1": "[reject]"},
        {"name-2": "Taylor Example"},
    ]
    assert responses == {"name": ["bad", "Taylor Example"], "approval": ["[reject]"]}


@pytest.mark.parametrize(
    "responses, repeats, cause",
    [
        ({}, 1, KeyError),
        ({"name": []}, 1, StopIteration),
        ({"name": ["bad"]}, 2, StopIteration),
    ],
)
def test_drive_fails_closed_on_missing_or_exhausted_script(responses, repeats, cause):
    graph = ScriptedGraph(
        *[
            {"__interrupt__": (Interrupt(value={"field": "name"}, id=f"name-{index}"),)}
            for index in range(repeats)
        ]
    )
    with pytest.raises(ValueError, match="No scripted response left for 'name'") as exc:
        asyncio.run(drive(graph, {}, None, {}, responses))
    assert isinstance(exc.value.__cause__, cause)
    assert len(graph.calls) == repeats


@pytest.mark.parametrize("max_resumes", [0, 2])
def test_drive_limits_interrupt_resumptions(max_resumes):
    graph = ScriptedGraph(
        *[
            {"__interrupt__": (Interrupt(value={"field": "name"}, id=f"name-{index}"),)}
            for index in range(max_resumes + 1)
        ]
    )
    with pytest.raises(RuntimeError, match=f"Exceeded {max_resumes} interrupt resumes"):
        asyncio.run(
            drive(
                graph,
                {},
                None,
                {},
                {"name": ["retry"] * (max_resumes + 1)},
                max_resumes=max_resumes,
            )
        )
    assert len(graph.calls) == max_resumes + 1


def test_drive_compiled_booking_graph_retries_and_rejects_without_services():
    graph = build_booking_graph().compile(checkpointer=InMemorySaver())
    statistics = Statistics()
    config = {
        "configurable": {"thread_id": "offline-exercise-booking"},
        "callbacks": [statistics],
        "metadata": {"exercise_graph": "booking"},
    }
    state = asyncio.run(
        drive(
            graph,
            {"messages": []},
            SimpleNamespace(),
            config,
            {
                "full_name": ["x", "Taylor Example"],
                "car_number": ["ABC1234"],
                "reservation_start": ["2030-01-01 10:00"],
                "reservation_end": ["2030-01-01 12:00"],
                "approval": ["[reject]"],
            },
        )
    )
    assert "__interrupt__" not in state
    assert state["full_name"] == "Taylor Example"
    assert state["booking_approved"] is False
    assert "booking_id" not in state
    assert "cancelled" in state["messages"][-1].content.lower()
    rows = {row["name"]: row for row in statistics.report()}
    assert rows["get_full_name"]["interrupts"] == 2
    assert rows["confirm_booking"]["interrupts"] == 1
    assert rows["cancel_booking"]["calls"] == 1
    assert "finish_booking" not in rows
    assert all(row["errors"] == 0 for row in rows.values())
    assert statistics.active == {}


@pytest.mark.parametrize(
    "database",
    [
        None,
        "parking_chatbot",
        "production",
        "testing",
        "test",
        "my_test_backup",
        "TEST_parking",
    ],
)
def test_check_write_safety_rejects_non_test_database(monkeypatch, database):
    if database is None:
        monkeypatch.delenv("POSTGRES_DB", raising=False)
    else:
        monkeypatch.setenv("POSTGRES_DB", database)
    check_write_safety(False)
    with pytest.raises(
        ValueError, match="--allow-writes requires POSTGRES_DB=test_\\* or \\*_test"
    ):
        check_write_safety(True)


@pytest.mark.parametrize(
    "database", ["test_parking", "parking_test", "test_parking_test"]
)
def test_check_write_safety_allows_explicit_test_database(monkeypatch, database):
    monkeypatch.setenv("POSTGRES_DB", database)
    check_write_safety(True)
