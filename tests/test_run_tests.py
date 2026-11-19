import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, call, sentinel

import pytest

from app.scripts import run_tests

PYTEST = (
    "pytest",
    "--cov",
    "--cov-report=term-missing",
    "--cov-report=html:/eval/htmlcov",
)
RESERVATIONS = "/offline-test/reservations.json"


@pytest.fixture
def effects(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "test-postgres")
    monkeypatch.setenv("POSTGRES_DB", "parking_chatbot_test")
    monkeypatch.setenv("RESERVATIONS_FILE", RESERVATIONS)
    effects = Mock()
    effects.SqlService.return_value = sentinel.service
    for name in ("SqlService", "seed", "run_module"):
        monkeypatch.setattr(run_tests, name, getattr(effects, name))

    def unlink(path, *, missing_ok=False):
        effects.unlink(path, missing_ok=missing_ok)

    @contextmanager
    def servers():
        effects.servers_start()
        try:
            yield
        finally:
            effects.servers_stop()

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(run_tests, "test_servers", servers)
    return effects


def test_default_all_orders_steps_and_wraps_graphs_in_servers(effects):
    run_tests.main([])

    assert effects.mock_calls == [
        call.SqlService(),
        call.seed(sentinel.service),
        call.run_module(*PYTEST),
        call.seed(sentinel.service),
        call.run_module("app.scripts.seed_vectorstore"),
        call.run_module("eval.evaluate_retrieval"),
        call.unlink(Path(RESERVATIONS), missing_ok=True),
        call.servers_start(),
        call.run_module("eval.exercise_graphs", "--allow-writes"),
        call.servers_stop(),
    ]


def test_pytest_forwards_selection_without_vector_or_graph_steps(effects):
    run_tests.main(["pytest", "tests/test_booking_workflow.py", "-k", "booking", "-q"])

    assert effects.mock_calls == [
        call.SqlService(),
        call.seed(sentinel.service),
        call.run_module(
            *PYTEST, "tests/test_booking_workflow.py", "-k", "booking", "-q"
        ),
    ]


def test_graphs_forwards_repeat(effects):
    run_tests.main(["graphs", "--repeat", "3"])

    assert effects.mock_calls == [
        call.SqlService(),
        call.seed(sentinel.service),
        call.run_module("app.scripts.seed_vectorstore"),
        call.unlink(Path(RESERVATIONS), missing_ok=True),
        call.servers_start(),
        call.run_module("eval.exercise_graphs", "--allow-writes", "--repeat", "3"),
        call.servers_stop(),
    ]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("POSTGRES_HOST", "localhost"),
        ("POSTGRES_DB", "parking_chatbot"),
        ("POSTGRES_HOST", None),
        ("POSTGRES_DB", None),
    ],
)
def test_unsafe_database_refused_before_seed(
    effects, monkeypatch, capsys, variable, value
):
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)

    with pytest.raises(SystemExit) as exc:
        run_tests.main([])

    assert exc.value.code == 2
    assert (
        "Refusing reset outside test-postgres/parking_chatbot_test"
        in capsys.readouterr().err
    )
    assert effects.mock_calls == []


@pytest.mark.parametrize(
    ("failed_module", "completed_calls"),
    [
        ("pytest", 3),
        ("app.scripts.seed_vectorstore", 5),
        ("eval.evaluate_retrieval", 6),
    ],
)
def test_failed_child_propagates_without_later_steps(
    effects, failed_module, completed_calls
):
    failure = subprocess.CalledProcessError(7, [failed_module])

    def run_module(module, *args):
        if module == failed_module:
            raise failure

    effects.run_module.side_effect = run_module
    expected = [
        call.SqlService(),
        call.seed(sentinel.service),
        call.run_module(*PYTEST),
        call.seed(sentinel.service),
        call.run_module("app.scripts.seed_vectorstore"),
        call.run_module("eval.evaluate_retrieval"),
    ]

    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_tests.main([])

    assert exc.value is failure
    assert effects.mock_calls == expected[:completed_calls]
