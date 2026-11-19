"""Container entrypoint: reset the isolated test DB, then run tests/evaluations."""

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from app.scripts.seed_sql import seed
from app.services.sql_service import SqlService


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], check=True)


@contextmanager
def test_servers():
    processes = []
    try:
        for module, port in (("app.mcp_server", 8001), ("app.admin_server", 8000)):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    f"{module}:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
            )
            processes.append(process)
            deadline = monotonic() + 30
            while True:
                if process.poll() is not None or monotonic() >= deadline:
                    raise RuntimeError(f"{module} failed to start")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                        break
                except (URLError, TimeoutError):
                    sleep(0.2)
        yield
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "pytest", "retrieval", "graphs"],
    )
    args, extra = parser.parse_known_args(argv)
    if args.suite == "all" and extra:
        parser.error("Pass extra arguments to pytest, retrieval, or graphs, not all")
    if (os.environ.get("POSTGRES_HOST"), os.environ.get("POSTGRES_DB")) != (
        "test-postgres",
        "parking_chatbot_test",
    ):
        parser.error("Refusing reset outside test-postgres/parking_chatbot_test")

    service = SqlService()
    seed(service)
    if args.suite in {"all", "pytest"}:
        run_module(
            "pytest",
            "--cov",
            "--cov-report=term-missing",
            "--cov-report=html:/eval/htmlcov",
            *(extra if args.suite == "pytest" else []),
        )
        if args.suite == "pytest":
            return
        # PostgreSQL tests truncate fixtures; graph scenarios need fresh seed data.
        seed(service)

    run_module("app.scripts.seed_vectorstore")
    if args.suite in {"all", "retrieval"}:
        run_module(
            "eval.evaluate_retrieval", *(extra if args.suite == "retrieval" else [])
        )
    if args.suite in {"all", "graphs"}:
        Path(os.environ["RESERVATIONS_FILE"]).unlink(missing_ok=True)
        with test_servers():
            run_module(
                "eval.exercise_graphs",
                "--allow-writes",
                *(extra if args.suite == "graphs" else []),
            )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
