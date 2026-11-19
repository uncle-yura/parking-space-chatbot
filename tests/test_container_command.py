import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_test_command_reuses_image_and_stops_databases(tmp_path, exit_code):
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'case "$*" in *" run "*) exit "$RUN_EXIT" ;; esac\n'
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    result = subprocess.run(
        ["sh", "test.sh", "graphs", "--repeat", "3"],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.defpath}",
            "DOCKER_LOG": str(log),
            "RUN_EXIT": str(exit_code),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    assert log.read_text().splitlines() == [
        "compose --file docker-compose-dev.yml run --rm graph-tests graphs --repeat 3",
        "compose --file docker-compose-dev.yml stop test-postgres test-weaviate",
    ]
