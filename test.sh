#!/bin/sh
set -eu

trap 'docker compose --file docker-compose-dev.yml stop test-postgres test-weaviate' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker compose --file docker-compose-dev.yml run --rm graph-tests "$@"
