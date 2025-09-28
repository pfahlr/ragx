#!/usr/bin/env bash
set -euo pipefail

echo "[ensure_green] linting..."
ruff check .

echo "[ensure_green] type checking..."
mypy .

echo "[ensure_green] yaml lint..."
yamllint .

echo "[ensure_green] unit + e2e tests..."
# Markers allow optional skips for gpu/native when unsupported.
# You can add -m "not slow" etc via PYTEST_ADDOPTS if needed.
pytest --maxfail=1 --disable-warnings ${PYTEST_ADDOPTS:-}

echo "[ensure_green] OK"
