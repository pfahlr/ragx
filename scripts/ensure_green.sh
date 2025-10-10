#!/usr/bin/env bash
set -euo pipefail

if [[ -f requirements.txt ]]; then
  echo "[ensure_green] installing python dependencies..."
  python -m pip install -r requirements.txt
fi

echo "[ensure_green] linting..."
ruff check .

echo "[ensure_green] type checking..."
python -m mypy .

echo "[ensure_green] yaml lint..."
python -m yamllint .

echo "[ensure_green] unit + e2e tests..."
# Markers allow optional skips for gpu/native when unsupported.
# You can add -m "not slow" etc via PYTEST_ADDOPTS if needed.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest --maxfail=1 --disable-warnings ${PYTEST_ADDOPTS:-}

echo "[ensure_green] OK"
