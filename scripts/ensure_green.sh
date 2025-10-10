#!/usr/bin/env bash
set -euo pipefail

echo "[ensure_green] linting..."
python -m ruff check apps pkgs ragcore scripts tests

echo "[ensure_green] type checking..."
python -m mypy --explicit-package-bases apps ragcore scripts

echo "[ensure_green] yaml lint..."
python -m yamllint codex/specs flows

echo "[ensure_green] unit + e2e tests..."
# Markers allow optional skips for gpu/native when unsupported.
# You can add -m "not slow" etc via PYTEST_ADDOPTS if needed.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest --maxfail=1 --disable-warnings ${PYTEST_ADDOPTS:-}

echo "[ensure_green] OK"
