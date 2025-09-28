.PHONY: lint typecheck test codex-bootstrap check unit integration e2e acceptance

PYTHON ?= python3
TASK_LIMIT ?= 5

lint:
$(PYTHON) -m ruff check .
yamllint -s codex/ flows/

typecheck:
$(PYTHON) -m mypy .

test:
$(PYTHON) -m pytest --maxfail=1 --disable-warnings

check: lint typecheck test

codex-bootstrap:
$(PYTHON) -m scripts.codex_next_tasks --limit $(TASK_LIMIT)

unit:
$(PYTHON) -m pytest -q tests/unit

integration:
$(PYTHON) -m pytest -q tests/integration

e2e:
$(PYTHON) -m pytest -q tests/e2e

acceptance:
$(PYTHON) -m pytest -q tests/acceptance
