.PHONY: lint typecheck test codex-bootstrap unit integration e2e

lint:
	ruff check .
	yamllint -s .

typecheck:
	mypy .

test:
	pytest -q

codex-bootstrap:
	python scripts/codex_next_tasks.py

unit:
	pytest -q tests/unit || true

integration:
	pytest -q tests/integration || true

e2e:
	pytest -q tests/e2e || true
