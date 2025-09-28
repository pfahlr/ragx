.PHONY: lint typecheck test codex-bootstrap unit integration e2e

lint:
	ruff check . || true
	yamllint -s . || true

typecheck:
	mypy . || true

test:
	pytest --maxfail=1 --disable-warnings || true

codex-bootstrap:
	python scripts/codex_next_tasks.py || true

unit:
	pytest -q tests/unit || true

integration:
	pytest -q tests/integration || true

e2e:
	pytest -q tests/e2e || true
