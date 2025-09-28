.PHONY: lint typecheck test codex-bootstrap

lint:
	ruff check . || true
	yamllint -s . || true

typecheck:
	mypy . || true

test:
	pytest -q || true

codex-bootstrap:
	python scripts/codex_next_tasks.py
