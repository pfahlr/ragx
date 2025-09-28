

.PHONY: unit integration e2e
unit:
	pytest -q tests/unit || true

integration:
	pytest -q tests/integration || true

e2e:
	pytest -q tests/e2e || true
