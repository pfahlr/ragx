"""Property-based contracts for MCP envelope validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError

try:  # pragma: no cover - hypothesis may be stubbed in CI
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - fallback to pytest.skip
    given = None
    st = None

from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

SCHEMA_BASE = Path("codex/specs/schemas")
FIELDS = ["id", "jsonrpc", "method", "params"]


def _schema_registry() -> SchemaRegistry:
    return SchemaRegistry(schema_dir=SCHEMA_BASE)


def _valid_payload() -> dict[str, Any]:
    path = Path("tests/fixtures/mcp/envelope/invalid_missing_method.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["method"] = "mcp.tool.invoke"
    return data


def _run_missing_field_case(missing_field: str) -> None:
    payload = dict(_valid_payload())
    payload.pop(missing_field)
    validator = _schema_registry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)


if given is not None:  # pragma: no branch - guarded import

    if hasattr(st, "sampled_from"):

        @pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
        @given(st.sampled_from(FIELDS))
        def test_missing_required_field_triggers_validation_error(missing_field: str) -> None:
            _run_missing_field_case(missing_field)

    else:

        @pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
        @pytest.mark.parametrize("missing_field", FIELDS)
        def test_missing_required_field_triggers_validation_error(missing_field: str) -> None:
            _run_missing_field_case(missing_field)

    @pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
    @given(st.text(min_size=1))
    def test_params_must_be_object(random_value: str) -> None:
        payload = dict(_valid_payload())
        payload["params"] = random_value
        validator = _schema_registry().load_envelope()
        with pytest.raises(ValidationError):
            validator.validate(payload)

else:  # pragma: no cover - executed only without hypothesis

    @pytest.mark.skip("hypothesis is not installed")
    def test_property_based_contracts_require_hypothesis() -> None:  # pragma: no cover
        raise AssertionError("This test should be skipped when Hypothesis is missing")
