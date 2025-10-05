"""Property-style fuzz tests for MCP envelope validation."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry


@given(st.text())
def test_envelope_validator_rejects_missing_method(random_id: str) -> None:
    """Payloads without a method must be rejected regardless of ID shape."""

    payload: dict[str, Any] = {
        "id": random_id,
        "jsonrpc": "2.0",
        "params": {},
    }
    validator = SchemaRegistry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)


@given(st.text())
def test_envelope_validator_rejects_non_object_params(random_value: str) -> None:
    """Params must be an object; scalar/text payloads should fail validation."""

    payload: dict[str, Any] = {
        "id": "req-fixed",
        "jsonrpc": "2.0",
        "method": "mcp.tool.invoke",
        "params": random_value,
    }
    validator = SchemaRegistry().load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(payload)
