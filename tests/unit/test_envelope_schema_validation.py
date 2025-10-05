"""Contract tests for MCP envelope schema validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "codex" / "specs" / "schemas"
ENVELOPE_SCHEMA_PATH = SCHEMA_ROOT / "envelope.schema.json"
TOOL_IO_SCHEMA_PATH = SCHEMA_ROOT / "tool_io.schema.json"
FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "mcp" / "envelope"


@pytest.fixture(scope="module")
def schema_registry() -> SchemaRegistry:
    return SchemaRegistry()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_envelope_schema_declares_required_fields() -> None:
    """The envelope schema must include mandatory fields specified by the spec."""
    schema = _load_json(ENVELOPE_SCHEMA_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    required = set(schema.get("required", []))
    assert {"id", "jsonrpc", "method", "params"}.issubset(required)


def test_tool_io_schema_declares_required_fields() -> None:
    """The shared tool IO schema must define its required base fields."""
    schema = _load_json(TOOL_IO_SCHEMA_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    required = set(schema.get("required", []))
    assert {"tool", "input"}.issubset(required)


def test_envelope_validator_rejects_missing_method(schema_registry: SchemaRegistry) -> None:
    """Invalid envelopes missing the method field should fail validation."""
    invalid_payload = _load_json(FIXTURES_ROOT / "invalid_missing_method.json")
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(invalid_payload)


def test_envelope_validator_rejects_invalid_params_type(schema_registry: SchemaRegistry) -> None:
    """The params field must be an object according to the schema contract."""
    invalid_payload = _load_json(FIXTURES_ROOT / "invalid_params_type.json")
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(invalid_payload)


def test_tool_io_registry_returns_named_validators(schema_registry: SchemaRegistry) -> None:
    """Per-tool validator objects must expose validate() for input/output payloads."""
    validators = schema_registry.load_tool_io("mcp.tool:web.search.query")
    assert hasattr(validators.input, "validate"), "input validator must expose validate()"
    assert hasattr(validators.output, "validate"), "output validator must expose validate()"
    # Once implemented these validations should raise when required fields are missing.
    with pytest.raises(ValidationError):
        validators.input.validate({})


def test_envelope_validator_is_cached(schema_registry: SchemaRegistry) -> None:
    """Repeated calls should return the same compiled validator instance."""
    first = schema_registry.load_envelope()
    second = schema_registry.load_envelope()
    assert first is second


def test_tool_io_validators_cache_per_tool(schema_registry: SchemaRegistry) -> None:
    """Validators for the same tool should be memoised."""
    first = schema_registry.load_tool_io("mcp.tool:web.search.query")
    second = schema_registry.load_tool_io("mcp.tool:web.search.query")
    assert first.input is second.input
    assert first.output is second.output
