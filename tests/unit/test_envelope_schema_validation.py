"""Contract tests for MCP envelope schema validation.

These tests describe the desired behavior for the SchemaRegistry stub
and the canonical envelope JSON Schema defined in the spec. The actual
validation logic will be implemented in a subsequent task; for now we
capture the expectations with xfail markers so they act as executable
specifications.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry import SchemaRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPO_ROOT / "apps" / "mcp_server" / "schemas" / "mcp"
ENVELOPE_SCHEMA_PATH = SCHEMA_ROOT / "envelope.schema.json"
TOOL_IO_SCHEMA_PATH = REPO_ROOT / "codex" / "specs" / "schemas" / "tool_io.schema.json"
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
    assert {"ok", "data", "error", "meta"}.issubset(required)
    meta = schema["properties"]["meta"]
    meta_required = set(meta.get("required", []))
    assert {
        "requestId",
        "traceId",
        "spanId",
        "schemaVersion",
        "deterministic",
        "transport",
        "route",
        "method",
        "status",
        "attempt",
        "execution",
        "idempotency",
    }.issubset(meta_required)


def test_tool_io_schema_declares_required_fields() -> None:
    """The shared tool IO schema must define its required base fields."""
    schema = _load_json(TOOL_IO_SCHEMA_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    required = set(schema.get("required", []))
    assert {"tool", "input"}.issubset(required)


def test_envelope_validator_rejects_missing_meta(schema_registry: SchemaRegistry) -> None:
    """Invalid envelopes missing the meta object should fail validation."""
    invalid_payload = _load_json(FIXTURES_ROOT / "invalid_missing_meta.json")
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(invalid_payload)


def test_envelope_validator_rejects_error_payload_for_success(
    schema_registry: SchemaRegistry,
) -> None:
    """When ok is true the error field must be null."""
    invalid_payload = _load_json(FIXTURES_ROOT / "invalid_success_error.json")
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError):
        validator.validate(invalid_payload)


def test_envelope_validator_accepts_valid_response_envelope(
    schema_registry: SchemaRegistry,
) -> None:
    """Response envelopes produced by the service schema should validate."""

    validator = schema_registry.load_envelope()
    valid_payload = {
        "ok": True,
        "data": {"result": "ok"},
        "error": None,
        "meta": {
            "requestId": "req-1",
            "traceId": "trace-1",
            "spanId": "span-1",
            "schemaVersion": "0.1.0",
            "deterministic": False,
            "transport": "http",
            "route": "discover",
            "method": "mcp.discover",
            "status": "ok",
            "attempt": 0,
            "execution": {
                "durationMs": 1.0,
                "inputBytes": 128,
                "outputBytes": 256,
            },
            "idempotency": {"cacheHit": False},
            "toolId": None,
            "promptId": None,
        },
    }

    # Should not raise if the registry is wired to the response envelope schema.
    validator.validate(valid_payload)


def test_tool_io_registry_returns_named_validators(schema_registry: SchemaRegistry) -> None:
    """Per-tool validator objects must expose validate() for input/output payloads."""
    validators = schema_registry.load_tool_io("mcp.tool:web.search.query")
    assert hasattr(validators.input, "validate"), "input validator must expose validate()"
    assert hasattr(validators.output, "validate"), "output validator must expose validate()"
    # Once implemented these validations should raise when required fields are missing.
    with pytest.raises(ValidationError):
        validators.input.validate({})
    with pytest.raises(ValidationError):
        validators.output.validate({})


def test_registry_reuses_compiled_validators(schema_registry: SchemaRegistry) -> None:
    """Calling ``load_envelope`` repeatedly should reuse cached validators."""
    first = schema_registry.load_envelope()
    second = schema_registry.load_envelope()
    assert first is second


def test_tool_io_registry_reuses_cached_validators(schema_registry: SchemaRegistry) -> None:
    """Per-tool validators should reuse cached compiled schema objects."""
    first = schema_registry.load_tool_io("mcp.tool:exports.render.markdown")
    second = schema_registry.load_tool_io("mcp.tool:exports.render.markdown")
    assert first.input is second.input
    assert first.output is second.output


def test_registry_indexes_all_tool_schemas(schema_registry: SchemaRegistry) -> None:
    """Every tool schema in the directory should be discoverable via the registry."""
    schema_root = Path("apps/mcp_server/schemas/tools")
    for path in sorted(schema_root.glob("*.input.schema.json")):
        stem = path.name[: -len(".input.schema.json")]
        tool_id = f"mcp.tool:{stem.replace('_', '.')}"
        validators = schema_registry.load_tool_io(tool_id)
        with pytest.raises(ValidationError):
            validators.input.validate({})
        with pytest.raises(ValidationError):
            validators.output.validate({})


def test_registry_raises_for_unknown_tool(schema_registry: SchemaRegistry) -> None:
    """Unknown tools should raise a clear error to surface missing schemas."""
    with pytest.raises(KeyError):
        schema_registry.load_tool_io("mcp.tool:unknown.tool")
