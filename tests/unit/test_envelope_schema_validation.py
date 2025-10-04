"""Executable specification for MCP envelope and tool I/O schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry_stub import (
    SchemaRegistry,
    ToolIOValidators,
)

SCHEMA_BASE = Path("codex/specs/schemas")
VALID_TOOL_ID = "mcp.tool:docs.load.fetch"


@pytest.fixture()
def schema_registry(tmp_path: Path) -> SchemaRegistry:
    """Return a SchemaRegistry bound to the canonical schema directory."""

    del tmp_path
    return SchemaRegistry(schema_dir=SCHEMA_BASE)


@pytest.fixture()
def valid_envelope_payload() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": "request-123",
        "method": "mcp.tool.invoke",
        "params": {
            "tool": VALID_TOOL_ID,
            "input": {"path": "tests/fixtures/mcp/docs/sample_article.md"},
        },
    }


@pytest.fixture()
def invalid_missing_method_payload() -> dict[str, Any]:
    path = Path("tests/fixtures/mcp/envelope/invalid_missing_method.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture()
def invalid_params_type_payload() -> dict[str, Any]:
    path = Path("tests/fixtures/mcp/envelope/invalid_params_type.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
def test_envelope_validator_accepts_valid_payload(
    schema_registry: SchemaRegistry,
    valid_envelope_payload: dict[str, Any],
) -> None:
    validator = schema_registry.load_envelope()
    validator.validate(valid_envelope_payload)


@pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
def test_envelope_validator_rejects_missing_method(
    schema_registry: SchemaRegistry,
    invalid_missing_method_payload: dict[str, Any],
) -> None:
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError) as exc:
        validator.validate(invalid_missing_method_payload)
    assert "method" in str(exc.value).lower()


@pytest.mark.xfail(strict=True, reason="Envelope schema validation not implemented yet")
def test_envelope_validator_rejects_non_object_params(
    schema_registry: SchemaRegistry,
    invalid_params_type_payload: dict[str, Any],
) -> None:
    validator = schema_registry.load_envelope()
    with pytest.raises(ValidationError) as exc:
        validator.validate(invalid_params_type_payload)
    assert "params" in str(exc.value).lower()
    assert "object" in str(exc.value).lower()


@pytest.mark.xfail(strict=True, reason="Tool IO schema validation not implemented yet")
def test_tool_io_validators_cover_input_and_output_contracts(
    schema_registry: SchemaRegistry,
) -> None:
    validators = schema_registry.load_tool_io(VALID_TOOL_ID)
    assert isinstance(validators, ToolIOValidators)

    input_payload = {
        "tool": VALID_TOOL_ID,
        "input": {"path": "tests/fixtures/mcp/docs/sample_article.md"},
    }
    output_payload = {
        "tool": VALID_TOOL_ID,
        "output": {
            "content": [
                {"type": "text", "text": "Sample output"},
            ],
        },
    }

    validators.input_validator.validate(input_payload)
    validators.output_validator.validate(output_payload)


@pytest.mark.xfail(strict=True, reason="Tool IO schema validation not implemented yet")
def test_tool_io_validators_flag_invalid_payloads(
    schema_registry: SchemaRegistry,
) -> None:
    validators = schema_registry.load_tool_io(VALID_TOOL_ID)

    missing_input = {"tool": VALID_TOOL_ID}
    with pytest.raises(ValidationError) as input_exc:
        validators.input_validator.validate(missing_input)
    assert "input" in str(input_exc.value).lower()

    wrong_type_output = {"tool": VALID_TOOL_ID, "output": "raw"}
    with pytest.raises(ValidationError) as output_exc:
        validators.output_validator.validate(wrong_type_output)
    assert "object" in str(output_exc.value).lower()


@pytest.mark.xfail(strict=True, reason="Schema caching not implemented yet")
def test_schema_registry_caches_validators(schema_registry: SchemaRegistry) -> None:
    first = schema_registry.load_tool_io(VALID_TOOL_ID)
    second = schema_registry.load_tool_io(VALID_TOOL_ID)
    assert first is second
    assert first.input_validator is second.input_validator
    assert first.output_validator is second.output_validator


@pytest.mark.xfail(strict=True, reason="Envelope fingerprint caching not implemented yet")
def test_envelope_validator_cached_instance(schema_registry: SchemaRegistry) -> None:
    first = schema_registry.load_envelope()
    second = schema_registry.load_envelope()
    assert first is second
