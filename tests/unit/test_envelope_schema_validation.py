"""Executable spec for MCP envelope and per-tool schema validation.

These tests define the expected behaviour for the SchemaRegistry contract
introduced in ``06cV2A``. They are marked as ``xfail`` until the validator
implementation lands in part B of the task sequence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mcp" / "envelope"
SCHEMAS = Path("codex/specs/schemas")


@pytest.fixture
def registry() -> SchemaRegistry:
    """Return a SchemaRegistry instance rooted at the canonical schema path."""

    return SchemaRegistry(schema_root=SCHEMAS)


@pytest.mark.xfail(
    reason="Envelope schema validation not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_envelope_missing_method_is_rejected(registry: SchemaRegistry) -> None:
    """Every envelope must include a ``method`` field.

    ``invalid_missing_method.json`` purposefully omits the ``method`` key. Once the
    validator is implemented this call should raise ``ValidationError``.
    """

    validator = registry.load_envelope()
    payload = json.loads((FIXTURES / "invalid_missing_method.json").read_text())
    with pytest.raises(ValidationError):
        validator.validate(payload)


@pytest.mark.xfail(
    reason="Envelope schema validation not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_envelope_params_must_be_object(registry: SchemaRegistry) -> None:
    """``params`` must always be a JSON object per the MCP spec."""

    validator = registry.load_envelope()
    payload = json.loads((FIXTURES / "invalid_params_type.json").read_text())
    with pytest.raises(ValidationError):
        validator.validate(payload)


@pytest.mark.xfail(
    reason="Tool I/O schema validation not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_tool_io_payload_enforces_tool_identity(registry: SchemaRegistry) -> None:
    """Tool payloads are validated against per-tool schemas.

    The base schema requires that the ``tool`` field equals the tool identifier that
    was requested. A mismatch should produce a schema violation.
    """

    validators = registry.load_tool_io("mcp.tool:docs.load.fetch")
    invalid_payload = {"tool": "mcp.tool:other", "input": {}}
    with pytest.raises(ValidationError):
        validators.input.validate(invalid_payload)


@pytest.mark.xfail(
    reason="Tool I/O schema validation not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_tool_output_shape_is_enforced(registry: SchemaRegistry) -> None:
    """The ``output`` validator must enforce the response envelope shape."""

    validators = registry.load_tool_io("mcp.tool:docs.load.fetch")
    invalid_output = {"tool": "mcp.tool:docs.load.fetch", "output": "not-an-object"}
    with pytest.raises(ValidationError):
        validators.output.validate(invalid_output)
