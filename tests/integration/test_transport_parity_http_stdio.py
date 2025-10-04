"""Transport parity contract for MCP envelope validation.

The MCP server must apply identical validation rules regardless of whether a
request arrives via HTTP or STDIO JSON-RPC. This test documents the expected
behaviour using fixtures and will fail (``xfail``) until the validation layer is
wired up in a later task.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from apps.mcp_server.service.errors_stub import CanonicalError
from apps.mcp_server.validation.schema_registry_stub import SchemaRegistry

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mcp"
SCHEMAS = Path("codex/specs/schemas")


@pytest.mark.xfail(
    reason="Transport validation parity not implemented",
    raises=NotImplementedError,
    strict=True,
)
def test_http_and_stdio_share_envelope_contract() -> None:
    """Invalid envelopes must be rejected consistently across transports."""

    registry = SchemaRegistry(schema_root=SCHEMAS)
    validator = registry.load_envelope()

    invalid_payload = json.loads(
        (FIXTURES / "envelope" / "invalid_missing_method.json").read_text()
    )

    with pytest.raises(ValidationError):
        validator.validate(invalid_payload)

    # The canonical error code communicated back to both transports is stable.
    error = CanonicalError.to_jsonrpc_error("INVALID_ENVELOPE")
    assert error["message"].startswith("Invalid MCP envelope")


def test_logging_shape_golden_fixture() -> None:
    """The golden log fixture defines the expected structured logging fields."""

    golden_path = FIXTURES / "envelope_validation_golden.jsonl"
    lines = [json.loads(line) for line in golden_path.read_text().splitlines() if line]
    assert len(lines) == 2, "golden must capture http and stdio transports"

    transports = {entry["transport"] for entry in lines}
    assert transports == {"http", "stdio"}

    for entry in lines:
        for field in (
            "ts",
            "requestId",
            "traceId",
            "spanId",
            "transport",
            "route",
            "method",
            "status",
            "durationMs",
            "error",
        ):
            assert field in entry, f"missing {field} in {entry}"

        metadata = entry.get("metadata")
        assert isinstance(metadata, dict)
        assert metadata.get("schemaVersion") == "0.1.0"
        assert metadata.get("deterministic") is True
