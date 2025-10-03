from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

pytest.importorskip("pydantic")

from apps.mcp_server.service.envelope import Envelope, EnvelopeError, EnvelopeMeta

SCHEMA_PATH = Path("apps/mcp_server/schemas/mcp/envelope.schema.json")


@pytest.fixture(scope="module")
def envelope_validator() -> Draft202012Validator:
    if not SCHEMA_PATH.exists():
        pytest.fail(f"Envelope schema missing: {SCHEMA_PATH}")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


def _meta_payload() -> EnvelopeMeta:
    return EnvelopeMeta(
        request_id="req-1234",
        trace_id="trace-1234",
        span_id="span-1234",
        schema_version="0.1.0",
        deterministic=True,
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        duration_ms=12.5,
        status="ok",
        attempt=1,
        input_bytes=128,
        output_bytes=256,
        tool_id="mcp.tool:example",
        prompt_id=None,
    )


def test_success_envelope_matches_schema(envelope_validator: Draft202012Validator) -> None:
    envelope = Envelope.success(data={"result": {}}, meta=_meta_payload())
    envelope_validator.validate(envelope.model_dump(by_alias=True))


def test_error_envelope_matches_schema(envelope_validator: Draft202012Validator) -> None:
    meta = _meta_payload()
    error = EnvelopeError(code="INVALID_INPUT", message="invalid request")
    error_envelope = Envelope.failure(error=error, meta=meta.model_copy(update={"status": "error"}))
    envelope_validator.validate(error_envelope.model_dump(by_alias=True))


def test_schema_rejects_missing_fields(envelope_validator: Draft202012Validator) -> None:
    with pytest.raises(ValidationError):
        envelope_validator.validate({"ok": True, "data": {}})
