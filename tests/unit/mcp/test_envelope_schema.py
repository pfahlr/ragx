from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

pytest.importorskip("pydantic")

from apps.mcp_server.service.envelope import (
    Envelope,
    EnvelopeError,
    EnvelopeMeta,
    ExecutionMeta,
    IdempotencyMeta,
)

SCHEMA_PATH = Path("apps/mcp_server/schemas/mcp/envelope.schema.json")


def _load_schema() -> dict[str, object]:
    if not SCHEMA_PATH.exists():
        pytest.fail(f"Envelope schema missing at {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validator() -> Draft202012Validator:
    schema = _load_schema()
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


def test_envelope_success_matches_schema() -> None:
    envelope = Envelope.success(
        data={"status": "ok"},
        meta=EnvelopeMeta(
            request_id="req-123",
            trace_id="trace-456",
            span_id="span-789",
            schema_version="0.1.0",
            deterministic=True,
            transport="http",
            route="discover",
            method="mcp.discover",
            status="ok",
            attempt=0,
            execution=ExecutionMeta(duration_ms=12.5, input_bytes=0, output_bytes=0),
            idempotency=IdempotencyMeta(cache_hit=False, cache_key=None),
        ),
    )
    payload = envelope.model_dump(by_alias=True)
    _validator().validate(payload)
    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["data"] == {"status": "ok"}
    assert payload["meta"]["execution"]["durationMs"] == pytest.approx(12.5)
    assert payload["meta"]["execution"]["inputBytes"] == 0
    assert payload["meta"]["execution"]["outputBytes"] == 0
    assert payload["meta"]["idempotency"] == {"cacheHit": False, "cacheKey": None}


def test_envelope_error_includes_details() -> None:
    envelope = Envelope.failure(
        error=EnvelopeError(code="MCP_TOOL_ERROR", message="boom"),
        meta=EnvelopeMeta(
            request_id="req-1",
            trace_id="trace-1",
            span_id="span-1",
            schema_version="0.1.0",
            deterministic=False,
            transport="stdio",
            route="tool",
            method="mcp.tool.invoke",
            status="error",
            attempt=0,
            execution=ExecutionMeta(duration_ms=3.4, input_bytes=0, output_bytes=0),
            idempotency=IdempotencyMeta(cache_hit=False, cache_key=None),
        ),
    )
    payload = envelope.model_dump(by_alias=True)
    _validator().validate(payload)
    assert payload["ok"] is False
    assert payload["error"] == {"code": "MCP_TOOL_ERROR", "message": "boom", "details": None}
    assert payload["data"] is None
    assert payload["meta"]["deterministic"] is False
    assert payload["meta"]["execution"]["durationMs"] == pytest.approx(3.4)
    assert payload["meta"]["idempotency"] == {"cacheHit": False, "cacheKey": None}


def test_envelope_serialises_optional_metadata() -> None:
    envelope = Envelope.success(
        data={"value": 1},
        meta=EnvelopeMeta(
            request_id="req-xyz",
            trace_id="trace-xyz",
            span_id="span-xyz",
            schema_version="0.1.0",
            deterministic=True,
            transport="http",
            route="prompt",
            method="mcp.prompt.get",
            status="ok",
            attempt=0,
            execution=ExecutionMeta(duration_ms=1.23, input_bytes=0, output_bytes=0),
            idempotency=IdempotencyMeta(cache_hit=True, cache_key="prompt-cache"),
            prompt_id="core.generic.welcome@1",
        ),
    )
    payload = envelope.model_dump(by_alias=True)
    _validator().validate(payload)
    assert payload["meta"]["promptId"] == "core.generic.welcome@1"
    assert payload["meta"]["toolId"] is None
    assert payload["meta"]["idempotency"] == {"cacheHit": True, "cacheKey": "prompt-cache"}
