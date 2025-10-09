from __future__ import annotations

from typing import TypedDict

import pytest


class MCPExecutionMeta(TypedDict):
    durationMs: float
    inputBytes: int
    outputBytes: int


class MCPIdempotencyMeta(TypedDict):
    cacheHit: bool


class MCPResponseMeta(TypedDict):
    tool: str
    version: str
    execution: MCPExecutionMeta
    idempotency: MCPIdempotencyMeta
    traceId: str
    warnings: list[str]


class MCPResponse(TypedDict):
    ok: bool
    data: dict[str, str]
    meta: MCPResponseMeta
    errors: list[str]


@pytest.mark.xfail(reason="MCP server not implemented yet", strict=False)
def test_mcp_envelope_schema_contract() -> None:
    # Minimal envelope contract example the server must uphold
    sample: MCPResponse = {
        "ok": True,
        "data": {"echo": "ok"},
        "meta": {
            "tool": "web.search.query",
            "version": "1.0.0",
            "execution": {"durationMs": 1.0, "inputBytes": 12, "outputBytes": 6},
            "idempotency": {"cacheHit": False},
            "traceId": "test-trace",
            "warnings": [],
        },
        "errors": [],
    }
    # In real test: validate against apps/mcp_server/schemas/envelope.schema.json
    assert set(sample) == {"ok", "data", "meta", "errors"}
    assert isinstance(sample["meta"]["execution"]["durationMs"], int | float)
