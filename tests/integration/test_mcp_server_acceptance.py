
from typing import TypedDict

import pytest


class ExecutionMeta(TypedDict):
    durationMs: int
    inputBytes: int
    outputBytes: int


class IdempotencyMeta(TypedDict):
    cacheHit: bool


class MCPResponseMeta(TypedDict):
    tool: str
    version: str
    execution: ExecutionMeta
    idempotency: IdempotencyMeta
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
            "execution": {"durationMs": 1, "inputBytes": 0, "outputBytes": 2},
            "idempotency": {"cacheHit": False},
            "traceId": "test-trace",
            "warnings": [],
        },
        "errors": [],
    }
    # In real test: validate against apps/mcp_server/schemas/envelope.schema.json
    assert set(sample) == {"ok", "data", "meta", "errors"}
    assert isinstance(sample["meta"]["execution"]["durationMs"], int)
