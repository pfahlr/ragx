from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService
from apps.mcp_server.stdio.server import JsonRpcStdioServer

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


def test_transport_parity_for_tool_invocation(tmp_path: Path) -> None:
    service = McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
    )

    client = TestClient(create_app(service, deterministic_ids=True))
    server = JsonRpcStdioServer(service, deterministic_ids=True)
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")

    http_response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path)}},
    )
    assert http_response.status_code == 200
    http_envelope = http_response.json()
    http_payload = http_envelope["data"]
    http_meta = http_envelope["meta"]

    execution_meta = http_meta.get("execution")
    assert execution_meta is not None, "HTTP transport must include execution metadata"
    assert execution_meta["inputBytes"] > 0
    assert execution_meta["outputBytes"] > 0
    assert execution_meta["durationMs"] >= 0

    idempotency_meta = http_meta.get("idempotency")
    assert idempotency_meta is not None, "HTTP transport must include idempotency metadata"
    assert idempotency_meta["cacheHit"] is False

    stdio_response = asyncio.run(
        server.handle_request({
            "jsonrpc": "2.0",
            "id": "tool-1",
            "method": "mcp.tool.invoke",
            "params": {
                "toolId": "mcp.tool:docs.load.fetch",
                "arguments": {"path": str(fixture_path)},
            },
        })
    )
    stdio_envelope = stdio_response["result"]
    stdio_payload = stdio_envelope["data"]
    stdio_meta = stdio_envelope["meta"]

    assert http_payload == stdio_payload
    assert http_envelope["meta"]["requestId"] == stdio_envelope["meta"]["requestId"]
    assert http_envelope["meta"]["traceId"] == stdio_envelope["meta"]["traceId"]
    assert http_envelope["meta"]["spanId"] == stdio_envelope["meta"]["spanId"]
    assert http_envelope["meta"]["deterministic"] is True
    assert stdio_envelope["meta"]["deterministic"] is True

    stdio_execution = stdio_meta.get("execution")
    assert stdio_execution is not None
    assert stdio_execution["inputBytes"] == execution_meta["inputBytes"]
    assert stdio_execution["outputBytes"] == execution_meta["outputBytes"]
    assert stdio_execution["durationMs"] >= 0

    stdio_idempotency = stdio_meta.get("idempotency")
    assert stdio_idempotency is not None
    assert stdio_idempotency["cacheHit"] is True

    assert stdio_idempotency.get("cacheKey") == idempotency_meta.get("cacheKey")

    parity_keys = {
        "schemaVersion",
        "toolId",
        "promptId",
        "inputBytes",
        "outputBytes",
        "status",
        "attempt",
    }
    for key in parity_keys:
        assert http_meta[key] == stdio_meta[key]
