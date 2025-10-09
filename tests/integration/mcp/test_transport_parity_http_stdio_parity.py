from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient

from apps.mcp_server.http.main import create_app
from apps.mcp_server.service.mcp_service import McpService, RequestContext
from apps.mcp_server.stdio.server import JsonRpcStdioServer

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
PROMPTS_DIR = Path("apps/mcp_server/prompts")
TOOLPACK_STUB = Path("tests/stubs/toolpacks/deterministic_sum.tool.yaml")
TOOL_ID = "tests.deterministic.sum"


def _canonical_size(payload: dict[str, object]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    shutil.copy(TOOLPACK_STUB, toolpacks_dir / "tests.deterministic.sum.tool.yaml")
    return McpService.create(
        toolpacks_dir=toolpacks_dir,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
        max_input_bytes=256,
        max_output_bytes=128,
        timeout_ms=500,
    )


@pytest.fixture
def transports(service: McpService) -> tuple[TestClient, JsonRpcStdioServer]:
    client = TestClient(create_app(service, deterministic_ids=True))
    stdio = JsonRpcStdioServer(service, deterministic_ids=True)
    return client, stdio


def test_http_stdio_parity_and_cache_hit(
    service: McpService, transports: tuple[TestClient, JsonRpcStdioServer]
) -> None:
    client, stdio = transports
    payload = {"values": [1, 2, 3], "text": "abc"}

    http_response = client.post(f"/mcp/tool/{TOOL_ID}", json={"arguments": payload})
    assert http_response.status_code == 200
    http_envelope = http_response.json()

    assert http_envelope["ok"] is True
    assert http_envelope["meta"]["idempotency"]["cacheHit"] is False
    execution_meta = http_envelope["meta"]["execution"]
    assert execution_meta["inputBytes"] == _canonical_size(payload)
    expected_output = {"sum": 6.0, "text": "abc"}
    assert execution_meta["outputBytes"] == _canonical_size(expected_output)
    assert execution_meta["durationMs"] >= 0.0

    stdio_response = asyncio.run(
        stdio.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "call-1",
                "method": "mcp.tool.invoke",
                "params": {"toolId": TOOL_ID, "arguments": payload},
            }
        )
    )

    assert "result" in stdio_response
    stdio_envelope = stdio_response["result"]
    assert stdio_envelope["meta"]["idempotency"]["cacheHit"] is True
    stdio_exec = stdio_envelope["meta"]["execution"]
    assert stdio_exec["inputBytes"] == execution_meta["inputBytes"]
    assert stdio_exec["outputBytes"] == execution_meta["outputBytes"]
    assert stdio_envelope["data"] == http_envelope["data"]


def test_service_enforces_limits_and_timeout(service: McpService) -> None:
    # Input larger than limit should be rejected before execution.
    oversized_text = "x" * 300
    context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=True,
    )
    envelope = service.invoke_tool(
        tool_id=TOOL_ID,
        arguments={"values": [1], "text": oversized_text},
        context=context,
    )
    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "INVALID_INPUT"
    meta = envelope.meta.model_dump(by_alias=True)
    assert meta["execution"]["inputBytes"] == _canonical_size(
        {"values": [1], "text": oversized_text}
    )
    assert meta["execution"]["outputBytes"] == 0

    # Output larger than limit should produce INVALID_OUTPUT.
    context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=True,
    )
    envelope = service.invoke_tool(
        tool_id=TOOL_ID,
        arguments={"values": [1, 2], "outputSize": 400},
        context=context,
    )
    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "INVALID_OUTPUT"
    meta = envelope.meta.model_dump(by_alias=True)
    assert meta["execution"]["outputBytes"] >= 400

    # Timeout scenario using delay greater than the configured timeout.
    context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=True,
    )
    envelope = service.invoke_tool(
        tool_id=TOOL_ID,
        arguments={"values": [1, 2], "delayMs": 600},
        context=context,
    )
    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.code == "TIMEOUT"
    meta = envelope.meta.model_dump(by_alias=True)
    assert meta["execution"]["durationMs"] >= service.timeout_ms
