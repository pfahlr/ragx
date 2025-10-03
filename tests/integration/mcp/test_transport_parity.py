from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
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

    client = TestClient(create_app(service))
    server = JsonRpcStdioServer(service)
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")

    http_response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"path": str(fixture_path)}},
    )
    assert http_response.status_code == 200
    http_payload = http_response.json()["data"]

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
    stdio_payload = stdio_response["result"]["data"]

    assert http_payload == stdio_payload
