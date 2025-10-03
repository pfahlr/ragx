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
    monkeypatch.setenv("RAGX_SEED", "40")


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "logs",
        schema_version="0.1.0",
    )


@pytest.fixture
def client(service: McpService) -> TestClient:
    app = create_app(service)
    return TestClient(app)


def test_invalid_tool_payload_returns_invalid_input_across_transports(
    client: TestClient, service: McpService
) -> None:
    response = client.post(
        "/mcp/tool/mcp.tool:docs.load.fetch",
        json={"arguments": {"encoding": "utf-8"}},
    )
    assert response.status_code == 400
    http_payload = response.json()
    assert http_payload["ok"] is False
    assert http_payload["error"]["code"] == "INVALID_INPUT"
    assert http_payload["error"]["details"]["stage"] == "input"

    server = JsonRpcStdioServer(service, deterministic_ids=True)
    stdio_payload = asyncio.run(
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "mcp.tool.invoke",
                "params": {
                    "toolId": "mcp.tool:docs.load.fetch",
                    "arguments": {"encoding": "utf-8"},
                },
            }
        )
    )
    assert stdio_payload["result"]["ok"] is False
    assert stdio_payload["result"]["error"]["code"] == "INVALID_INPUT"
    assert stdio_payload["result"]["error"]["details"]["stage"] == "input"
