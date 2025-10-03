from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService
from apps.mcp_server.stdio.server import JsonRpcStdioServer

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


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
def server(service: McpService) -> JsonRpcStdioServer:
    return JsonRpcStdioServer(service)


def _request(method: str, *, params: dict[str, object] | None = None) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": "req-1", "method": method, "params": params or {}}


def test_stdio_discover(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(server.handle_request(_request("mcp.discover")))
    assert response["id"] == "req-1"
    assert response["result"]["ok"] is True
    assert response["result"]["data"]["prompts"]


def test_stdio_tool_invoke(server: JsonRpcStdioServer) -> None:
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    response = asyncio.run(
        server.handle_request(
            _request(
                "mcp.tool.invoke",
                params={
                    "toolId": "mcp.tool:docs.load.fetch",
                    "arguments": {"path": str(fixture_path)},
                },
            )
        )
    )
    assert response["result"]["ok"] is True
    assert response["result"]["data"]["result"]["document"]["path"].endswith("sample_article.md")


def test_stdio_cancel_notification(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(
        server.handle_notification(
            {"jsonrpc": "2.0", "method": "$/cancel", "params": {"id": "req-1"}}
        )
    )
    assert response is None
