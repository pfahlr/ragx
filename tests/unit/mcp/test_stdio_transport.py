from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext
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
    assert response["result"]["meta"]["transport"] == "stdio"


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
    assert response["result"]["meta"]["execution"]["inputBytes"] > 0
    assert response["result"]["meta"]["execution"]["outputBytes"] > 0
    assert response["result"]["meta"]["idempotency"]["cacheHit"] is False


def test_stdio_prompt_get_accepts_string_major(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(
        server.handle_request(
            _request(
                "mcp.prompt.get",
                params={"promptId": "core.generic.bootstrap", "major": "1"},
            )
        )
    )
    assert response["result"]["ok"] is True
    assert response["result"]["data"]["id"] == "core.generic.bootstrap@1"


def test_stdio_cancel_notification(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(
        server.handle_notification(
            {"jsonrpc": "2.0", "method": "$/cancel", "params": {"id": "req-1"}}
        )
    )
    assert response is None


def test_stdio_rejects_positional_params(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(
        server.handle_request(
            {"jsonrpc": "2.0", "id": "req-1", "method": "mcp.tool.invoke", "params": ["foo"]}
        )
    )
    assert response["error"]["code"] == -32602
    assert response["id"] == "req-1"


def test_stdio_prompt_get_rejects_non_integer_major(server: JsonRpcStdioServer) -> None:
    response = asyncio.run(
        server.handle_request(
            _request(
                "mcp.prompt.get",
                params={"promptId": "core.generic.bootstrap", "major": "foo"},
            )
        )
    )
    assert response["error"]["code"] == -32602
    assert response["id"] == "req-1"


def test_stdio_invocation_preserves_http_executor_cache(
    service: McpService, server: JsonRpcStdioServer
) -> None:
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    http_context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
    )

    first = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=http_context,
    )
    first_payload = first.model_dump(by_alias=True)
    assert first_payload["meta"]["idempotency"]["cacheHit"] is False

    stdio_response = asyncio.run(
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
    assert stdio_response["result"]["meta"]["idempotency"]["cacheHit"] is False

    second = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=http_context,
    )
    second_payload = second.model_dump(by_alias=True)
    assert second_payload["meta"]["idempotency"]["cacheHit"] is True
