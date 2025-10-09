from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


VOLATILE_META_FIELDS = {"requestId", "traceId", "spanId"}


def _stable_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    stable = {key: value for key, value in meta.items() if key not in VOLATILE_META_FIELDS}
    stable.pop("transport", None)
    execution = dict(stable.get("execution", {}))
    execution.pop("durationMs", None)
    stable["execution"] = execution
    return stable


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

    assert http_payload == stdio_payload
    assert http_envelope["meta"]["deterministic"] is True
    assert stdio_envelope["meta"]["deterministic"] is True
    assert _stable_meta(http_envelope["meta"]) == _stable_meta(stdio_envelope["meta"])
