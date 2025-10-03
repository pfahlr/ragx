from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from apps.mcp_server.service.mcp_service import McpService
from apps.mcp_server.stdio.server import JsonRpcStdioServer

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


@pytest.fixture
def server(tmp_path: Path) -> JsonRpcStdioServer:
    service = McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
    )
    return JsonRpcStdioServer(service)


def _validator(name: str) -> Draft202012Validator:
    schema_path = SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    return validator


def test_stdio_end_to_end(server: JsonRpcStdioServer) -> None:
    discover = asyncio.run(server.handle_request({"jsonrpc": "2.0", "id": "d1", "method": "mcp.discover"}))
    _validator("discover.response.schema.json").validate(discover["result"]["data"])

    prompt = asyncio.run(
        server.handle_request({
            "jsonrpc": "2.0",
            "id": "p1",
            "method": "mcp.prompt.get",
            "params": {"promptId": "core.generic.bootstrap@1"},
        })
    )
    _validator("prompt.response.schema.json").validate(prompt["result"]["data"])

    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    tool = asyncio.run(
        server.handle_request({
            "jsonrpc": "2.0",
            "id": "t1",
            "method": "mcp.tool.invoke",
            "params": {
                "toolId": "mcp.tool:docs.load.fetch",
                "arguments": {"path": str(fixture_path)},
            },
        })
    )
    _validator("tool.response.schema.json").validate(tool["result"]["data"])
