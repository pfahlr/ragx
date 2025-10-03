from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext

SCHEMA_DIR = Path("apps/mcp_server/schemas/mcp")
TOOLPACKS_DIR = Path("apps/mcp_server/toolpacks")
PROMPTS_DIR = Path("apps/mcp_server/prompts")


class _NullLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._counter = 0

    def next_step_id(self) -> int:
        self._counter += 1
        return self._counter

    def emit(self, payload: dict[str, Any]) -> None:
        self.records.append(payload)


@pytest.fixture(autouse=True)
def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "42")


@pytest.fixture
def service(tmp_path: Path) -> McpService:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return McpService.create(
        toolpacks_dir=TOOLPACKS_DIR,
        prompts_dir=PROMPTS_DIR,
        schema_dir=SCHEMA_DIR,
        log_dir=logs_dir,
        logger=_NullLogger(),
        schema_version="0.1.0",
    )


def _context(route: str) -> RequestContext:
    return RequestContext(
        transport="http",
        route=route,
        method={
            "discover": "mcp.discover",
            "prompt": "mcp.prompt.get",
            "tool": "mcp.tool.invoke",
        }[route],
        deterministic_ids=True,
    )

def test_invalid_tool_arguments_produce_invalid_input_error(service: McpService) -> None:
    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={},
        context=context,
    )
    payload = envelope.model_dump(by_alias=True)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_INPUT"
    details = payload["error"]["details"]
    assert details["stage"] == "tool.input"
    assert details["toolId"] == "mcp.tool:docs.load.fetch"
    assert details["schemaPath"], "expected schema path to be populated"
    assert payload["meta"]["status"] == "error"


def test_invalid_tool_output_is_reported(
    service: McpService, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {}

    class _FakeExecutor:
        @staticmethod
        def run_toolpack(toolpack: Any, payload: dict[str, Any]) -> dict[str, Any]:
            called["toolpack"] = toolpack.id
            return {
                "document": {
                    "path": "file:///tmp/invalid.md",
                    "content": "irrelevant",
                    "sha256": "0" * 64,
                }
            }

    monkeypatch.setattr(service, "_executor", _FakeExecutor())

    context = _context("tool")
    envelope = service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": "file:///tmp/valid.md"},
        context=context,
    )
    payload = envelope.model_dump(by_alias=True)
    assert called["toolpack"] == "mcp.tool:docs.load.fetch"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INTERNAL"
    details = payload["error"]["details"]
    assert details["stage"] == "tool.output"
    assert details["toolId"] == "mcp.tool:docs.load.fetch"
    assert details["schemaPath"], "expected schema path to be present"

