from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from apps.mcp_server.service.mcp_service import McpService, RequestContext

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
        log_dir=tmp_path / "runs",
        schema_version="0.1.0",
        deterministic_logs=True,
    )


def test_server_logging_creates_latest_symlink(service: McpService) -> None:
    fixture_path = Path("tests/fixtures/mcp/docs/sample_article.md")
    context = RequestContext(transport="http", route="discover", method="mcp.discover")
    service.discover(context)
    tool_context = RequestContext(
        transport="http",
        route="tool",
        method="mcp.tool.invoke",
        deterministic_ids=True,
    )
    service.invoke_tool(
        tool_id="mcp.tool:docs.load.fetch",
        arguments={"path": str(fixture_path)},
        context=tool_context,
    )

    log_root = service.log_manager.latest_symlink.parent
    latest = service.log_manager.latest_symlink
    assert latest.exists()
    records = [
        json.loads(line)
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) >= 2
    first = records[0]
    assert first["transport"] in {"http", "stdio"}
    assert first["route"] in {"discover", "tool", "prompt", "health"}
    assert first["metadata"]["deterministic"] is True

    # Ensure retention policy keeps at most 5 files
    paths = sorted(
        p for p in log_root.iterdir() if p.is_file() and p.name.endswith(".jsonl")
    )
    assert len(paths) <= 5
