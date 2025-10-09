from __future__ import annotations

import json
from pathlib import Path

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
DOC_FIXTURE_DIR = Path("tests/fixtures/mcp/core_tools/docs")


def test_flow_from_example_fixture(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "21")
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    writer = JsonLogWriter(
        agent_id="mcp-flow",
        task_id="core-tools",
        storage_prefix=storage_prefix,
        latest_symlink=latest,
        schema_version="0.1.0",
        deterministic=True,
    )

    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=writer,
        agent_id="mcp-flow",
        task_id="core-tools",
    )

    document = runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": str(DOC_FIXTURE_DIR / "example.md"),
            "metadataPath": str(DOC_FIXTURE_DIR / "example.json"),
        },
    )

    render = runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {
            "title": document["metadata"]["title"],
            "template": "# {{ title }}\n\n{{ summary }}\n",
            "body": document["metadata"]["summary"],
            "frontMatter": document["metadata"],
        },
    )

    assert "markdown" in render and render["markdown"].startswith("---")
    hits = runtime.invoke(
        "mcp.tool:vector.query.search",
        {"query": document["metadata"]["title"], "topK": 3},
    )
    assert len(hits["hits"]) == 2

    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    tool_ids = {record["toolId"] for record in records}
    assert tool_ids == {
        "mcp.tool:docs.load.fetch",
        "mcp.tool:exports.render.markdown",
        "mcp.tool:vector.query.search",
    }
