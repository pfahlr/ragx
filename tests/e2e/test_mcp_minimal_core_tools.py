from __future__ import annotations

from pathlib import Path

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader
from scripts.diff_core_tool_logs import compare_logs

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
GOLDEN_LOG = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")
WHITELIST = ["ts", "duration_ms", "run_id", "trace_id", "span_id", "attempt_id"]


def test_core_tools_minimal_end_to_end(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    log_writer = JsonLogWriter(
        log_path,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
    )
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=log_writer)

    runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {
            "title": "Demo",
            "template": "# {{ title }}\n{{ body }}",
            "body": "x",
        },
    )
    runtime.invoke("mcp.tool:vector.query.search", {"query": "retrieval testing", "top_k": 2})
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": "tests/fixtures/mcp/core_tools/docs/example.md",
            "metadata_path": "tests/fixtures/mcp/core_tools/docs/example_metadata.json",
        },
    )

    diff = compare_logs(log_path, GOLDEN_LOG, whitelist=WHITELIST)
    assert not diff, f"End-to-end run deviated from golden logs: {diff}"
