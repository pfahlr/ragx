from __future__ import annotations

import json
import subprocess
from pathlib import Path

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
RUNS_PATH = Path("runs/core_tools/minimal.jsonl")
GOLDEN = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")
DIFF_SCRIPT = Path("scripts/diff_core_tool_logs.py")


def test_core_tools_minimal_end_to_end(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RUNS_PATH.exists():
        RUNS_PATH.unlink()

    logger = JsonLogWriter(RUNS_PATH, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)

    runtime.invoke("mcp.tool:exports.render.markdown", {"title": "Demo", "template": "{{ title }}", "body": "x"})
    runtime.invoke("mcp.tool:vector.query.search", {"query": "retrieval testing", "top_k": 2})
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": "tests/fixtures/mcp/core_tools/docs/example.md",
            "metadata_path": "tests/fixtures/mcp/core_tools/docs/example.json",
        },
    )
    logger.close()

    assert RUNS_PATH.exists()

    with RUNS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            for key in ("ts", "agent_id", "task_id", "step_id"):
                assert key in payload

    result = subprocess.run(
        ["python", str(DIFF_SCRIPT), "--actual", str(RUNS_PATH), "--expected", str(GOLDEN)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
