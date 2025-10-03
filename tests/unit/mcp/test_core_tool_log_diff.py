# tests/regression/mcp/test_core_tool_log_diff.py
from __future__ import annotations

import json
from pathlib import Path

from deepdiff import DeepDiff

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
GOLDEN_LOG = Path("tests/fixtures/mcp/logs/core_tools_minimal_golden.jsonl")
WHITELIST = {"ts", "trace_id", "span_id", "duration_ms", "run_id", "attempt_id"}


def _normalise_log(path: Path) -> list[dict[str, object]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for field in WHITELIST:
            record.pop(field, None)
        events.append(record)
    return sorted(events, key=lambda evt: (evt["step_id"], evt["attempt"]))


def test_log_diff_against_golden(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    logger = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06a_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)

    runtime.invoke("mcp.tool:exports.render.markdown", {"title": "Demo", "template": "{{ title }}", "body": "x"})
    runtime.invoke(
        "mcp.tool:vector.query.search",
        {"query": "retrieval testing", "top_k": 2},
    )
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": "tests/fixtures/mcp/docs/sample_article.md",
            "metadata_path": "tests/fixtures/mcp/docs/sample_metadata.json",
        },
    )

    produced = _normalise_log(log_path)
    golden = _normalise_log(GOLDEN_LOG)
    diff = DeepDiff(golden, produced, ignore_order=True)
    assert not diff, f"Structured log regression detected: {diff}"
