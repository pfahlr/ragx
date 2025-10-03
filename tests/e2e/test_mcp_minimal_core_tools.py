"""End-to-end regression for the minimal core tools runtime."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from deepdiff import DeepDiff

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
GOLDEN_LOG = Path("tests/fixtures/mcp/logs/core_tools_minimal_golden.jsonl")
DOC_FIXTURE = Path("tests/fixtures/mcp/docs/sample_article.md")
DOC_META = Path("tests/fixtures/mcp/docs/sample_metadata.json")


def _load_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _normalise(events: list[dict[str, object]]) -> list[dict[str, object]]:
    volatile = {"ts", "duration_ms", "trace_id", "span_id", "run_id", "attempt_id"}
    result: list[dict[str, object]] = []
    for event in events:
        filtered = {key: value for key, value in event.items() if key not in volatile}
        result.append(filtered)
    return sorted(result, key=lambda payload: (payload["step_id"], payload["attempt"]))


@pytest.mark.parametrize(
    "tool_id",
    [
        "mcp.tool:exports.render.markdown",
        "mcp.tool:vector.query.search",
        "mcp.tool:docs.load.fetch",
    ],
)
def test_toolpack_entrypoints_are_present(tool_id: str) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    toolpacks = {tool.id for tool in loader.list()}
    assert tool_id in toolpacks


def test_runtime_matches_golden_log(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    logger = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)

    runtime.invoke("mcp.tool:exports.render.markdown", {"title": "Demo", "template": "{{ title }}", "body": "x"})
    runtime.invoke("mcp.tool:vector.query.search", {"query": "retrieval testing", "top_k": 2})
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {"path": str(DOC_FIXTURE), "metadata_path": str(DOC_META)},
    )

    produced = _normalise(_load_events(log_path))
    golden = _normalise(_load_events(GOLDEN_LOG))
    diff = DeepDiff(golden, produced, ignore_order=True)
    assert not diff, f"Structured log regression detected: {diff}"


def test_diff_script_reports_clean(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    logger = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=Executor(), log_writer=logger)

    runtime.invoke("mcp.tool:exports.render.markdown", {"title": "Demo", "template": "{{ title }}", "body": "x"})
    runtime.invoke("mcp.tool:vector.query.search", {"query": "retrieval testing", "top_k": 2})
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {"path": str(DOC_FIXTURE), "metadata_path": str(DOC_META)},
    )

    cmd = [sys.executable, "scripts/diff_core_tool_logs.py", "--actual", str(log_path), "--golden", str(GOLDEN_LOG)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
