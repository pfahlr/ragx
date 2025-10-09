# tests/regression/mcp/test_core_tool_log_diff.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

deepdiff = pytest.importorskip("deepdiff")
DeepDiff = deepdiff.DeepDiff

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
GOLDEN_LOG = Path("tests/fixtures/mcp/logs/core_tools_minimal_golden.jsonl")
WHITELIST = {
    "ts",
    "traceId",
    "spanId",
    "runId",
    "attemptId",
    "logPath",
    "execution.durationMs",
    "metadata.execution.durationMs",
    "metadata.runId",
    "metadata.attemptId",
    "metadata.logPath",
    "schemaVersion",
}


def _pop_nested(record: dict[str, object], path: str) -> None:
    parts = path.split(".")
    target: object = record
    for key in parts[:-1]:
        if not isinstance(target, dict):
            return
        target = target.get(key)
        if target is None:
            return
    if isinstance(target, dict):
        target.pop(parts[-1], None)


def _normalise_log(path: Path) -> list[dict[str, object]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for field in WHITELIST:
            _pop_nested(record, field)
        events.append(record)
    return sorted(events, key=lambda evt: (evt["stepId"], evt["attempt"]))


def test_log_diff_against_golden(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest_symlink = tmp_path / "runs/core_tools/minimal.jsonl"
    logger = JsonLogWriter(
        agent_id="mcp_server",
        task_id="06a_core_tools_minimal_subset",
        storage_prefix=storage_prefix,
        latest_symlink=latest_symlink,
        schema_version="0.1.0",
        deterministic=True,
    )
    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=logger,
        agent_id="mcp_server",
        task_id="06a_core_tools_minimal_subset",
    )

    runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {
            "title": "Demo",
            "template": "# {{ title }}",
            "body": "example",
        },
    )
    runtime.invoke(
        "mcp.tool:vector.query.search",
        {"query": "retrieval", "topK": 2},
    )
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": "tests/fixtures/mcp/docs/sample_article.md",
            "metadataPath": "tests/fixtures/mcp/docs/sample_metadata.json",
        },
    )

    logger.close()
    produced = _normalise_log(logger.path)
    golden = _normalise_log(GOLDEN_LOG)
    diff = DeepDiff(golden, produced, ignore_order=True)
    assert not diff, f"Structured log regression detected: {diff}"
