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
GOLDEN_LOG = Path("tests/fixtures/mcp/core_tools/minimal_golden.jsonl")
WHITELIST = {"ts", "traceId", "spanId", "runId", "attemptId", "logPath"}
EXECUTION_WHITELIST = {"durationMs"}


def _normalise(path: Path) -> list[dict[str, object]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        for field in WHITELIST:
            payload["metadata"].pop(field, None)
            payload.pop(field, None)
        execution = payload.get("execution")
        if isinstance(execution, dict):
            for field in EXECUTION_WHITELIST:
                execution.pop(field, None)
        events.append(payload)
    return sorted(events, key=lambda item: (item["toolId"], item["event"], item["attempt"]))


def test_logs_match_golden(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "7")
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    storage_prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    writer = JsonLogWriter(
        agent_id="mcp_server",
        task_id="06a_core_tools_minimal_subset",
        storage_prefix=storage_prefix,
        latest_symlink=latest,
        schema_version="0.1.0",
        deterministic=True,
    )

    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=writer,
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
    runtime.invoke("mcp.tool:vector.query.search", {"query": "retrieval", "topK": 2})
    runtime.invoke(
        "mcp.tool:docs.load.fetch",
        {
            "path": "tests/fixtures/mcp/docs/sample_article.md",
            "metadataPath": "tests/fixtures/mcp/docs/sample_metadata.json",
        },
    )

    writer.close()
    produced = _normalise(writer.path)
    golden = _normalise(GOLDEN_LOG)
    diff = DeepDiff(golden, produced, ignore_order=True)
    assert diff == {}, diff
