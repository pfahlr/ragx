"""Runtime logging behaviour for the minimal core tools runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.mcp_server.logging import JsonLogWriter
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")


class FailThenSucceedExecutor(Executor):
    """Executor that fails on the first attempt then delegates to super."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def run_toolpack(self, toolpack, payload):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            raise ToolpackExecutionError("planned failure")
        return super().run_toolpack(toolpack, payload)


class AlwaysFailExecutor(Executor):
    """Executor that raises for every invocation."""

    def run_toolpack(self, toolpack, payload):  # type: ignore[override]
        raise ToolpackExecutionError("fatal failure")


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _runtime(tmp_path: Path, executor: Executor) -> tuple[CoreToolsRuntime, Path]:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    log_path = tmp_path / "core-tools.jsonl"
    log_writer = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")
    runtime = CoreToolsRuntime(toolpacks=loader.list(), executor=executor, log_writer=log_writer)
    return runtime, log_path


def test_runtime_logs_start_and_success(tmp_path: Path) -> None:
    runtime, log_path = _runtime(tmp_path, Executor())
    payload = {"title": "Demo", "template": "# {{ title }}", "body": "example"}
    result = runtime.invoke("mcp.tool:exports.render.markdown", payload)

    assert "markdown" in result

    events = _read_events(log_path)
    assert len(events) >= 2
    start, success = events[-2:]
    assert start["event"] == "invocation_start"
    assert start["status"] == "running"
    assert success["event"] == "invocation_success"
    assert success["status"] == "success"
    assert success["attempt"] == 1


def test_runtime_logs_retry_then_success(tmp_path: Path) -> None:
    runtime, log_path = _runtime(tmp_path, FailThenSucceedExecutor())
    payload = {"query": "retrieval", "top_k": 1}
    result = runtime.invoke("mcp.tool:vector.query.search", payload, max_attempts=2)

    assert result["hits"]

    events = _read_events(log_path)
    retry_event = next(evt for evt in events if evt["event"] == "invocation_retry")
    success_event = events[-1]
    assert retry_event["attempt"] == 1
    assert retry_event["status"] == "error"
    assert retry_event["error"]["code"] == "ToolpackExecutionError"
    assert success_event["attempt"] == 2
    assert success_event["status"] == "success"


def test_runtime_logs_failure_after_max_attempts(tmp_path: Path) -> None:
    runtime, log_path = _runtime(tmp_path, AlwaysFailExecutor())
    payload = {"path": "tests/fixtures/mcp/docs/sample_article.md"}

    with pytest.raises(ToolpackExecutionError):
        runtime.invoke("mcp.tool:docs.load.fetch", payload, max_attempts=2)

    events = _read_events(log_path)
    failure_event = events[-1]
    assert failure_event["event"] == "invocation_failure"
    assert failure_event["status"] == "error"
    assert failure_event["attempt"] == 2
    assert failure_event["error"]["code"] == "ToolpackExecutionError"
