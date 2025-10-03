"""End-to-end smoke test for the minimal core tools runtime."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent
from apps.mcp_server.runtime.core_tools import CoreToolsRuntime
from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import ToolpackLoader

TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")
LOG_PATH = Path("runs/core_tools/minimal.jsonl")


def _ensure_clean_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def test_runtime_produces_structured_logs(tmp_path: Path) -> None:
    _ensure_clean_log()
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)
    logger = JsonLogWriter(
        LOG_PATH,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
    )
    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=Executor(),
        log_writer=logger,
    )

    result = runtime.invoke(
        "mcp.tool:exports.render.markdown",
        {"title": "Demo", "template": "# {{ title }}", "body": "payload"},
    )
    assert "markdown" in result

    assert LOG_PATH.exists()
    entries = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    assert entries, "Log file must not be empty"

    fields = entries[0]
    for key in [
        "ts",
        "agent_id",
        "task_id",
        "step_id",
        "event",
        "status",
        "trace_id",
        "span_id",
        "tool_id",
        "metadata",
    ]:
        assert key in fields


def test_runtime_records_failure_events(tmp_path: Path) -> None:
    _ensure_clean_log()

    loader = ToolpackLoader()
    loader.load_dir(TOOLPACK_DIR)

    class FailingExecutor(Executor):
        def run_toolpack(self, toolpack, payload):  # type: ignore[override]
            raise RuntimeError("simulated failure")

    logger = JsonLogWriter(
        LOG_PATH,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
    )
    runtime = CoreToolsRuntime(
        toolpacks=loader.list(),
        executor=FailingExecutor(),
        log_writer=logger,
    )

    try:
        runtime.invoke(
            "mcp.tool:exports.render.markdown",
            {"title": "x", "template": "{{ title }}", "body": "payload"},
        )
    except RuntimeError:
        pass

    assert LOG_PATH.exists()
    entries = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    status_values = {entry["status"] for entry in entries}
    assert "error" in status_values


def test_log_writer_flushes_on_shutdown(tmp_path: Path) -> None:
    log_path = tmp_path / "flush.jsonl"
    writer = JsonLogWriter(log_path, agent_id="agent", task_id="task")
    writer.write(
        McpLogEvent(
            ts="2024-01-01T00:00:00Z",
            agent_id="agent",
            task_id="task",
            step_id=1,
            trace_id=str(uuid4()),
            span_id=str(uuid4()),
            tool_id="tool",
            event="test",
            status="success",
            duration_ms=1.0,
            attempt=1,
            input_bytes=0,
            output_bytes=0,
            metadata={},
            error=None,
        )
    )
    writer.flush()
    assert log_path.exists() and log_path.read_text(encoding="utf-8").strip()
