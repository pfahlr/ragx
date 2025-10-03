# tests/unit/mcp/test_core_tool_logging.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent

REQUIRED_KEYS = {
    "ts",
    "agent_id",
    "task_id",
    "step_id",
    "trace_id",
    "span_id",
    "tool_id",
    "event",
    "status",
    "attempt",
    "duration_ms",
    "input_bytes",
    "output_bytes",
    "metadata",
    "run_id",
    "attempt_id",
}

def _base_event(event: str, status: str, attempt: int) -> McpLogEvent:
    return McpLogEvent(
        ts=datetime.now(timezone.utc),
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
        step_id=attempt,
        trace_id=str(uuid4()),
        span_id=str(uuid4()),
        tool_id="mcp.tool:exports.render.markdown",
        event=event,
        status=status,
        duration_ms=12.5,
        attempt=attempt,
        input_bytes=42,
        output_bytes=128,
        metadata={"schema_version": "0.1.0", "deterministic": True},
        error=None,
    )

def test_json_log_writer_persists_success_event(tmp_path: Path) -> None:
    log_path = tmp_path / "core-tools.jsonl"
    writer = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")

    event = _base_event(event="invocation_success", status="success", attempt=1)
    writer.write(event)

    content = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    payload = json.loads(content[0])
    assert REQUIRED_KEYS.issubset(payload.keys())
    assert payload["event"] == "invocation_success"
    assert payload["status"] == "success"
    assert payload["metadata"]["deterministic"] is True


def test_json_log_writer_includes_error_payload(tmp_path: Path) -> None:
    log_path = tmp_path / "core-tools.jsonl"
    writer = JsonLogWriter(log_path, agent_id="mcp_server", task_id="06ab_core_tools_minimal_subset")

    failure = _base_event(event="invocation_failure", status="error", attempt=2)
    failure.error = {"code": "SIMULATED", "message": "forced failure"}
    writer.write(failure)

    retry = _base_event(event="invocation_retry", status="error", attempt=3)
    retry.error = {"code": "RETRY", "message": "retryable"}
    writer.write(retry)

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["error"] == {"code": "SIMULATED", "message": "forced failure"}
    assert lines[1]["event"] == "invocation_retry"
    assert lines[1]["attempt"] == 3
    assert lines[1]["error"]["code"] == "RETRY"



