from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent


def _event(event: str, status: str, attempt: int) -> McpLogEvent:
    return McpLogEvent(
        ts=datetime.now(UTC),
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
        run_id="runtime-demo",
    )


def test_core_tools_minimal_logging_success(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    writer = JsonLogWriter(
        log_path,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
        retention="keep-last-5",
    )

    writer.write(_event("invocation_success", "success", attempt=1))
    writer.close()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "invocation_success"
    assert payload["status"] == "success"
    assert payload["metadata"]["deterministic"] is True
    assert payload["agent_id"] == "mcp_server"
    assert payload["task_id"] == "06ab_core_tools_minimal_subset"


def test_core_tools_minimal_logging_failure_and_retry(tmp_path: Path) -> None:
    log_path = tmp_path / "logs.jsonl"
    writer = JsonLogWriter(
        log_path,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
        retention="keep-last-5",
    )

    failure = _event("invocation_failure", "error", attempt=1)
    failure.error = {"code": "SIMULATED", "message": "forced failure"}
    writer.write(failure)

    retry = _event("invocation_retry", "error", attempt=2)
    retry.error = {"code": "RETRY", "message": "retryable"}
    writer.write(retry)
    writer.flush()

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["error"] == {"code": "SIMULATED", "message": "forced failure"}
    assert events[1]["event"] == "invocation_retry"
    assert events[1]["attempt"] == 2


def test_core_tools_minimal_log_writer_retention(tmp_path: Path) -> None:
    base = tmp_path / "retained.jsonl"
    base.write_text("{}\n", encoding="utf-8")
    for idx in range(1, 6):
        rotated = base.with_suffix(base.suffix + f".{idx}")
        rotated.write_text(f"{idx}", encoding="utf-8")

    writer = JsonLogWriter(
        base,
        agent_id="mcp_server",
        task_id="06ab_core_tools_minimal_subset",
        retention="keep-last-5",
    )
    writer.write(_event("invocation_success", "success", attempt=1))
    writer.close()

    assert base.exists()
    # ensure highest rotation dropped and cascade happened
    assert not base.with_suffix(base.suffix + ".6").exists()
    assert base.with_suffix(base.suffix + ".5").exists()
