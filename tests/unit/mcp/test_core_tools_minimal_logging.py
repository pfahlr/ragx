from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from apps.mcp_server.logging import JsonLogWriter, McpLogEvent


def _event(event: str, status: str, attempt: int) -> McpLogEvent:
    return McpLogEvent(
        ts=datetime.now(UTC),
        agent_id="mcp_server",
        task_id="06aV2",
        step_id=attempt,
        trace_id=str(uuid4()),
        span_id=str(uuid4()),
        tool_id="mcp.tool:exports.render.markdown",
        event=event,
        status=status,
        attempt=attempt,
        execution={"durationMs": 12.5, "inputBytes": 42, "outputBytes": 128},
        idempotency={"cacheHit": False},
        metadata={"schemaVersion": "0.1.0"},
        error=None,
    )


def test_log_writer_serialises_event(tmp_path: Path) -> None:
    prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    writer = JsonLogWriter(
        agent_id="test-agent",
        task_id="core-tools",
        storage_prefix=prefix,
        latest_symlink=latest,
        schema_version="0.1.0",
        deterministic=True,
        root_dir=tmp_path,
    )

    event = _event("tool.ok", "ok", 0)
    writer.write(event, attempt_id=writer.new_attempt_id())
    writer.close()

    payloads = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    record = payloads[0]
    for required in (
        "ts",
        "agentId",
        "taskId",
        "stepId",
        "traceId",
        "spanId",
        "toolId",
        "event",
        "status",
        "attempt",
        "execution",
        "idempotency",
        "metadata",
    ):
        assert required in record
    assert record["metadata"]["schemaVersion"] == "0.1.0"
    assert record["metadata"]["deterministic"] is True
    assert record["logPath"] == str(latest.relative_to(tmp_path))
    assert record["runId"]
    assert record["attemptId"]


def test_log_rotation_keeps_latest_symlink(tmp_path: Path) -> None:
    prefix = tmp_path / "runs/core_tools/minimal"
    latest = tmp_path / "runs/core_tools/minimal.jsonl"
    created_paths: list[Path] = []

    for attempt in range(6):
        writer = JsonLogWriter(
            agent_id="test-agent",
            task_id="core-tools",
            storage_prefix=prefix,
            latest_symlink=latest,
            schema_version="0.1.0",
            deterministic=True,
            root_dir=tmp_path,
        )
        writer.write(_event("tool.ok", "ok", attempt), attempt_id=writer.new_attempt_id())
        writer.close()
        created_paths.append(writer.path)

    run_files = sorted(
        (
            path
            for path in prefix.parent.glob("minimal*.jsonl")
            if path.exists() and not path.is_symlink()
        ),
        key=lambda item: (item.stat().st_mtime, item.name),
    )
    assert len(run_files) == 5, "Rotation should keep last five files"
    assert latest.is_symlink()
    assert latest.resolve() == created_paths[-1]
    assert created_paths[-1] in run_files
    assert len(set(created_paths) - set(run_files)) >= 1, "Rotation must drop older logs"
