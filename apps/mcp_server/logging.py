from __future__ import annotations

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

__all__ = ["JsonLogWriter", "McpLogEvent"]


@dataclass(slots=True)
class McpLogEvent:
    """Structured event emitted for tool invocations."""

    ts: datetime
    agent_id: str
    task_id: str
    step_id: int
    trace_id: str
    span_id: str
    tool_id: str | None
    event: str
    status: str
    duration_ms: float
    attempt: int
    input_bytes: int
    output_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_serialisable(self) -> dict[str, Any]:
        """Return a spec-compliant dictionary for JSON serialisation."""

        return {
            "ts": self.ts.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "agentId": self.agent_id,
            "taskId": self.task_id,
            "stepId": self.step_id,
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "toolId": self.tool_id,
            "event": self.event,
            "status": self.status,
            "durationMs": self.duration_ms,
            "attempt": self.attempt,
            "inputBytes": self.input_bytes,
            "outputBytes": self.output_bytes,
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class JsonLogWriter:
    """Persist structured events to JSONL with rotation."""

    def __init__(
        self,
        *,
        agent_id: str,
        task_id: str,
        storage_prefix: Path,
        latest_symlink: Path,
        schema_version: str,
        deterministic: bool,
        retention: int = 5,
    ) -> None:
        self._agent_id = agent_id
        self._task_id = task_id
        self._storage_prefix = storage_prefix
        self._latest_symlink = latest_symlink
        self._schema_version = schema_version
        self._deterministic = deterministic
        self._retention = retention
        self._run_id = str(uuid4())

        self._path = self._initialise_log_path()
        self._file = self._path.open("w", encoding="utf-8")
        self._enforce_retention()

    def _initialise_log_path(self) -> Path:
        prefix = self._storage_prefix
        directory = prefix.parent
        directory.mkdir(parents=True, exist_ok=True)
        latest_dir = self._latest_symlink.parent
        latest_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = f"{prefix.name}.{timestamp}.{self._run_id}.jsonl"
        path = directory / name

        # Update latest symlink atomically.
        tmp_symlink = latest_dir / f".{self._latest_symlink.name}.tmp"
        relative = os.path.relpath(path, latest_dir)
        if tmp_symlink.exists() or tmp_symlink.is_symlink():
            tmp_symlink.unlink()
        tmp_symlink.symlink_to(relative)
        os.replace(tmp_symlink, self._latest_symlink)
        return path

    def _enforce_retention(self) -> None:
        directory = self._storage_prefix.parent
        pattern = f"{self._storage_prefix.name}*.jsonl"
        candidates = [
            path
            for path in directory.glob(pattern)
            if path.exists() and not path.is_symlink()
        ]
        files = sorted(
            candidates,
            key=lambda item: (item.stat().st_mtime, item.name),
        )
        excess = len(files) - self._retention
        for path in files[:max(excess, 0)]:
            try:
                path.unlink()
            except OSError:
                pass

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    def new_attempt_id(self) -> str:
        return str(uuid4())

    def write(self, event: McpLogEvent, *, attempt_id: str) -> None:
        record = event.to_serialisable()
        metadata = dict(record.get("metadata", {}))
        metadata.update(
            {
                "runId": self._run_id,
                "attemptId": attempt_id,
                "schemaVersion": self._schema_version,
                "deterministic": self._deterministic,
                "logPath": self._relative_log_path(),
            }
        )
        record["metadata"] = metadata
        record.setdefault("error", None)
        record.setdefault("agentId", self._agent_id)
        record.setdefault("taskId", self._task_id)
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> JsonLogWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _relative_log_path(self) -> str:
        parent_parts = list(self._storage_prefix.parent.parts)
        tail = parent_parts[-2:] if len(parent_parts) >= 2 else parent_parts
        return str(Path(*tail, self._path.name))
