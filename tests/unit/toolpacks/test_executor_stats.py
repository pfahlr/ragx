"""Tests for the Toolpack executor metrics and caching semantics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from apps.toolpacks.executor import ExecutionStats, Executor
from apps.toolpacks.loader import Toolpack


@pytest.fixture
def sample_toolpack(tmp_path: Path) -> Toolpack:
    """Create a deterministic python toolpack pointing at ``toolpack_samples.echo``."""

    return Toolpack(
        id="tests.echo.tool",
        version="0.1.0",
        deterministic=True,
        timeout_ms=1000,
        limits={"maxInputBytes": 4096, "maxOutputBytes": 4096},
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "additionalProperties": True,
        },
        output_schema={
            "type": "object",
            "properties": {
                "echo": {
                    "type": "object",
                    "additionalProperties": True,
                }
            },
            "required": ["echo"],
            "additionalProperties": False,
        },
        execution={"kind": "python", "module": "tests.helpers.toolpack_samples:echo"},
        caps={},
        env={},
        templating={},
        source_path=tmp_path / "echo.tool.yaml",
    )


def _canonical_size(payload: dict[str, object]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_executor_records_duration_and_bytes(sample_toolpack: Toolpack) -> None:
    executor = Executor()
    payload = {"text": "hello"}

    result = executor.run_toolpack(sample_toolpack, payload)

    assert result["echo"]["text"] == "hello"

    stats = executor.last_run_stats()
    assert isinstance(stats, ExecutionStats)
    assert stats.cache_hit is False
    assert stats.input_bytes == _canonical_size(payload)
    assert stats.output_bytes == _canonical_size(result)
    assert stats.duration_ms >= 0


def test_executor_reports_cache_hit_on_deterministic_toolpack(sample_toolpack: Toolpack) -> None:
    executor = Executor()
    payload = {"text": "cached"}

    first = executor.run_toolpack(sample_toolpack, payload)
    first_stats = executor.last_run_stats()
    assert isinstance(first_stats, ExecutionStats)
    assert first_stats.cache_hit is False

    # Sleep to ensure measurable elapsed time even for cached hits.
    time.sleep(0.01)

    second = executor.run_toolpack(sample_toolpack, payload)
    assert second == first

    second_stats = executor.last_run_stats()
    assert isinstance(second_stats, ExecutionStats)
    assert second_stats.cache_hit is True
    assert second_stats.input_bytes == first_stats.input_bytes
    assert second_stats.output_bytes == first_stats.output_bytes
    assert second_stats.duration_ms >= 0
