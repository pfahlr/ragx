from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.toolpacks.executor import Executor
from apps.toolpacks.loader import Toolpack

pytest.importorskip("pydantic")


def _make_schema(properties: dict[str, Any], *, required: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _make_toolpack(*, module: str) -> Toolpack:
    return Toolpack(
        id="tests.deterministic.sum",
        version="1.0.0",
        deterministic=True,
        timeout_ms=2000,
        limits={"maxInputBytes": 2048, "maxOutputBytes": 2048},
        input_schema=_make_schema(
            {
                "values": {"type": "array", "items": {"type": "number"}},
                "text": {"type": "string"},
            },
            required=["values"],
        ),
        output_schema=_make_schema(
            {
                "sum": {"type": "number"},
                "text": {"type": "string"},
            },
            required=["sum"],
        ),
        execution={"kind": "python", "module": module},
        caps={},
        env={},
        templating={},
        source_path=Path("tests.deterministic.sum.tool.yaml"),
    )


def _payload_size(payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(encoded)


def test_executor_exposes_last_run_stats() -> None:
    module = "tests.toolpacks_runtime.deterministic_sum"
    toolpack = _make_toolpack(module=f"{module}:run")
    executor = Executor()

    assert executor.last_run_stats() is None

    payload = {"values": [1, 2, 3], "text": "ok"}
    result = executor.run_toolpack(toolpack, payload)

    assert result == {"sum": 6.0, "text": "ok"}

    stats = executor.last_run_stats()
    assert stats is not None
    assert stats.cache_hit is False
    assert stats.input_bytes == _payload_size(payload)
    assert stats.output_bytes == _payload_size(result)
    assert stats.duration_ms >= 0.0

    repeated = executor.run_toolpack(toolpack, payload)
    assert repeated == result

    cached_stats = executor.last_run_stats()
    assert cached_stats is not None
    assert cached_stats.cache_hit is True
    assert cached_stats.input_bytes == _payload_size(payload)
    assert cached_stats.output_bytes == _payload_size(result)
    assert cached_stats.duration_ms == pytest.approx(0.0, abs=1.0)
