from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack, ToolpackLoader

TOOLPACKS_DIR = Path("tests/stubs/toolpacks")
TOOL_ID = "tests.toolpacks.deterministicsum"


@pytest.fixture(scope="module")
def deterministic_sum_toolpack() -> Toolpack:
    loader = ToolpackLoader()
    loader.load_dir(TOOLPACKS_DIR)
    return loader.get(TOOL_ID)


def _patch_perf_counter(monkeypatch: pytest.MonkeyPatch, values: Iterable[float]) -> None:
    iterator: Iterator[float] = iter(values)
    monkeypatch.setattr(
        "apps.toolpacks.executor.time.perf_counter",
        lambda: next(iterator),
    )


def _json_size(payload: object) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_last_run_stats_records_bytes_and_cache_hits(
    monkeypatch: pytest.MonkeyPatch, deterministic_sum_toolpack: Toolpack
) -> None:
    _patch_perf_counter(monkeypatch, [10.0, 10.125, 20.0, 20.0])
    executor = Executor()

    payload = {"numbers": [1, 2, 3]}
    result = executor.run_toolpack(deterministic_sum_toolpack, payload)
    assert result["total"] == pytest.approx(6.0)

    stats_first = executor.last_run_stats()
    assert stats_first is not None
    assert stats_first.cache_hit is False
    assert stats_first.input_bytes == _json_size(payload)
    assert stats_first.output_bytes == _json_size(result)
    assert stats_first.duration_ms == pytest.approx(125.0)

    cached_result = executor.run_toolpack(deterministic_sum_toolpack, payload)
    assert cached_result == result

    stats_second = executor.last_run_stats()
    assert stats_second is not None
    assert stats_second.cache_hit is True
    assert stats_second.input_bytes == stats_first.input_bytes
    assert stats_second.output_bytes == stats_first.output_bytes
    assert stats_second.duration_ms == pytest.approx(0.0)


def test_last_run_stats_updates_after_failures(
    monkeypatch: pytest.MonkeyPatch, deterministic_sum_toolpack: Toolpack
) -> None:
    _patch_perf_counter(monkeypatch, [5.0, 5.05, 10.0, 10.25])
    executor = Executor()

    executor.run_toolpack(deterministic_sum_toolpack, {"numbers": [2, 3]})
    stats_success = executor.last_run_stats()
    assert stats_success is not None

    from tests.stubs.toolpacks import deterministic_sum

    def _boom(payload: dict[str, object]) -> dict[str, object]:
        """Runtime failure used to verify stats reset."""

        raise RuntimeError("boom")

    monkeypatch.setattr(deterministic_sum, "run", _boom)

    payload = {"numbers": [9]}
    with pytest.raises(ToolpackExecutionError):
        executor.run_toolpack(deterministic_sum_toolpack, payload)

    stats_failure = executor.last_run_stats()
    assert stats_failure is not None
    assert stats_failure.cache_hit is False
    assert stats_failure.input_bytes == _json_size(payload)
    assert stats_failure.output_bytes == 0
    assert stats_failure.duration_ms == pytest.approx(250.0)

    assert stats_failure != stats_success
