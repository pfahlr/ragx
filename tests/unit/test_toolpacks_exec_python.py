from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from apps.toolpacks.executor import Executor, ToolpackExecutionError
from apps.toolpacks.loader import Toolpack


def _make_schema(required: str) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            required: {"type": "number"},
        },
        "required": [required],
    }


def _make_toolpack(
    *,
    module: str,
    deterministic: bool = True,
) -> Toolpack:
    return Toolpack(
        id="calc.double",
        version="1.0.0",
        deterministic=deterministic,
        timeout_ms=1000,
        limits={"maxInputBytes": 4096, "maxOutputBytes": 4096},
        input_schema=_make_schema("value"),
        output_schema=_make_schema("result"),
        execution={"kind": "python", "module": module},
        caps={},
        env={},
        templating={},
        source_path=Path("calc.double.tool.yaml"),
    )


def _register_module(monkeypatch: pytest.MonkeyPatch, name: str, func) -> None:
    module = types.ModuleType(name)
    module.run = func  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)


def test_exec_python_toolpack_runs_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def run(payload: dict[str, object]) -> dict[str, object]:
        captured.append(payload)
        value = payload["value"]
        assert isinstance(value, int | float)
        return {"result": value * 2}

    module_name = "toolpacks_tests.runtime"
    _register_module(monkeypatch, module_name, run)
    toolpack = _make_toolpack(module=f"{module_name}:run")

    executor = Executor()
    result = executor.run_toolpack(toolpack, {"value": 4})

    assert result == {"result": 8}
    assert captured == [{"value": 4}]


def test_exec_python_toolpack_validates_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(payload: dict[str, object]) -> dict[str, object]:
        return {"result": payload.get("value", 0)}

    module_name = "toolpacks_tests.validation"
    _register_module(monkeypatch, module_name, run)
    toolpack = _make_toolpack(module=f"{module_name}:run")

    executor = Executor()

    with pytest.raises(ToolpackExecutionError) as excinfo:
        executor.run_toolpack(toolpack, {})

    assert "input" in str(excinfo.value)


def test_exec_python_toolpack_validates_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(payload: dict[str, object]) -> dict[str, object]:
        return {"unexpected": payload["value"]}

    module_name = "toolpacks_tests.output"
    _register_module(monkeypatch, module_name, run)
    toolpack = _make_toolpack(module=f"{module_name}:run")

    executor = Executor()

    with pytest.raises(ToolpackExecutionError) as excinfo:
        executor.run_toolpack(toolpack, {"value": 1})

    assert "output" in str(excinfo.value)


def test_exec_python_toolpack_idempotent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def run(payload: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        return {"result": payload["value"]}

    module_name = "toolpacks_tests.cache"
    _register_module(monkeypatch, module_name, run)
    toolpack = _make_toolpack(module=f"{module_name}:run")

    executor = Executor()
    first = executor.run_toolpack(toolpack, {"value": 3})
    second = executor.run_toolpack(toolpack, {"value": 3})

    assert calls == [1]
    assert first == second == {"result": 3}

    first["result"] = 0
    cached = executor.run_toolpack(toolpack, {"value": 3})
    assert cached == {"result": 3}


def test_exec_python_toolpack_skips_cache_for_non_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def run(payload: dict[str, object]) -> dict[str, object]:
        calls.append(1)
        return {"result": payload["value"]}

    module_name = "toolpacks_tests.cache_disabled"
    _register_module(monkeypatch, module_name, run)
    toolpack = _make_toolpack(module=f"{module_name}:run", deterministic=False)

    executor = Executor()
    executor.run_toolpack(toolpack, {"value": 5})
    executor.run_toolpack(toolpack, {"value": 5})

    assert calls == [1, 1]
