from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from apps.toolpacks.executor import ExecutionContext, ToolpackExecutor
from apps.toolpacks.loader import ToolpackLoader


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_python_module(
    tmp_path: Path,
    body: str,
    *,
    package_name: str = "toolpacks_unit",
) -> str:
    package_dir = tmp_path / package_name
    _write(package_dir / "__init__.py", "")
    module_path = package_dir / "handlers.py"
    _write(module_path, body)
    return f"{package_name}.handlers"


def _prepare_toolpack(tmp_path: Path) -> tuple[ToolpackLoader, Path]:
    module = _create_python_module(
        tmp_path,
        """
        def greet(payload, ctx):
            prefix = ctx.env.get("GREETING_PREFIX", "Hello")
            return {"message": f"{prefix} {payload['name']}!"}

        def invalid_output(payload):
            return {"missing": "field"}
        """,
    )

    schemas_dir = tmp_path / "schemas"
    _write_json(
        schemas_dir / "input.schema.json",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    _write_json(
        schemas_dir / "output.schema.json",
        {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    _write(
        toolpacks_dir / "greet.tool.yaml",
        f"""
        id: tool.greet
        version: 1.0.0
        kind: python
        deterministic: true
        execution:
          runtime: python
          handler: {module}:greet
        input_schema:
          $ref: ../schemas/input.schema.json
        output_schema:
          $ref: ../schemas/output.schema.json
        env:
          GREETING_PREFIX: Hey
        """,
    )

    _write(
        toolpacks_dir / "invalid_output.tool.yaml",
        f"""
        id: tool.invalid
        version: 0.1.0
        kind: python
        execution:
          runtime: python
          handler: {module}:invalid_output
        input_schema:
          $ref: ../schemas/input.schema.json
        output_schema:
          $ref: ../schemas/output.schema.json
        """,
    )

    loader = ToolpackLoader.load_dir(toolpacks_dir)
    return loader, tmp_path


@pytest.fixture
def loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToolpackLoader:
    loader, module_root = _prepare_toolpack(tmp_path)
    monkeypatch.syspath_prepend(str(module_root))
    return loader


@pytest.fixture
def executor(loader: ToolpackLoader) -> ToolpackExecutor:
    return ToolpackExecutor(loader=loader, base_environment={"GREETING_PREFIX": "Hello"})


def test_executor_runs_python_toolpack(executor: ToolpackExecutor) -> None:
    result = executor.run("tool.greet", {"name": "Ada"})
    assert result == {"message": "Hey Ada!"}


def test_executor_validates_input(executor: ToolpackExecutor) -> None:
    with pytest.raises(ValueError, match="input payload"):
        executor.run("tool.greet", {"bad": "data"})


def test_executor_validates_output(loader: ToolpackLoader) -> None:
    executor = ToolpackExecutor(loader=loader)
    with pytest.raises(ValueError, match="output payload"):
        executor.run("tool.invalid", {"name": "Bob"})


def test_executor_supports_direct_toolpack(loader: ToolpackLoader) -> None:
    executor = ToolpackExecutor(base_environment={"GREETING_PREFIX": "Hi"})
    toolpack = loader.get("tool.greet")
    context = ExecutionContext(env={"GREETING_PREFIX": "Yo"})
    result = executor.run_toolpack(toolpack, {"name": "Lin"}, context=context)
    assert result == {"message": "Yo Lin!"}


def test_executor_caches_deterministic_results(executor: ToolpackExecutor) -> None:
    first = executor.run("tool.greet", {"name": "Ada"})
    first["message"] = "mutated"
    second = executor.run("tool.greet", {"name": "Ada"})
    assert second == {"message": "Hey Ada!"}


def test_executor_rejects_non_python(loader: ToolpackLoader) -> None:
    toolpack = loader.get("tool.greet")
    toolpack.config["kind"] = "node"
    executor = ToolpackExecutor()
    with pytest.raises(ValueError, match="Unsupported toolpack kind"):
        executor.run_toolpack(toolpack, {"name": "Lin"})
