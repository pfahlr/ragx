from __future__ import annotations

import json
import textwrap
from pathlib import Path

from apps.toolpacks.executor import ToolpackExecutor
from apps.toolpacks.loader import ToolpackLoader


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).strip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_python_toolpack_roundtrip(tmp_path: Path, monkeypatch) -> None:
    package_name = "toolpacks_e2e"
    package_dir = tmp_path / package_name
    _write(package_dir / "__init__.py", "")
    _write(
        package_dir / "handlers.py",
        """
        def echo(payload, ctx):
            return {"echo": payload["text"], "context": dict(ctx.env)}
        """,
    )

    schemas_dir = tmp_path / "schemas"
    _write_json(
        schemas_dir / "input.schema.json",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    _write_json(
        schemas_dir / "output.schema.json",
        {
            "type": "object",
            "properties": {
                "echo": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["echo", "context"],
        },
    )

    toolpacks_dir = tmp_path / "runtime-toolpacks"
    _write(
        toolpacks_dir / "echo.tool.yaml",
        """
        id: tool.echo
        version: 0.0.1
        kind: python
        deterministic: true
        execution:
          runtime: python
          handler: toolpacks_e2e.handlers:echo
        input_schema:
          $ref: ../schemas/input.schema.json
        output_schema:
          $ref: ../schemas/output.schema.json
        env:
          MODE: integration
        """,
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    loader = ToolpackLoader.load_dir(toolpacks_dir)
    executor = ToolpackExecutor(loader=loader, base_environment={"BASE": "yes"})

    result = executor.run("tool.echo", {"text": "hello"})

    assert result["echo"] == "hello"
    assert result["context"]["MODE"] == "integration"
    assert result["context"]["BASE"] == "yes"

    # cache hit should return a deep copy
    result["echo"] = "mutated"
    repeat = executor.run("tool.echo", {"text": "hello"})
    assert repeat["echo"] == "hello"
