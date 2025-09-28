from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from apps.toolpacks.loader import ToolpackLoader


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised = textwrap.dedent(contents).strip() + "\n"
    path.write_text(normalised, encoding="utf-8")


def test_toolpack_loader_resolves_refs(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    _write_json(
        schemas_dir / "echo.input.schema.json",
        {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["prompt"],
        },
    )
    _write_json(
        schemas_dir / "echo.output.schema.json",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    _write_yaml(
        toolpacks_dir / "echo.tool.yaml",
        """
        id: tool.echo
        version: 0.1.0
        kind: python
        deterministic: true
        timeout_ms: 1000
        execution:
          runtime: python
          handler: toolpacks.echo:run
        input_schema:
          $ref: ../schemas/echo.input.schema.json
        output_schema:
          $ref: ../schemas/echo.output.schema.json
        env:
          LOG_LEVEL: INFO
        """,
    )

    loader = ToolpackLoader.load_dir(toolpacks_dir)

    toolpacks = loader.list()
    assert [toolpack.id for toolpack in toolpacks] == ["tool.echo"]

    tool = loader.get("tool.echo")
    assert tool.config["execution"]["handler"] == "toolpacks.echo:run"
    assert tool.input_schema["required"] == ["prompt"]
    assert tool.output_schema["properties"]["text"]["type"] == "string"
    assert tool.path == (toolpacks_dir / "echo.tool.yaml").resolve()


def test_toolpack_loader_rejects_duplicates(tmp_path: Path) -> None:
    schema_path = tmp_path / "schemas" / "noop.output.schema.json"
    _write_json(schema_path, {"type": "object"})

    toolpacks_dir = tmp_path / "toolpacks"
    yaml_body = """
    id: tool.duplicate
    version: 1.0.0
    output_schema:
      $ref: ../schemas/noop.output.schema.json
    """
    _write_yaml(toolpacks_dir / "first.tool.yaml", yaml_body)
    _write_yaml(toolpacks_dir / "second.tool.yaml", yaml_body)

    with pytest.raises(ValueError, match="duplicate toolpack id 'tool.duplicate'"):
        ToolpackLoader.load_dir(toolpacks_dir)


def test_toolpack_loader_validates_required_fields(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", "version: 0.1.0\n")

    with pytest.raises(ValueError, match="missing required field 'id'"):
        ToolpackLoader.load_dir(toolpacks_dir)


def test_toolpack_loader_missing_ref_file(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    _write_yaml(
        toolpacks_dir / "broken.tool.yaml",
        """
        id: tool.broken
        version: 0.0.1
        input_schema:
          $ref: ./schemas/missing.json
        """,
    )

    with pytest.raises(FileNotFoundError, match="schemas/missing.json"):
        ToolpackLoader.load_dir(toolpacks_dir)
