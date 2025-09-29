from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from apps.toolpacks.loader import ToolpackLoader, ToolpackValidationError

try:
    from hypothesis import given as hypothesis_given
    from hypothesis import strategies as hypothesis_strategies
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    hypothesis_given = None
    hypothesis_strategies = None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised = contents.strip() + "\n"
    path.write_text(normalised, encoding="utf-8")


def _spec_compliant_toolpack(
    *,
    input_ref: str,
    output_ref: str,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "tool.echo",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 1000,
        "limits": {"maxInputBytes": 4096, "maxOutputBytes": 8192},
        "caps": {"cpu": "500m"},
        "env": {"LOG_LEVEL": "INFO"},
        "templating": {"prompt": "{{ text }}"},
        "inputSchema": {"$ref": input_ref},
        "outputSchema": {"$ref": output_ref},
        "execution": {"kind": "python", "module": "toolpacks.echo:run"},
    }
    if overrides:
        base.update(overrides)
    return base


def test_toolpack_loader_resolves_spec_fields(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    input_schema_path = schemas_dir / "echo.input.schema.json"
    output_schema_path = schemas_dir / "echo.output.schema.json"
    _write_json(
        input_schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    )
    _write_json(
        output_schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    _write_yaml(
        toolpacks_dir / "tool.echo.tool.yaml",
        yaml.safe_dump(
            _spec_compliant_toolpack(
                input_ref=os.path.relpath(input_schema_path, toolpacks_dir),
                output_ref=os.path.relpath(output_schema_path, toolpacks_dir),
                overrides={"env": {"LOG_LEVEL": "DEBUG"}},
            ),
            sort_keys=False,
        ),
    )

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)

    toolpack = loader.get("tool.echo")
    assert toolpack.timeout_ms == 1000
    assert toolpack.limits["maxInputBytes"] == 4096
    assert toolpack.execution["kind"] == "python"
    assert toolpack.env == {"LOG_LEVEL": "DEBUG"}
    assert toolpack.input_schema["required"] == ["prompt"]
    assert toolpack.output_schema["properties"]["text"]["type"] == "string"


def test_toolpack_loader_resolves_nested_refs(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    shared_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "inner": {
                "$ref": "./leaf.schema.json"
            }
        },
        "required": ["inner"],
    }
    leaf_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    _write_json(schemas_dir / "shared.schema.json", shared_schema)
    _write_json(schemas_dir / "leaf.schema.json", leaf_schema)

    toolpacks_dir = tmp_path / "toolpacks"
    _write_yaml(
        toolpacks_dir / "nested.tool.yaml",
        yaml.safe_dump(
            _spec_compliant_toolpack(
                input_ref="../schemas/shared.schema.json",
                output_ref="../schemas/leaf.schema.json",
            ),
            sort_keys=False,
        ),
    )

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)

    pack = loader.get("tool.echo")
    assert pack.input_schema["properties"]["inner"]["properties"]["value"]["type"] == "integer"


def test_toolpack_loader_rejects_snake_case_fields(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()

    _write_yaml(
        toolpacks_dir / "invalid.tool.yaml",
        """
        id: tool.invalid
        version: 1.0.0
        deterministic: true
        timeout_ms: 1000
        limits:
          max_input_bytes: 100
          max_output_bytes: 100
        execution:
          kind: python
          module: tool.invalid:run
        input_schema:
          type: object
        output_schema:
          type: object
        """,
    )

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    schema_path = tmp_path / "schemas" / "tiny.schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )
    missing = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"id": "missing.tool"},
    )
    missing.pop("timeoutMs")
    _write_yaml(
        toolpacks_dir / "missing.tool.yaml",
        yaml.safe_dump(missing, sort_keys=False),
    )

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="timeoutMs"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()

    schema_path = tmp_path / "schemas" / "noop.schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    base = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"id": "dup.tool"},
    )

    _write_yaml(toolpacks_dir / "first.tool.yaml", yaml.safe_dump(base, sort_keys=False))
    _write_yaml(
        toolpacks_dir / "second.tool.yaml",
        yaml.safe_dump({**base, "version": "1.0.1"}, sort_keys=False),
    )

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="Duplicate toolpack id"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_invalid_execution_kind(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "echo.schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    _write_yaml(
        toolpacks_dir / "tool.invalid.tool.yaml",
        yaml.safe_dump(
            _spec_compliant_toolpack(
                input_ref=os.path.relpath(schema_path, toolpacks_dir),
                output_ref=os.path.relpath(schema_path, toolpacks_dir),
                overrides={"execution": {"kind": "perl", "module": "tool.invalid:run"}},
            ),
            sort_keys=False,
        ),
    )

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_validates_schema_structure(tmp_path: Path) -> None:
    bad_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"prompt": {"type": "unsupported"}},
    }

    schemas_dir = tmp_path / "schemas"
    bad_schema_path = schemas_dir / "bad.schema.json"
    _write_json(bad_schema_path, bad_schema)

    good_schema_path = schemas_dir / "good.schema.json"
    _write_json(
        good_schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    _write_yaml(
        toolpacks_dir / "tool.bad.tool.yaml",
        yaml.safe_dump(
            _spec_compliant_toolpack(
                input_ref=os.path.relpath(bad_schema_path, toolpacks_dir),
                output_ref=os.path.relpath(good_schema_path, toolpacks_dir),
            ),
            sort_keys=False,
        ),
    )

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(toolpacks_dir)


if hypothesis_given is not None:  # pragma: no branch - definition guarded at import time

    @hypothesis_given(
        hypothesis_strategies.text(min_size=1).filter(
            lambda value: value not in {
                "null",
                "boolean",
                "object",
                "array",
                "number",
                "string",
                "integer",
            }
        )
    )
    def test_toolpack_loader_property_invalid_schema_types(
        invalid_type: str, tmp_path: Path
    ) -> None:
        schemas_dir = tmp_path / "schemas"
        invalid_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": invalid_type,
        }
        valid_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        }
        invalid_path = schemas_dir / "invalid.schema.json"
        valid_path = schemas_dir / "valid.schema.json"
        _write_json(invalid_path, invalid_schema)
        _write_json(valid_path, valid_schema)

        toolpacks_dir = tmp_path / "toolpacks"
        _write_yaml(
            toolpacks_dir / "property.tool.yaml",
            yaml.safe_dump(
                _spec_compliant_toolpack(
                    input_ref=os.path.relpath(invalid_path, toolpacks_dir),
                    output_ref=os.path.relpath(valid_path, toolpacks_dir),
                ),
                sort_keys=False,
            ),
        )

        loader = ToolpackLoader()
        with pytest.raises(ToolpackValidationError):
            loader.load_dir(toolpacks_dir)

else:

    @pytest.mark.skip("hypothesis is not installed")
    def test_toolpack_loader_property_invalid_schema_types() -> None:
        pytest.skip("hypothesis is not installed")
