from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from apps.toolpacks.loader import (
    ToolpackLoader,
    ToolpackValidationError,
    _apply_json_pointer,
    _require_mapping,
)

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
        "caps": {"network": ["https"], "subprocess": False},
        "env": {"passthrough": ["LOG_LEVEL"], "set": {"REGION": "us-east-1"}},
        "templating": {
            "engine": "jinja2",
            "cacheKey": "{{ id }}",
            "context": {"stable": True},
        },
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
                overrides={
                    "caps": {"network": ["https"], "subprocess": True},
                    "env": {
                        "passthrough": ["REQUEST_ID"],
                        "set": {"LOG_LEVEL": "DEBUG"},
                    },
                    "templating": {
                        "engine": "jinja2",
                        "cacheKey": "{{ id }}",
                        "context": {"stable": False},
                    },
                },
            ),
            sort_keys=False,
        ),
    )

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)

    toolpack_ids = [pack.id for pack in loader.list()]
    assert toolpack_ids == ["tool.echo"]

    toolpack = loader.get("tool.echo")
    assert toolpack.timeout_ms == 1000
    assert toolpack.limits["maxInputBytes"] == 4096
    assert toolpack.execution["kind"] == "python"
    assert toolpack.caps == {"network": ["https"], "subprocess": True}
    assert toolpack.env == {
        "passthrough": ["REQUEST_ID"],
        "set": {"LOG_LEVEL": "DEBUG"},
    }
    assert toolpack.templating == {
        "engine": "jinja2",
        "cacheKey": "{{ id }}",
        "context": {"stable": False},
    }
    assert toolpack.input_schema["required"] == ["prompt"]
    assert toolpack.output_schema["properties"]["text"]["type"] == "string"


def test_require_mapping_returns_empty_for_none() -> None:
    assert _require_mapping(None, "env", "tool") == {}


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


def test_toolpack_loader_shims_snake_case_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "simple.schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()

    legacy_config = {
        "id": "tool.legacy",
        "version": "1.0.0",
        "deterministic": True,
        "timeout_ms": 1000,
        "limits": {"max_input_bytes": 100, "max_output_bytes": 200},
        "execution": {"kind": "python", "module": "tool.legacy:run"},
        "input_schema": {
            "$ref": os.path.relpath(schema_path, toolpacks_dir),
        },
        "output_schema": {
            "$ref": os.path.relpath(schema_path, toolpacks_dir),
        },
    }
    _write_yaml(
        toolpacks_dir / "legacy.tool.yaml",
        yaml.safe_dump(legacy_config, sort_keys=False),
    )

    caplog.set_level("WARNING")
    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)

    legacy = loader.get("tool.legacy")
    assert legacy.timeout_ms == 1000
    assert legacy.limits["maxInputBytes"] == 100
    assert legacy.input_schema == legacy.output_schema

    warning_messages = " ".join(record.message for record in caplog.records)
    assert "legacy snake_case" in warning_messages


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


def test_toolpack_loader_python_requires_module_format(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "python", "module": "pkg.module"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="module must use"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_python_accepts_script_only(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    config["execution"] = {"kind": "python", "script": "python run.py"}
    _write_yaml(toolpacks_dir / "script.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    assert loader.get("tool.echo").execution["script"] == "python run.py"


def test_toolpack_loader_python_module_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "python", "module": ""}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="module must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_python_script_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    config["execution"] = {"kind": "python", "script": ""}
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="script must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_python_requires_entrypoint(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "python"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="requires 'module' or 'script'"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_requires_boolean_deterministic(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"deterministic": "yes"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="deterministic must be a boolean"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_requires_positive_timeout(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"timeoutMs": 0},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="timeoutMs must be a positive integer"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_limits_require_keys(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"limits": {"maxInputBytes": 10}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(
        ToolpackValidationError,
        match="limits missing required key 'maxOutputBytes'",
    ):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_limits_require_positive_values(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"limits": {"maxInputBytes": -1, "maxOutputBytes": 10}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(
        ToolpackValidationError,
        match=r"limits\['maxInputBytes'\] must be a positive integer",
    ):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_must_be_mapping(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": "not-a-mapping"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="field 'caps' must be a mapping"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_network_requires_non_empty_strings(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"network": [""]}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"network\[0\] must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_network_unsupported_protocol(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"network": ["ftp"]}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="unsupported protocol 'ftp'"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_network_must_be_string_or_list(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"network": {"protocol": "https"}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="network must be a string or list"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_filesystem_mode_supported(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"filesystem": {"delete": []}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"filesystem\[delete\] is not supported"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_filesystem_paths_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"filesystem": {"read": [""]}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError) as excinfo:
        loader.load_dir(toolpacks_dir)
    assert "filesystem[read][0]" in str(excinfo.value)


def test_toolpack_loader_caps_subprocess_must_be_boolean(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"subprocess": "yes"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="caps.subprocess must be a boolean"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_filesystem_validates_success(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"filesystem": {"read": ["/data"], "write": ["/tmp"]}}},
    )
    _write_yaml(toolpacks_dir / "valid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    caps = loader.get("tool.echo").caps
    assert caps["filesystem"] == {"read": ["/data"], "write": ["/tmp"]}


def test_toolpack_loader_env_passthrough_must_be_list(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": "TOKEN", "set": {}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="env.passthrough must be a list"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_env_passthrough_entries_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": [""], "set": {}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError) as excinfo:
        loader.load_dir(toolpacks_dir)
    assert "passthrough[0]" in str(excinfo.value)


def test_toolpack_loader_env_unknown_key(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": [], "extra": []}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="env contains unknown key 'extra'"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_env_set_keys_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": [], "set": {"": "value"}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="env.set keys must be non-empty strings"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_env_set_keys_must_match_pattern(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": [], "set": {"lower": "value"}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(
        ToolpackValidationError,
        match=r"env.set key 'lower' must be uppercase",
    ):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_env_must_be_mapping(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": "not-a-mapping"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="field 'env' must be a mapping"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_optional_sections_default_to_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    config.pop("caps", None)
    config.pop("env", None)
    config.pop("templating", None)
    _write_yaml(toolpacks_dir / "minimal.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    pack = loader.get("tool.echo")
    assert pack.caps == {}
    assert pack.env == {}
    assert pack.templating == {}


def test_toolpack_loader_cli_requires_cmd_list(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "cli", "cmd": "echo"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="cli execution requires 'cmd' list"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_cli_args_must_be_non_empty_strings(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "cli", "cmd": ["", "run"]}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"cmd\[0\] must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    _write_yaml(toolpacks_dir / "broken.tool.yaml", "- item\n")

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="Expected mapping"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_invalid_yaml(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    (toolpacks_dir / "broken.tool.yaml").write_text("[unbalanced", encoding="utf-8")

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="Failed to parse YAML"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_directory_not_found(tmp_path: Path) -> None:
    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="Toolpacks directory not found"):
        loader.load_dir(tmp_path / "missing")


def test_toolpack_loader_http_requires_string_headers(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {
                "kind": "http",
                "url": "https://example.com",
                "headers": {"Auth": 123},
            }
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="header 'Auth' must be a string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_http_method_must_be_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "http", "url": "https://example.com", "method": ""}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="method must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_http_header_keys_non_empty(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {
                "kind": "http",
                "url": "https://example.com",
                "headers": {"": "value"},
            }
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="headers must use non-empty string keys"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_http_requires_url(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "http"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="http execution requires 'url'"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_http_timeout_positive(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {"kind": "http", "url": "https://example.com", "timeoutMs": 0}
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="timeoutMs must be a positive integer"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_node_requires_entry(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "node"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="node execution requires"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_node_args_must_be_list(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {
                "kind": "node",
                "script": "loader.mjs",
                "args": "--flag",
            }
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="node execution args must be a list"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_node_args_must_be_strings(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {
                "kind": "node",
                "script": "loader.mjs",
                "args": ["--flag", 123],
            }
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"args\[1\] must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_php_requires_entry(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"execution": {"kind": "php"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="php execution requires"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_php_binary_must_be_string(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={
            "execution": {
                "kind": "php",
                "php": "tool.php",
                "phpBinary": False,
            }
        },
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="phpBinary must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_invalid_tool_id(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"id": "InvalidTool"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="dotted lowercase"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_requires_non_empty_id(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"id": ""},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="Field 'id' in toolpack"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_invalid_version(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"version": "2024.1"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"major\.minor\.patch"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_unparseable_version(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"version": "not-a-version!"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="follow semantic versioning"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_unknown_caps_key(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"network": ["https"], "invalid": True}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="caps contains unknown key"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_caps_network_normalises_protocols(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    legacy = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"network": "HTTPS"}},
    )
    _write_yaml(toolpacks_dir / "valid.tool.yaml", yaml.safe_dump(legacy, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)

    caps = loader.get("tool.echo").caps
    assert caps["network"] == ["https"]


def test_toolpack_loader_caps_filesystem_requires_list(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"caps": {"filesystem": {"read": "./data"}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"caps.filesystem\[read\] must be a list"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_env_passthrough_case(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": ["lower"], "set": {}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="env.passthrough"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_env_set_values_must_be_strings(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"env": {"passthrough": ["TOKEN"], "set": {"VALUE": 1}}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"env.set\['VALUE'\] must be a string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_rejects_templating_engine(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"templating": {"engine": "liquid"}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="templating.engine"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_templating_must_be_mapping(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"templating": "not-a-mapping"},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="field 'templating' must be a mapping"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_templating_unknown_key(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"templating": {"engine": "jinja2", "unknown": True}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="templating contains unknown key 'unknown'"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_templating_requires_cache_key(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    invalid = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
        overrides={"templating": {"engine": "jinja2", "cacheKey": ""}},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(invalid, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="cacheKey must be a non-empty string"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_templating_context_must_be_json(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    templating_yaml = (
        "id: tool.context\n"
        "version: 1.0.0\n"
        "deterministic: true\n"
        "timeoutMs: 1000\n"
        "limits:\n"
        "  maxInputBytes: 10\n"
        "  maxOutputBytes: 10\n"
        "execution:\n"
        "  kind: python\n"
        "  module: tool.context:run\n"
        f"inputSchema:\n  $ref: {os.path.relpath(schema_path, toolpacks_dir)}\n"
        f"outputSchema:\n  $ref: {os.path.relpath(schema_path, toolpacks_dir)}\n"
        "templating:\n"
        "  context:\n"
        "    invalid: !!set\n"
        "      ? foo\n"
        "      ? bar\n"
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", templating_yaml)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="context must be JSON serialisable"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_schema_ref_not_found(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref="missing.schema.json",
        output_ref="missing.schema.json",
    )
    _write_yaml(toolpacks_dir / "missing.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="schema reference not found"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_schema_yaml_parse_error(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text("[unbalanced", encoding="utf-8")

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="failed to parse schema"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_schema_json_parse_error(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{ not-json", encoding="utf-8")

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="failed to load schema"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_schema_json_pointer(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=f"{os.path.relpath(schema_path, toolpacks_dir)}#/properties/value",
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "pointer.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    schema = loader.get("tool.echo").input_schema
    assert schema["type"] == "string"


def test_toolpack_loader_schema_json_pointer_list_index(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "allOf": [
                {"type": "string"},
                {"type": "number"},
            ],
        },
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=f"{os.path.relpath(schema_path, toolpacks_dir)}#allOf/1",
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "pointer.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    schema = loader.get("tool.echo").input_schema
    assert schema["type"] == "number"


def test_toolpack_loader_schema_pointer_empty_fragment(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=f"{os.path.relpath(schema_path, toolpacks_dir)}#",
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "pointer.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    assert loader.get("tool.echo").input_schema["type"] == "object"


def test_toolpack_loader_schema_pointer_root(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_doc = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "Root",
    }
    _write_json(schema_path, schema_doc)

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=f"{os.path.relpath(schema_path, toolpacks_dir)}#/",
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "pointer.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    loader.load_dir(toolpacks_dir)
    assert loader.get("tool.echo").input_schema["title"] == "Root"


def test_apply_json_pointer_root_returns_document() -> None:
    document = {"key": "value"}
    assert _apply_json_pointer(document, None) == document


def test_toolpack_loader_schema_open_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    original_open = Path.open

    def _raise_os_error(self: Path, *args, **kwargs):
        if self == schema_path:
            raise OSError("boom")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raise_os_error)

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="failed to open schema"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_input_schema_must_be_mapping(tmp_path: Path) -> None:
    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref="not-used",
        output_ref="not-used",
        overrides={"inputSchema": 123},
    )
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match="schema definition must be a mapping"):
        loader.load_dir(toolpacks_dir)


def test_toolpack_loader_schema_ref_must_be_string(tmp_path: Path) -> None:
    schemas_dir = tmp_path / "schemas"
    schema_path = schemas_dir / "schema.json"
    _write_json(
        schema_path,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )

    toolpacks_dir = tmp_path / "toolpacks"
    toolpacks_dir.mkdir()
    config = _spec_compliant_toolpack(
        input_ref=os.path.relpath(schema_path, toolpacks_dir),
        output_ref=os.path.relpath(schema_path, toolpacks_dir),
    )
    config["inputSchema"] = {"$ref": 123}
    _write_yaml(toolpacks_dir / "invalid.tool.yaml", yaml.safe_dump(config, sort_keys=False))

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError, match=r"schema \$ref must be a non-empty string"):
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
