import json
import os
from pathlib import Path

import pytest
import yaml

from apps.toolpacks.loader import ToolpackLoader, ToolpackValidationError


@pytest.fixture
def schema_dir(tmp_path: Path) -> Path:
    schema_root = tmp_path / "schemas"
    schema_root.mkdir(parents=True)
    return schema_root


def _write_schema(path: Path, name: str, schema: dict) -> Path:
    file_path = path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(schema))
    return file_path


def _write_toolpack(path: Path, name: str, data: dict) -> Path:
    file_path = path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return file_path


def test_load_toolpacks(tmp_path: Path, schema_dir: Path) -> None:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["results"],
    }

    input_schema_path = _write_schema(schema_dir, "search.input.schema.json", input_schema)
    output_schema_path = _write_schema(schema_dir, "search.output.schema.json", output_schema)

    packs_dir = tmp_path / "toolpacks"
    packs_dir.mkdir()

    toolpack_data = {
        "id": "search.query",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 5000,
        "limits": {"maxInputBytes": 4096, "maxOutputBytes": 8192},
        "execution": {"kind": "python", "module": "pkg.tool:run"},
        "caps": {"network": ["https"], "subprocess": False},
        "env": {"passthrough": ["API_TOKEN"], "set": {"REGION": "us-east"}},
        "templating": {
            "engine": "jinja2",
            "cacheKey": "{{ id }}:{{ version }}",
            "context": {"stable": True},
        },
    }
    toolpack_data["inputSchema"] = {"$ref": os.path.relpath(input_schema_path, packs_dir)}
    toolpack_data["outputSchema"] = {"$ref": os.path.relpath(output_schema_path, packs_dir)}

    _write_toolpack(packs_dir, "search.query.tool.yaml", toolpack_data)
    nested_dir = packs_dir / "beta"
    nested_toolpack_data = {
        **toolpack_data,
        "id": "search.summary",
        "execution": {"kind": "cli", "cmd": ["echo", "hello"]},
        "inputSchema": {"$ref": os.path.relpath(input_schema_path, nested_dir)},
        "outputSchema": {"$ref": os.path.relpath(output_schema_path, nested_dir)},
    }
    _write_toolpack(nested_dir, "search.summary.tool.yaml", nested_toolpack_data)

    loader = ToolpackLoader()
    loader.load_dir(tmp_path)

    ids = [pack.id for pack in loader.list()]
    assert ids == sorted(ids)

    query_pack = loader.get("search.query")
    assert query_pack.input_schema == input_schema
    assert query_pack.output_schema == output_schema
    assert query_pack.execution["kind"] == "python"
    assert query_pack.source_path.name == "search.query.tool.yaml"

    summary_pack = loader.get("search.summary")
    assert summary_pack.execution["kind"] == "cli"


def test_missing_required_field(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "input.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpack_data = {
        "id": "broken.tool",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 1000,
        "limits": {"maxInputBytes": 1, "maxOutputBytes": 1},
        "inputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
        "outputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
        # missing execution
    }

    _write_toolpack(tmp_path, "broken.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_duplicate_tool_ids(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    base_data = {
        "id": "dup.tool",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 1000,
        "limits": {"maxInputBytes": 1, "maxOutputBytes": 1},
        "execution": {"kind": "http", "url": "https://example.com"},
        "inputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
        "outputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
    }

    _write_toolpack(tmp_path, "one.tool.yaml", base_data)
    _write_toolpack(tmp_path, "two.tool.yaml", {**base_data, "version": "2.0.0"})

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_get_missing_tool_raises_key_error(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpack_data = {
        "id": "exists.tool",
        "version": "1.0.0",
        "deterministic": True,
        "timeoutMs": 1000,
        "limits": {"maxInputBytes": 1, "maxOutputBytes": 1},
        "execution": {"kind": "python", "module": "pkg.tool:run"},
        "inputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
        "outputSchema": {"$ref": os.path.relpath(schema_path, tmp_path)},
    }

    _write_toolpack(tmp_path, "exists.tool.yaml", toolpack_data)

    loader = ToolpackLoader()
    loader.load_dir(tmp_path)

    with pytest.raises(KeyError):
        loader.get("missing.tool")


def _build_valid_toolpack(
    *,
    schema_path: Path,
    toolpack_dir: Path,
    tool_id: str = "example.tool",
    overrides: dict | None = None,
) -> dict:
    base = {
        "id": tool_id,
        "version": "1.2.3",
        "deterministic": True,
        "timeoutMs": 2000,
        "limits": {"maxInputBytes": 1024, "maxOutputBytes": 2048},
        "execution": {"kind": "python", "module": "pkg.mod:run"},
        "inputSchema": {"$ref": os.path.relpath(schema_path, toolpack_dir)},
        "outputSchema": {"$ref": os.path.relpath(schema_path, toolpack_dir)},
    }
    if overrides:
        base.update(overrides)
    return base


def test_version_must_follow_semver(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "io.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={"version": "2024.1"},
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_python_execution_requires_entrypoint(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={"execution": {"kind": "python"}},
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_cli_execution_requires_string_list(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "cli.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={"execution": {"kind": "cli", "cmd": "echo"}},
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_env_passthrough_requires_uppercase_names(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "env.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={
            "env": {"passthrough": ["GOOD", "bad"], "set": {"ANOTHER": "value"}},
        },
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_caps_network_requires_known_protocol(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "caps.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={"caps": {"network": ["ftp"]}},
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_templating_engine_must_be_supported(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "templating.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    invalid = _build_valid_toolpack(
        schema_path=schema_path,
        toolpack_dir=tmp_path,
        overrides={"templating": {"engine": "liquid"}},
    )
    _write_toolpack(tmp_path, "invalid.tool.yaml", invalid)

    loader = ToolpackLoader()
    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)
