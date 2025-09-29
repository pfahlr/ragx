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


def _make_toolpack_data(
    tool_dir: Path,
    schema_path: Path,
    *,
    execution: dict,
    tool_id: str = "sample.tool",
    version: str = "1.0.0",
) -> dict:
    rel_schema = os.path.relpath(schema_path, tool_dir)
    return {
        "id": tool_id,
        "version": version,
        "deterministic": True,
        "timeoutMs": 1000,
        "limits": {"maxInputBytes": 1, "maxOutputBytes": 1},
        "inputSchema": {"$ref": rel_schema},
        "outputSchema": {"$ref": rel_schema},
        "execution": execution,
    }


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


def test_cli_execution_requires_command_list(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpack_data = _make_toolpack_data(
        tmp_path,
        schema_path,
        execution={"kind": "cli"},
        tool_id="cli.tool",
    )

    _write_toolpack(tmp_path, "cli.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_python_module_entrypoint_requires_callable(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpack_data = _make_toolpack_data(
        tmp_path,
        schema_path,
        execution={"kind": "python", "module": "pkg.module"},
        tool_id="python.tool",
    )

    _write_toolpack(tmp_path, "python.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_http_execution_requires_url(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    toolpack_data = _make_toolpack_data(
        tmp_path,
        schema_path,
        execution={"kind": "http", "method": "POST"},
        tool_id="http.tool",
    )

    _write_toolpack(tmp_path, "http.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_env_passthrough_requires_strings(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    execution = {"kind": "python", "module": "pkg.tool:run"}
    toolpack_data = _make_toolpack_data(
        tmp_path,
        schema_path,
        execution=execution,
        tool_id="env.tool",
    )
    toolpack_data["env"] = {"passthrough": ["TOKEN", 123]}

    _write_toolpack(tmp_path, "env.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_caps_network_requires_strings(tmp_path: Path, schema_dir: Path) -> None:
    schema_path = _write_schema(
        schema_dir,
        "schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        },
    )

    execution = {"kind": "python", "module": "pkg.tool:run"}
    toolpack_data = _make_toolpack_data(
        tmp_path,
        schema_path,
        execution=execution,
        tool_id="caps.tool",
    )
    toolpack_data["caps"] = {"network": ["https", 5]}

    _write_toolpack(tmp_path, "caps.tool.yaml", toolpack_data)

    loader = ToolpackLoader()

    with pytest.raises(ToolpackValidationError):
        loader.load_dir(tmp_path)


def test_valid_optional_sections_are_preserved(tmp_path: Path, schema_dir: Path) -> None:
    input_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    output_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"results": {"type": "array", "items": {"type": "string"}}},
        "required": ["results"],
    }

    input_path = _write_schema(schema_dir, "input.schema.json", input_schema)
    output_path = _write_schema(schema_dir, "output.schema.json", output_schema)

    toolpack_dir = tmp_path / "packs"
    execution = {
        "kind": "http",
        "url": "https://example.com/run",
        "method": "POST",
        "headers": {"Authorization": "Bearer token"},
    }
    toolpack_data = {
        "id": "search.http",
        "version": "2.0.0",
        "deterministic": True,
        "timeoutMs": 5000,
        "limits": {"maxInputBytes": 4096, "maxOutputBytes": 8192},
        "inputSchema": {"$ref": os.path.relpath(input_path, toolpack_dir)},
        "outputSchema": {"$ref": os.path.relpath(output_path, toolpack_dir)},
        "execution": execution,
        "caps": {"network": ["https"]},
        "env": {"passthrough": ["TOKEN"]},
        "templating": {"engine": "jinja2", "cacheKey": "{{id}}-{{version}}"},
    }

    _write_toolpack(toolpack_dir, "search.http.tool.yaml", toolpack_data)

    loader = ToolpackLoader()
    loader.load_dir(tmp_path)

    pack = loader.get("search.http")
    assert pack.caps["network"] == ["https"]
    assert pack.env["passthrough"] == ["TOKEN"]
    assert pack.templating["engine"] == "jinja2"
    assert pack.execution["method"] == "POST"
    assert pack.execution["headers"] == {"Authorization": "Bearer token"}
