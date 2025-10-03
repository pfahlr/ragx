"""Legacy schema regression tests aligned with 06ab task spec."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from packaging.version import Version

SCHEMA_DIR = Path("apps/mcp_server/schemas/tools")
TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")

SCHEMA_FILES = [
    ("exports_render_markdown.input.schema.json"),
    ("exports_render_markdown.output.schema.json"),
    ("vector_query_search.input.schema.json"),
    ("vector_query_search.output.schema.json"),
    ("docs_load_fetch.input.schema.json"),
    ("docs_load_fetch.output.schema.json"),
]

TOOLPACK_FILES = {
    "exports.render.markdown": "exports.render.markdown.tool.yaml",
    "vector.query.search": "vector.query.search.tool.yaml",
    "docs.load.fetch": "docs.load.fetch.tool.yaml",
}


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("filename", SCHEMA_FILES)
def test_schema_files_exist_and_validate(filename: str) -> None:
    schema_path = SCHEMA_DIR / filename
    assert schema_path.exists(), f"Missing schema: {schema_path}"
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    assert schema.get("$schema", "").endswith("draft/2020-12/schema")
    assert schema.get("type") == "object"
    assert "$id" in schema


@pytest.mark.parametrize(("tool_id", "filename"), TOOLPACK_FILES.items())
def test_toolpacks_reference_schemas_and_metadata(tool_id: str, filename: str) -> None:
    toolpack_path = TOOLPACK_DIR / filename
    assert toolpack_path.exists(), f"Missing toolpack definition: {toolpack_path}"
    data = _load_yaml(toolpack_path)

    assert data["id"] == tool_id
    Version(str(data["version"]))
    assert data["deterministic"] is True
    assert isinstance(data["timeoutMs"], int) and data["timeoutMs"] > 0

    limits = data["limits"]
    for key in ("maxInputBytes", "maxOutputBytes"):
        assert isinstance(limits[key], int) and limits[key] > 0

    execution = data["execution"]
    assert execution["kind"] == "python"
    module_path = execution["module"]
    assert module_path.startswith("apps.toolpacks.python.core.")
    assert ":" in module_path

    for key in ("inputSchema", "outputSchema"):
        schema_ref = data[key]
        assert isinstance(schema_ref, dict) and "$ref" in schema_ref
        resolved = (toolpack_path.parent / schema_ref["$ref"]).resolve()
        assert resolved.exists(), f"Schema ref {schema_ref['$ref']} not found for {tool_id}"
