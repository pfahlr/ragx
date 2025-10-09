from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path("apps/mcp_server/schemas/tools")
TOOLPACK_DIR = Path("apps/mcp_server/toolpacks/core")

SCHEMA_VARIANTS = [
    ("exports_render_markdown", "input"),
    ("exports_render_markdown", "output"),
    ("vector_query_search", "input"),
    ("vector_query_search", "output"),
    ("docs_load_fetch", "input"),
    ("docs_load_fetch", "output"),
]

TOOLPACK_IDS = {
    "exports.render.markdown": "apps.toolpacks.python.core.exports.render_markdown:run",
    "vector.query.search": "apps.toolpacks.python.core.vector.query_search:run",
    "docs.load.fetch": "apps.toolpacks.python.core.docs.load_fetch:run",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "id",
    "name",
    "version",
    "description",
    "deterministic",
    "timeoutMs",
    "limits",
    "inputSchema",
    "outputSchema",
    "execution",
}


@pytest.mark.parametrize(("stem", "io_kind"), SCHEMA_VARIANTS)
def test_schema_files_exist_and_are_valid(stem: str, io_kind: str) -> None:
    schema_path = SCHEMA_DIR / f"{stem}.{io_kind}.schema.json"
    assert schema_path.exists(), f"Missing schema: {schema_path}"

    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    Draft202012Validator.check_schema(schema)
    assert schema.get("$schema", "").endswith("2020-12/schema")
    assert schema.get("type") == "object", "Schemas must validate objects"
    assert isinstance(schema.get("properties"), dict), "Schema must declare properties"

    for key in schema["properties"].keys():
        assert "_" not in key, f"Schema property '{key}' must use camelCase"
        assert key[0].islower(), f"Schema property '{key}' should start lower-case"

    required = schema.get("required", [])
    assert isinstance(required, list), "required must be a list"
    for item in required:
        assert item in schema["properties"], f"Required key '{item}' missing from properties"


@pytest.mark.parametrize("tool_suffix", TOOLPACK_IDS.keys())
def test_toolpack_definitions_reference_expected_modules(tool_suffix: str) -> None:
    filename = TOOLPACK_DIR / f"{tool_suffix}.tool.yaml"
    assert filename.exists(), f"Toolpack definition missing for {tool_suffix}"

    with filename.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in data, f"Toolpack {tool_suffix} missing key '{key}'"

    tool_id = f"mcp.tool:{tool_suffix}"
    assert data["id"] == tool_id
    version = str(data["version"])
    assert re.match(r"^\d+\.\d+\.\d+$", version), "version must follow semver"
    assert data["deterministic"] is True

    limits = data["limits"]
    assert limits["maxInputBytes"] >= 4096
    assert limits["maxOutputBytes"] >= 4096

    execution = data["execution"]
    assert execution["kind"] == "python"
    assert execution["module"] == TOOLPACK_IDS[tool_suffix]

    for key, suffix in (("inputSchema", "input"), ("outputSchema", "output")):
        assert "$ref" in data[key]
        resolved = (filename.parent / data[key]["$ref"]).resolve()
        assert resolved.exists(), f"Schema ref {data[key]['$ref']} missing"
        assert resolved.name == f"{tool_suffix.replace('.', '_')}.{suffix}.schema.json"

    metadata = data.get("metadata", {})
    assert metadata.get("owner") == "core"
    assert "tags" in metadata and "core-tools" in metadata["tags"]


def test_vector_query_search_output_schema_documents_hits_structure() -> None:
    path = SCHEMA_DIR / "vector_query_search.output.schema.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    hits = payload["properties"]["hits"]
    assert hits["type"] == "array"
    hit_item = hits["items"]
    required_keys = {"id", "score", "document"}
    assert required_keys.issubset(set(hit_item["required"]))
    doc_props = hit_item["properties"]["document"]["properties"]
    assert {"title", "snippet", "sourcePath"}.issubset(doc_props.keys())


def test_docs_load_fetch_input_schema_requires_paths() -> None:
    schema_path = SCHEMA_DIR / "docs_load_fetch.input.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "path" in schema["required"]
    path_schema = schema["properties"]["path"]
    assert path_schema["type"] == "string"
    assert path_schema.get("format") == "uri-reference"
    metadata_path = schema["properties"]["metadataPath"]
    assert metadata_path.get("type") == "string"
