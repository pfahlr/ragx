from __future__ import annotations

import json
from pathlib import Path

import json

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_DIR = Path("apps/mcp_server/schemas/tools")

INPUT_SCHEMAS = {
    "mcp.tool:exports.render.markdown": "exports_render_markdown.input.schema.json",
    "mcp.tool:vector.query.search": "vector_query_search.input.schema.json",
    "mcp.tool:docs.load.fetch": "docs_load_fetch.input.schema.json",
}

OUTPUT_SCHEMAS = {
    "mcp.tool:exports.render.markdown": "exports_render_markdown.output.schema.json",
    "mcp.tool:vector.query.search": "vector_query_search.output.schema.json",
    "mcp.tool:docs.load.fetch": "docs_load_fetch.output.schema.json",
}


@pytest.mark.parametrize("schema_name", INPUT_SCHEMAS.values())
def test_core_tools_minimal_input_schema_exists(schema_name: str) -> None:
    schema_path = SCHEMA_DIR / schema_name
    assert schema_path.exists(), f"Expected schema at {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema.get("$schema", "").endswith("2020-12/schema"), "Schema must use Draft2020-12"
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("schema_name", OUTPUT_SCHEMAS.values())
def test_core_tools_minimal_output_schema_exists(schema_name: str) -> None:
    schema_path = SCHEMA_DIR / schema_name
    assert schema_path.exists(), f"Expected schema at {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema.get("$schema", "").endswith("2020-12/schema"), "Schema must use Draft2020-12"
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    ("tool_id", "payload"),
    [
        (
            "mcp.tool:exports.render.markdown",
            {
                "title": "Demo",
                "template": "# {{ title }}",
                "body": "sample body",
                "front_matter": {"authors": ["RAGX"]},
            },
        ),
        (
            "mcp.tool:docs.load.fetch",
            {
                "path": "tests/fixtures/mcp/core_tools/docs/example.md",
                "metadata_path": "tests/fixtures/mcp/core_tools/docs/example.json",
            },
        ),
        (
            "mcp.tool:vector.query.search",
            {
                "query": "vector search",
                "top_k": 2,
            },
        ),
    ],
)
def test_core_tools_minimal_input_schema_accepts_valid_payload(tool_id: str, payload: dict[str, object]) -> None:
    schema_name = INPUT_SCHEMAS[tool_id]
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    ("tool_id", "payload", "missing_field"),
    [
        (
            "mcp.tool:exports.render.markdown",
            {
                "template": "# {{ title }}",
                "body": "sample body",
            },
            "title",
        ),
        (
            "mcp.tool:docs.load.fetch",
            {
                "metadata_path": "tests/fixtures/mcp/core_tools/docs/example.json",
            },
            "path",
        ),
        (
            "mcp.tool:vector.query.search",
            {
                "top_k": 3,
            },
            "query",
        ),
    ],
)
def test_core_tools_minimal_input_schema_rejects_missing_required_field(
    tool_id: str, payload: dict[str, object], missing_field: str
) -> None:
    schema_name = INPUT_SCHEMAS[tool_id]
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    with pytest.raises(ValidationError):
        validator.validate(payload)
