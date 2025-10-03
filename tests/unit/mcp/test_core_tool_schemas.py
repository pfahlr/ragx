"""Validates JSON schemas for core MCP tools using Draft2020-12."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_DIR = Path("apps/mcp_server/schemas/tools")

SCHEMAS = {
    "exports": (
        SCHEMA_DIR / "exports_render_markdown.input.schema.json",
        SCHEMA_DIR / "exports_render_markdown.output.schema.json",
    ),
    "vector": (
        SCHEMA_DIR / "vector_query_search.input.schema.json",
        SCHEMA_DIR / "vector_query_search.output.schema.json",
    ),
    "docs": (
        SCHEMA_DIR / "docs_load_fetch.input.schema.json",
        SCHEMA_DIR / "docs_load_fetch.output.schema.json",
    ),
}


@pytest.mark.parametrize("tool_id", sorted(SCHEMAS))
def test_schema_files_exist(tool_id: str) -> None:
    input_schema, output_schema = SCHEMAS[tool_id]
    assert input_schema.exists(), f"Missing schema file {input_schema}"
    assert output_schema.exists(), f"Missing schema file {output_schema}"


@pytest.mark.parametrize(
    ("schema_path", "payload"),
    [
        (
            SCHEMAS["exports"][0],
            {
                "title": "Sample",
                "template": "# {{ title }}",
                "body": "Hello",
                "front_matter": {"tags": ["demo"]},
            },
        ),
        (
            SCHEMAS["exports"][1],
            {"markdown": "# Sample", "content_hash": "abc123"},
        ),
        (
            SCHEMAS["vector"][0],
            {"query": "demo", "top_k": 3},
        ),
        (
            SCHEMAS["vector"][1],
            {"hits": [{"document_id": "doc", "score": 1.0, "metadata": {}}]},
        ),
        (
            SCHEMAS["docs"][0],
            {"path": "docs/example.md", "metadata_path": "docs/example.json"},
        ),
        (
            SCHEMAS["docs"][1],
            {"document": "Body", "metadata": {"title": "Demo"}, "checksum": "abc"},
        ),
    ],
)
def test_schemas_accept_valid_payload(schema_path: Path, payload: dict[str, object]) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    ("schema_path", "payload"),
    [
        (
            SCHEMAS["exports"][0],
            {"title": "Sample"},
        ),
        (
            SCHEMAS["vector"][0],
            {"top_k": 3},
        ),
        (
            SCHEMAS["docs"][0],
            {"metadata_path": "docs/example.json"},
        ),
    ],
)
def test_schemas_reject_missing_required_fields(
    schema_path: Path,
    payload: dict[str, object],
) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    with pytest.raises(ValidationError):
        validator.validate(payload)
