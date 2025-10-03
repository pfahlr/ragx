from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

SCHEMA_ROOT = Path("apps/mcp_server/schemas/tools")


def _load_schema(name: str) -> tuple[dict[str, object], Draft202012Validator]:
    path = SCHEMA_ROOT / name
    assert path.exists(), f"Schema file missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    validator = Draft202012Validator(data)
    return data, validator


@pytest.mark.parametrize(
    "schema_name",
    [
        "exports_render_markdown.input.schema.json",
        "exports_render_markdown.output.schema.json",
        "vector_query_search.input.schema.json",
        "vector_query_search.output.schema.json",
        "docs_load_fetch.input.schema.json",
        "docs_load_fetch.output.schema.json",
    ],
)
def test_schema_metadata(schema_name: str) -> None:
    schema, _ = _load_schema(schema_name)
    assert "title" in schema
    assert schema.get("type") == "object"
    assert "properties" in schema


def test_exports_render_markdown_schema_roundtrip() -> None:
    _, input_validator = _load_schema("exports_render_markdown.input.schema.json")
    _, output_validator = _load_schema("exports_render_markdown.output.schema.json")

    valid_input = {
        "title": "Doc",
        "template": "# {{ title }}\n{{ body }}",
        "body": "content",
        "front_matter": {"authors": ["Tester"], "tags": ["demo"]},
    }
    input_validator.validate(valid_input)

    invalid_input = {"title": "Doc", "body": "missing template"}
    with pytest.raises(ValidationError):
        input_validator.validate(invalid_input)

    valid_output = {
        "markdown": "---\na: 1\n---\nbody",
        "content_hash": "a" * 64,
        "metadata": {"title": "Doc", "content_type": "text/markdown", "hash_algorithm": "sha256"},
    }
    output_validator.validate(valid_output)

    with pytest.raises(ValidationError):
        output_validator.validate({"markdown": "body"})


def test_vector_query_search_schema_validation() -> None:
    _, input_validator = _load_schema("vector_query_search.input.schema.json")
    _, output_validator = _load_schema("vector_query_search.output.schema.json")

    input_validator.validate({"query": "retrieval", "top_k": 2})
    with pytest.raises(ValidationError):
        input_validator.validate({"query": "test", "top_k": 0})

    valid_output = {
        "hits": [
            {
                "id": "doc-1",
                "score": 0.98,
                "snippet": "Example",
                "metadata": {"source": "stub"},
            }
        ],
        "meta": {"provider": "stub", "query": "retrieval", "top_k": 1},
    }
    output_validator.validate(valid_output)

    with pytest.raises(ValidationError):
        output_validator.validate({"hits": []})


def test_docs_load_fetch_schema_validation() -> None:
    _, input_validator = _load_schema("docs_load_fetch.input.schema.json")
    _, output_validator = _load_schema("docs_load_fetch.output.schema.json")

    valid_input = {
        "path": "tests/fixtures/mcp/core_tools/docs/example.md",
        "metadata_path": "tests/fixtures/mcp/core_tools/docs/example_metadata.json",
    }
    input_validator.validate(valid_input)

    with pytest.raises(ValidationError):
        input_validator.validate({})

    valid_output = {
        "document": {"path": "example.md", "contents": "hello"},
        "metadata": {"title": "Example", "authors": ["Tester"]},
    }
    output_validator.validate(valid_output)

    with pytest.raises(ValidationError):
        output_validator.validate({"document": {"path": "example.md"}})
