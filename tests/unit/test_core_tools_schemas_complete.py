from __future__ import annotations

import json
from pathlib import Path

SCHEMA_ROOT = Path("apps/mcp_server/schemas/tools")


def _load(name: str) -> dict[str, object]:
    path = SCHEMA_ROOT / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_exports_render_markdown_schema_details() -> None:
    schema = _load("exports_render_markdown.output.schema.json")
    properties = schema["properties"]
    assert properties["markdown"]["type"] == "string"
    assert properties["metadata"]["properties"]["content_type"]["const"] == "text/markdown"
    required = set(schema["required"])
    assert {"markdown", "content_hash", "metadata"}.issubset(required)


def test_vector_query_search_hit_shape() -> None:
    schema = _load("vector_query_search.output.schema.json")
    hit_schema = schema["properties"]["hits"]["items"]
    assert hit_schema["properties"]["score"]["minimum"] == 0
    assert hit_schema["properties"]["score"]["maximum"] == 1
    assert "snippet" in hit_schema["required"]
    assert hit_schema["properties"]["metadata"]["type"] == "object"


def test_docs_load_fetch_schema_details() -> None:
    input_schema = _load("docs_load_fetch.input.schema.json")
    assert input_schema["properties"]["path"]["format"] == "path"
    assert "metadata_path" in input_schema["required"]

    output_schema = _load("docs_load_fetch.output.schema.json")
    doc_props = output_schema["properties"]["document"]["properties"]
    assert doc_props["contents"]["type"] == "string"
    meta_required = set(output_schema["properties"]["metadata"]["required"])
    assert {"title", "authors"}.issubset(meta_required)
