# tests/unit/mcp/test_vector_query_stub.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.toolpacks.python.core import vector_query_search

INDEX_FIXTURE = Path("tests/fixtures/mcp/vector_query_stub_index.json")


@pytest.fixture(scope="module")
def index_documents() -> list[dict[str, object]]:
    with INDEX_FIXTURE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["documents"]


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_vector_query_stub_returns_top_k_sorted(index_documents, top_k: int) -> None:
    response = vector_query_search.run({"query": "retrieval augmented generation", "top_k": top_k})
    assert "hits" in response and isinstance(response["hits"], list)
    assert len(response["hits"]) == top_k

    expected_ids = [
        doc["id"]
        for doc in sorted(index_documents, key=lambda doc: doc["score"], reverse=True)[:top_k]
    ]
    assert [hit["id"] for hit in response["hits"]] == expected_ids
    for hit in response["hits"]:
        assert "score" in hit and "metadata" in hit
        assert isinstance(hit["metadata"], dict)


def test_vector_query_stub_validates_top_k_positive() -> None:
    with pytest.raises(ValueError):
        vector_query_search.run({"query": "bad request", "top_k": 0})
