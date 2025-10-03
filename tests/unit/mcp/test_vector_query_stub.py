"""Unit tests for the deterministic vector.query.search stub."""
from __future__ import annotations

import pytest

from apps.toolpacks.python.core.vector import query_search
from apps.toolpacks.python.core.vector.state import SAMPLE_INDEX


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_query_search_respects_top_k(top_k: int) -> None:
    response = query_search.query_search({"query": "retrieval", "top_k": top_k})
    assert "hits" in response
    hits = response["hits"]
    assert len(hits) == top_k
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_query_search_includes_metadata_when_requested() -> None:
    response = query_search.query_search(
        {"query": "resilience", "include_metadata": True, "top_k": 2}
    )
    assert all("metadata" in hit and hit["metadata"] for hit in response["hits"])


def test_query_search_omits_metadata_when_disabled() -> None:
    response = query_search.query_search(
        {"query": "resilience", "include_metadata": False, "top_k": 2}
    )
    assert all(hit["metadata"] == {} for hit in response["hits"])


def test_query_search_handles_unknown_terms() -> None:
    response = query_search.query_search({"query": "unknown-term", "top_k": 3})
    assert len(response["hits"]) == 3
    assert {hit["document_id"] for hit in response["hits"]} == {
        entry["document_id"] for entry in SAMPLE_INDEX
    }
