from __future__ import annotations

from typing import Any

import pytest

from apps.toolpacks.python.core.vector import query_search


def _invoke(payload: dict[str, Any]) -> dict[str, Any]:
    return query_search.run(payload)


def test_query_search_limits_topk_to_document_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "41")
    total_documents = len(query_search._DOCUMENTS)

    result = _invoke({"query": "retrieval augmented generation", "topK": total_documents + 5})

    assert result["total"] == total_documents
    assert len(result["hits"]) == total_documents
    ranks = [hit["rank"] for hit in result["hits"]]
    assert ranks == list(range(1, total_documents + 1))


def test_query_search_enforces_minimum_topk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "57")
    response = _invoke({"query": "retrieval pipelines", "topK": 0})

    assert len(response["hits"]) == 1
    assert response["hits"][0]["rank"] == 1


def test_query_search_sorts_by_score_then_tie_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "91")
    payload = {"query": "retrieval quality", "topK": 2}

    first = _invoke(payload)
    second = _invoke(payload)

    assert first == second
    scores = [hit["score"] for hit in first["hits"]]
    assert scores == sorted(scores, reverse=True)
    expected_order = sorted(
        query_search._DOCUMENTS,
        key=lambda doc: (
            -query_search._score(payload["query"], doc),
            query_search._tie_breaker(doc.doc_id),
        ),
    )[:2]
    ids = [hit["id"] for hit in first["hits"]]
    assert ids == [doc.doc_id for doc in expected_order]


def test_query_search_prefers_matching_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "75")
    result = _invoke({"query": "demonstrates", "topK": 1})

    assert result["hits"], "Expected at least one hit"
    top_hit = result["hits"][0]
    assert top_hit["id"] == "example_fixture"
    assert top_hit["score"] >= 1.0


def test_query_search_includes_document_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGX_SEED", "100")
    result = _invoke({"query": "retrieval", "topK": 1})

    hit = result["hits"][0]
    document = hit["document"]
    assert document["title"]
    assert document["snippet"]
    assert document["sourcePath"].endswith(".md")
