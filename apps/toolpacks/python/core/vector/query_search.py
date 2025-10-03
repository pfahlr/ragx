from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_HITS = [
    {
        "id": "example-doc-1",
        "snippet": "RAGX deterministic vector search stub.",
        "metadata": {"source": "fixture", "chunk": 1},
    },
    {
        "id": "example-doc-2",
        "snippet": "Demonstrates retrieval scoring for tests.",
        "metadata": {"source": "fixture", "chunk": 2},
    },
    {
        "id": "example-doc-3",
        "snippet": "Vector hit shaped according to schema requirements.",
        "metadata": {"source": "fixture", "chunk": 3},
    },
]


def query_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic vector hits for testing the MCP runtime."""

    query = str(payload.get("query", ""))
    top_k = max(1, int(payload.get("top_k", 3)))

    hits: list[dict[str, Any]] = []
    for index, base in enumerate(_HITS[:top_k]):
        score = max(0.0, round(1.0 - index * 0.1, 3))
        hit = dict(base)
        hit["score"] = score
        hits.append(hit)

    return {
        "hits": hits,
        "meta": {
            "provider": "stub",
            "query": query,
            "top_k": top_k,
        },
    }
