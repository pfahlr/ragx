"""Deterministic stub for vector.query.search."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

_CORPUS = [
    {
        "id": "sample-article",
        "chunk": "RAGX builds deterministic retrieval pipelines for testing.",
        "metadata": {
            "path": "tests/fixtures/mcp/docs/sample_article.md",
            "source": "local",
            "title": "Sample Article",
        },
    },
    {
        "id": "spec-overview",
        "chunk": "The master specification defines schemas, logging, and observability.",
        "metadata": {
            "path": "codex/specs/ragx_master_spec.yaml",
            "source": "spec",
            "title": "RAGX Master Spec",
        },
    },
    {
        "id": "tooling-notes",
        "chunk": "Toolpacks run deterministically and validate input schemas.",
        "metadata": {
            "path": "codex/agents/TASKS/06ab_core_tools_minimal_subset.yaml",
            "source": "tasks",
            "title": "Core Tools Minimal Subset",
        },
    },
]


def _score(query: str, doc_id: str) -> float:
    digest = hashlib.sha256(f"{query}|{doc_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return round(value, 6)


def query_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic search hits for ``query``."""

    query = str(payload["query"]).strip()
    if not query:
        raise ValueError("query must be a non-empty string")

    top_k_raw = payload.get("top_k", 5)
    try:
        top_k = max(1, min(int(top_k_raw), len(_CORPUS)))
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("top_k must be an integer") from exc

    hits = []
    for entry in _CORPUS:
        score = _score(query, entry["id"])
        hits.append(
            {
                "id": entry["id"],
                "score": score,
                "chunk": entry["chunk"],
                "metadata": dict(entry["metadata"]),
            }
        )

    hits.sort(key=lambda item: item["score"], reverse=True)
    return {
        "hits": hits[:top_k],
        "metadata": {
            "query": query,
            "top_k": top_k,
            "corpus_size": len(_CORPUS),
        },
    }


__all__ = ["query_search"]

