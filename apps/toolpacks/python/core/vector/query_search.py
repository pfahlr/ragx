from __future__ import annotations

import math
import re
from typing import Any, Mapping

_CORPUS = [
    {
        "id": "doc:example",
        "title": "Example Document",
        "content": (
            "Example Document\n\n"
            "This is a deterministic fixture used to validate the minimal core tool runtime.\n\n"
            "It contains two paragraphs to provide enough text for search scoring."
        ),
        "metadata": {"source": "fixtures", "position": 1},
    },
    {
        "id": "doc:tutorial",
        "title": "Tool Runtime Tutorial",
        "content": (
            "Tool Runtime Tutorial\n\n"
            "Core tools expose deterministic behaviours for rendering markdown, loading documents, and performing simple search.\n\n"
            "This stub ensures the runtime can score overlap between a query and known corpus entries."
        ),
        "metadata": {"source": "fixtures", "position": 2},
    },
]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenise(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)}


def _score(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & document_tokens)
    if overlap == 0:
        return 0.0
    normaliser = math.sqrt(len(document_tokens)) or 1.0
    return round(overlap / (len(query_tokens) * normaliser), 6)


def query_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    query = str(payload["query"]).strip()
    top_k = int(payload.get("top_k", 3))
    top_k = max(1, min(top_k, 5))

    query_tokens = _tokenise(query)
    hits: list[dict[str, Any]] = []
    for entry in _CORPUS:
        document_tokens = _tokenise(entry["content"])
        score = _score(query_tokens, document_tokens)
        snippet = entry["content"].splitlines()[1] if "\n" in entry["content"] else entry["content"]
        hits.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "score": score,
                "snippet": snippet[:240],
                "metadata": dict(entry["metadata"]),
            }
        )

    hits.sort(key=lambda item: (item["score"], item["id"]), reverse=True)
    sliced = hits[:top_k]
    return {
        "hits": sliced,
        "elapsed_ms": 0.12,
        "total_hits": len(hits),
    }
