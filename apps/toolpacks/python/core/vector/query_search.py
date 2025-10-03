"""Deterministic stub implementation for vector.query.search."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from apps.toolpacks.python.core.vector.state import SAMPLE_INDEX


def query_search(payload: Mapping[str, Any]) -> dict[str, Any]:
    query = str(payload["query"]).lower()
    top_k = int(payload.get("top_k", 3))
    include_metadata = bool(payload.get("include_metadata", True))

    hits = []
    for entry in SAMPLE_INDEX:
        score = entry["score"]
        if query and query not in entry["document_id"] and query not in (
            entry["metadata"]["summary"].lower()
        ):
            score *= 0.5
        metadata = entry["metadata"] if include_metadata else {}
        hits.append(
            {
                "document_id": entry["document_id"],
                "score": score,
                "metadata": metadata,
            }
        )
    return {"hits": hits[:top_k]}
