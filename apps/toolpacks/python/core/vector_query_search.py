from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .vector.query_search import query_search


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Backward compatible wrapper used by legacy tests."""

    result = query_search(payload)
    top_k = int(payload.get("top_k", len(result["hits"])))
    return {
        "hits": result["hits"][:top_k],
        "meta": result["meta"],
    }
