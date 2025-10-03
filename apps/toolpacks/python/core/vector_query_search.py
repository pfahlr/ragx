from __future__ import annotations

from typing import Any, Mapping

from .vector.query_search import query_search


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility shim mapping legacy entrypoint to new implementation."""

    return query_search(payload)
