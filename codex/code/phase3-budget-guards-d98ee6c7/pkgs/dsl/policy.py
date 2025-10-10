"""Minimal PolicyStack implementation for sandbox runner tests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class PolicyStack:
    """Maintain a simple stack of node policies."""

    def __init__(self) -> None:
        self._stack: List[Tuple[str, Dict[str, Any]]] = []

    def push(self, node_id: str, metadata: Dict[str, Any]) -> None:
        self._stack.append((node_id, dict(metadata)))

    def resolve(self, node_id: str) -> None:
        if not self._stack or self._stack[-1][0] != node_id:
            raise ValueError(f"Policy resolve order mismatch for {node_id}")

    def pop(self, node_id: str) -> None:
        if not self._stack:
            raise ValueError("Policy stack empty")
        top_id, _ = self._stack.pop()
        if top_id != node_id:
            raise ValueError(f"Policy pop order mismatch: expected {top_id}, got {node_id}")
