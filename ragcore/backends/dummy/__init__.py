"""Pure-Python dummy backend used for tests and local smoke checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragcore.interfaces import IndexSpec, VectorIndexHandle


class DummyBackend:
    """Backend that stores vectors in-memory using :class:`VectorIndexHandle`."""

    name = "dummy"

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": False,
            "kinds": {
                "flat": {"requires_training": False},
            },
            "metrics": ["l2", "ip", "cosine"],
            "description": "Pure Python backend for smoke tests.",
        }

    def build(self, spec: Mapping[str, Any]) -> VectorIndexHandle:
        parsed = IndexSpec.from_mapping(spec, default_backend=self.name)
        if parsed.backend != self.name:
            raise ValueError(f"dummy backend cannot build for '{parsed.backend}'")
        return DummyHandle(parsed)


class DummyHandle(VectorIndexHandle):
    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=False, supports_gpu=False)
        self._factory_kwargs = {}


__all__ = ["DummyBackend", "DummyHandle"]

