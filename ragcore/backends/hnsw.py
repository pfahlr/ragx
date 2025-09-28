from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import IndexSpec, VectorIndexHandle


class HnswBackend:
    """Simulated HNSW backend using the shared vector handle."""

    name = "hnsw"

    _SUPPORTED_KINDS = {"hnsw"}
    _METRICS = {"l2", "ip", "cosine"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": False,
            "kinds": {
                "hnsw": {"requires_training": False, "params": ["m", "ef", "ef_construction"]},
            },
            "metrics": sorted(self._METRICS),
        }

    def build(self, spec: Mapping[str, Any]) -> VectorIndexHandle:
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.backend != self.name:
            raise ValueError(f"HNSW backend cannot build for '{config.backend}'")
        if config.kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"unsupported HNSW kind '{config.kind}'")
        if config.metric not in self._METRICS:
            raise ValueError(f"unsupported HNSW metric '{config.metric}'")
        return HnswHandle(config)


class HnswHandle(VectorIndexHandle):
    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=False, supports_gpu=False)
        self._factory_kwargs = {}


__all__ = ["HnswBackend", "HnswHandle"]

