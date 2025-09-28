from __future__ import annotations

from typing import Any, Mapping

from .base import IndexSpec, VectorIndexHandle


class CuVSBackend:
    """Simulated RAFT/cuVS backend with GPU-aware handle."""

    name = "cuvs"

    _SUPPORTED_KINDS = {"ivf_flat", "ivf_pq"}
    _METRICS = {"l2", "ip", "cosine"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": True,
            "kinds": {
                "ivf_flat": {"requires_training": True, "params": ["nlist", "nprobe"]},
                "ivf_pq": {"requires_training": True, "params": ["nlist", "m", "nbits"]},
            },
            "metrics": sorted(self._METRICS),
        }

    def build(self, spec: Mapping[str, Any]) -> VectorIndexHandle:
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.backend != self.name:
            raise ValueError(f"cuVS backend cannot build for '{config.backend}'")
        if config.kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"unsupported cuVS kind '{config.kind}'")
        if config.metric not in self._METRICS:
            raise ValueError(f"unsupported cuVS metric '{config.metric}'")
        return CuVSHandle(config)


class CuVSHandle(VectorIndexHandle):
    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=True, supports_gpu=True)


__all__ = ["CuVSBackend", "CuVSHandle"]

