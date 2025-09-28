from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import IndexSpec, VectorIndexHandle


class FaissBackend:
    """Simulated FAISS backend covering flat, IVF-Flat, and IVFPQ indexes."""

    name = "faiss"

    _SUPPORTED_KINDS = {"flat", "ivf_flat", "ivf_pq"}
    _METRICS = {"l2", "ip", "cosine"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": True,
            "kinds": {
                "flat": {"requires_training": False},
                "ivf_flat": {"requires_training": True, "params": ["nlist", "nprobe"]},
                "ivf_pq": {"requires_training": True, "params": ["nlist", "m", "nbits"]},
            },
            "metrics": sorted(self._METRICS),
        }

    def build(self, spec: Mapping[str, Any]) -> VectorIndexHandle:
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.backend != self.name:
            raise ValueError(f"FAISS backend cannot build for '{config.backend}'")
        if config.kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"unsupported FAISS kind '{config.kind}'")
        if config.metric not in self._METRICS:
            raise ValueError(f"unsupported FAISS metric '{config.metric}'")
        requires_training = config.kind in {"ivf_flat", "ivf_pq"}
        return FaissHandle(config, requires_training=requires_training)


class FaissHandle(VectorIndexHandle):
    """Python implementation mirroring the FAISS handle protocol."""

    def __init__(self, spec: IndexSpec, *, requires_training: bool) -> None:
        super().__init__(spec, requires_training=requires_training, supports_gpu=True)
        self._factory_kwargs = {"requires_training": requires_training}


__all__ = ["FaissBackend", "FaissHandle"]

