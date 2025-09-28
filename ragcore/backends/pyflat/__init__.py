"""Pure-Python flat (brute-force) vector index backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragcore.interfaces import IndexSpec, SerializedIndex, VectorIndexHandle


class PyFlatBackend:
    """Backend that performs brute-force search in Python/NumPy."""

    name = "py_flat"

    _SUPPORTED_KINDS = {"flat"}
    _METRICS = {"ip", "l2"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": False,
            "kinds": {
                "flat": {
                    "requires_training": False,
                }
            },
            "metrics": sorted(self._METRICS),
        }

    def build(self, spec: Mapping[str, Any]) -> PyFlatHandle:
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.backend != self.name:
            raise ValueError(f"py_flat backend cannot build for '{config.backend}'")
        if config.kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"unsupported py_flat kind '{config.kind}'")
        if config.metric not in self._METRICS:
            raise ValueError(f"unsupported py_flat metric '{config.metric}'")
        return PyFlatHandle(config)


class PyFlatHandle(VectorIndexHandle):
    """Concrete handle for the Python flat backend."""

    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=False, supports_gpu=False)

    @classmethod
    def from_serialized(cls, payload: SerializedIndex) -> PyFlatHandle:
        """Reconstruct a handle instance from :meth:`serialize_cpu` output."""

        spec = IndexSpec.from_mapping(payload.spec, default_backend=PyFlatBackend.name)
        if spec.backend != PyFlatBackend.name:
            raise ValueError("serialized payload was not produced by py_flat backend")

        handle = cls(spec)
        if payload.vectors.size:
            handle.add(payload.vectors, ids=payload.ids)
        handle._is_trained = payload.is_trained  # type: ignore[attr-defined]
        handle.is_gpu = payload.is_gpu
        return handle


__all__ = ["PyFlatBackend", "PyFlatHandle"]

