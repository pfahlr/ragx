from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from ragcore.interfaces import (
    FloatArray,
    IndexSpec,
    IntArray,
    SerializedIndex,
    VectorIndexHandle,
)


class PyFlatBackend:
    """Pure Python flat index backend with L2 and inner-product metrics."""

    name = "py_flat"
    _SUPPORTED_KIND = "flat"
    _SUPPORTED_METRICS = {"l2", "ip"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": False,
            "kinds": {self._SUPPORTED_KIND: {"requires_training": False}},
            "metrics": sorted(self._SUPPORTED_METRICS),
            "description": "Pure Python reference implementation for flat indexes.",
        }

    def build(self, spec: Mapping[str, Any]) -> PyFlatHandle:
        parsed = IndexSpec.from_mapping(spec, default_backend=self.name)
        if parsed.backend != self.name:
            raise ValueError(f"py_flat backend cannot build for '{parsed.backend}'")
        if parsed.kind != self._SUPPORTED_KIND:
            raise ValueError(f"unsupported py_flat kind '{parsed.kind}'")
        if parsed.metric not in self._SUPPORTED_METRICS:
            raise ValueError(f"unsupported py_flat metric '{parsed.metric}'")
        return PyFlatHandle(parsed)

    def deserialize_cpu(
        self, serialized: SerializedIndex | Mapping[str, Any]
    ) -> PyFlatHandle:
        payload = serialized
        if not isinstance(serialized, SerializedIndex):
            payload = _serialized_index_from_mapping(serialized)
        return PyFlatHandle.from_serialized(cast(SerializedIndex, payload))


class PyFlatHandle(VectorIndexHandle):
    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=False, supports_gpu=False)
        self._factory_kwargs = {}

    @classmethod
    def from_serialized(cls, serialized: SerializedIndex) -> PyFlatHandle:
        spec = IndexSpec.from_mapping(serialized.spec, default_backend=PyFlatBackend.name)
        handle = cls(spec)

        vectors = np.asarray(serialized.vectors, dtype=np.float32)
        if vectors.ndim == 1:
            if vectors.size != 0:
                raise ValueError("serialised vectors must be 2D")
            vectors = np.empty((0, spec.dim), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("serialised vectors must be 2D")
        if vectors.shape[1] != spec.dim:
            raise ValueError(
                "serialised vectors dimension "
                f"{vectors.shape[1]} does not match spec dim {spec.dim}"
            )

        ids = np.asarray(serialized.ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("serialised ids must be a 1D array")
        if ids.shape[0] != vectors.shape[0]:
            raise ValueError("serialised ids must align with vector rows")

        handle._vectors = cast(FloatArray, vectors.copy())
        handle._ids = cast(IntArray, ids.copy())
        handle._next_id = int(ids.max()) + 1 if ids.size else 0
        handle._is_trained = bool(serialized.is_trained)
        handle.is_gpu = bool(serialized.is_gpu)
        handle.device = None
        return handle


def _serialized_index_from_mapping(data: Mapping[str, Any]) -> SerializedIndex:
    try:
        spec_mapping = data["spec"]
        vectors_value = data["vectors"]
        ids_value = data["ids"]
    except KeyError as exc:
        raise ValueError("serialised index mapping missing required keys") from exc

    if not isinstance(spec_mapping, Mapping):
        raise ValueError("serialised index spec must be a mapping")

    spec_dict = dict(spec_mapping)
    if "dim" not in spec_dict:
        raise ValueError("serialised index spec must include 'dim'")

    dim = int(spec_dict["dim"])
    if dim <= 0:
        raise ValueError("serialised index dim must be positive")

    vectors = np.asarray(vectors_value, dtype=np.float32)
    if vectors.ndim == 1:
        if vectors.size != 0:
            raise ValueError("serialised vectors must be 2D")
        vectors = np.empty((0, dim), dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("serialised vectors must be 2D")
    if vectors.shape[1] != dim:
        raise ValueError("serialised vectors dimension mismatch")

    ids = np.asarray(ids_value, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("serialised ids must be 1D")
    if ids.shape[0] != vectors.shape[0]:
        raise ValueError("serialised ids must align with vectors")

    metadata_value = data.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    is_trained = bool(data.get("is_trained", False))
    is_gpu = bool(data.get("is_gpu", False))

    return SerializedIndex(
        spec=spec_dict,
        vectors=cast(FloatArray, vectors),
        ids=cast(IntArray, ids),
        metadata=metadata,
        is_trained=is_trained,
        is_gpu=is_gpu,
    )


__all__ = ["PyFlatBackend", "PyFlatHandle"]
