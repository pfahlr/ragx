"""Canonical Python interfaces for RAGX vector database backends."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, TypeAlias, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]


class SearchResults(TypedDict):
    """Typed result mapping returned by :meth:`Handle.search`."""

    ids: IntArray
    distances: FloatArray


@runtime_checkable
class Backend(Protocol):
    """Protocol implemented by vector database backend factories."""

    name: str

    def capabilities(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of backend capabilities."""

    def build(self, spec: Mapping[str, Any]) -> Handle:
        """Construct a handle implementing :class:`Handle` for the given spec."""


@runtime_checkable
class Handle(Protocol):
    """Protocol describing the concrete vector index handle API."""

    is_gpu: bool
    device: str | None

    def requires_training(self) -> bool: ...

    def train(self, vectors: FloatArray) -> None: ...

    def add(self, vectors: FloatArray, ids: IntArray | None = None) -> None: ...

    def search(
        self,
        queries: FloatArray,
        k: int,
        **kwargs: Any,
    ) -> SearchResults: ...

    def ntotal(self) -> int: ...

    def serialize_cpu(self) -> SerializedIndex: ...

    def to_gpu(self, device: str | None = None) -> Handle: ...

    def merge_with(self, other: Handle) -> Handle: ...

    def spec(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class IndexSpec:
    """Parsed configuration for an index."""

    backend: str
    kind: str
    metric: str
    dim: int
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        default_backend: str | None = None,
    ) -> IndexSpec:
        backend_value = mapping.get("backend", default_backend)
        if backend_value is None:
            raise ValueError("index spec must include a backend")
        backend = str(backend_value)

        kind_value = mapping.get("kind")
        metric_value = mapping.get("metric")
        dim_value = mapping.get("dim")

        if kind_value is None or metric_value is None or dim_value is None:
            raise ValueError("index spec missing required fields: kind, metric, dim")

        dim = int(dim_value)
        if dim <= 0:
            raise ValueError("dim must be a positive integer")

        params_value = mapping.get("params") or {}
        if not isinstance(params_value, Mapping):
            raise ValueError("params must be a mapping if provided")

        return cls(
            backend=backend,
            kind=str(kind_value),
            metric=str(metric_value),
            dim=dim,
            params=dict(params_value),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "kind": self.kind,
            "metric": self.metric,
            "dim": self.dim,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class SerializedIndex:
    """CPU-serialised payload for an index."""

    spec: Mapping[str, Any]
    vectors: FloatArray
    ids: IntArray
    metadata: Mapping[str, Any]
    is_trained: bool
    is_gpu: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the serialised index."""

        return {
            "spec": dict(self.spec),
            "vectors": self.vectors.tolist(),
            "ids": self.ids.tolist(),
            "metadata": dict(self.metadata),
            "is_trained": self.is_trained,
            "is_gpu": self.is_gpu,
        }


def _ensure_2d_float32(data: FloatArray, *, dim: int) -> FloatArray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("expected a 2D array of float32 vectors")
    if array.shape[1] != dim:
        raise ValueError(f"expected vectors with dimension {dim}, got {array.shape[1]}")
    return cast(FloatArray, array)


def _ensure_ids(ids: IntArray | None, *, count: int) -> IntArray | None:
    if ids is None:
        return None
    array = np.asarray(ids, dtype=np.int64)
    if array.ndim != 1 or array.shape[0] != count:
        raise ValueError("ids must be a 1D array matching the number of vectors")
    return cast(IntArray, array)


def _distance_matrix(vectors: FloatArray, queries: FloatArray, metric: str) -> FloatArray:
    if metric == "l2":
        diffs = queries[:, None, :] - vectors[None, :, :]
        return cast(FloatArray, np.sum(diffs * diffs, axis=2))
    if metric == "ip":
        sims = queries @ vectors.T
        return cast(FloatArray, -sims)
    if metric == "cosine":
        vectors_norm = np.linalg.norm(vectors, axis=1)
        queries_norm = np.linalg.norm(queries, axis=1)
        denom = np.clip(np.outer(queries_norm, vectors_norm), a_min=1e-12, a_max=None)
        cosine = (queries @ vectors.T) / denom
        return cast(FloatArray, 1.0 - cosine)
    raise ValueError(f"unsupported metric '{metric}'")


class VectorIndexHandle:
    """Concrete handle implementation shared by simulated backends."""

    def __init__(
        self,
        spec: IndexSpec,
        *,
        requires_training: bool,
        supports_gpu: bool = False,
    ) -> None:
        self._spec = spec
        self._requires_training = requires_training
        self._supports_gpu = supports_gpu
        self._factory_kwargs: dict[str, Any] = {
            "requires_training": requires_training,
            "supports_gpu": supports_gpu,
        }
        self._vectors = cast(FloatArray, np.empty((0, spec.dim), dtype=np.float32))
        self._ids = cast(IntArray, np.empty((0,), dtype=np.int64))
        self._next_id = 0
        self._is_trained = not requires_training
        self.is_gpu = False
        self.device: str | None = None

    def requires_training(self) -> bool:
        return self._requires_training

    def train(self, vectors: FloatArray) -> None:
        if not self._requires_training:
            self._is_trained = True
            return
        _ensure_2d_float32(vectors, dim=self._spec.dim)
        self._is_trained = True

    def add(self, vectors: FloatArray, ids: IntArray | None = None) -> None:
        if self._requires_training and not self._is_trained:
            raise RuntimeError("index requires training before adding vectors")
        batch = _ensure_2d_float32(vectors, dim=self._spec.dim)
        ids_array = _ensure_ids(ids, count=batch.shape[0])
        if ids_array is None:
            ids_array = cast(
                IntArray,
                np.arange(self._next_id, self._next_id + batch.shape[0], dtype=np.int64),
            )
        self._next_id = max(self._next_id, int(ids_array.max()) + 1)
        self._vectors = cast(FloatArray, np.concatenate([self._vectors, batch], axis=0))
        self._ids = cast(IntArray, np.concatenate([self._ids, ids_array], axis=0))

    def search(
        self,
        queries: FloatArray,
        k: int,
        **_: Any,
    ) -> SearchResults:
        if self._vectors.shape[0] == 0:
            raise RuntimeError("cannot search an empty index")
        query_array = _ensure_2d_float32(queries, dim=self._spec.dim)
        k = min(k, self._vectors.shape[0])
        distances = _distance_matrix(self._vectors, query_array, self._spec.metric)
        order = np.argsort(distances, axis=1)
        topk = order[:, :k]
        ids_matrix = cast(IntArray, np.tile(self._ids, (query_array.shape[0], 1)))
        top_distances = cast(FloatArray, np.take_along_axis(distances, topk, axis=1))
        top_ids = cast(IntArray, np.take_along_axis(ids_matrix, topk, axis=1))
        return {"ids": top_ids, "distances": top_distances}

    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def serialize_cpu(self) -> SerializedIndex:
        metadata: dict[str, Any] = {
            "requires_training": self._requires_training,
            "supports_gpu": self._supports_gpu,
        }
        return SerializedIndex(
            spec=self._spec.as_dict(),
            vectors=cast(FloatArray, self._vectors.copy()),
            ids=cast(IntArray, self._ids.copy()),
            metadata=metadata,
            is_trained=self._is_trained,
            is_gpu=self.is_gpu,
        )

    def to_gpu(self, device: str | None = None) -> VectorIndexHandle:
        clone = self._clone()
        if self._supports_gpu:
            clone.is_gpu = True
            clone.device = device or "cuda:0"
        return clone

    def merge_with(self, other: Handle) -> VectorIndexHandle:
        if not isinstance(other, VectorIndexHandle):
            raise TypeError("can only merge with another VectorIndexHandle")
        if other._spec != self._spec:
            raise ValueError("cannot merge indexes with different specs")
        merged = self._clone(empty=True)
        merged._vectors = cast(
            FloatArray,
            np.concatenate([self._vectors, other._vectors], axis=0),
        )
        merged._ids = cast(IntArray, np.arange(merged._vectors.shape[0], dtype=np.int64))
        merged._next_id = merged._ids.shape[0]
        merged._is_trained = self._is_trained or other._is_trained
        merged.is_gpu = self.is_gpu or other.is_gpu
        merged.device = self.device if self.is_gpu else other.device
        return merged

    def spec(self) -> Mapping[str, Any]:
        return self._spec.as_dict()

    def _clone(self, *, empty: bool = False) -> VectorIndexHandle:
        clone = self.__class__(self._spec, **self._factory_kwargs)
        clone._requires_training = self._requires_training
        clone._supports_gpu = self._supports_gpu
        clone._is_trained = self._is_trained
        if not empty:
            clone._vectors = cast(FloatArray, self._vectors.copy())
            clone._ids = cast(IntArray, self._ids.copy())
            clone._next_id = self._next_id
            clone.is_gpu = self.is_gpu
            clone.device = self.device
        return clone


def ensure_protocol_conformance(backends: Iterable[Backend]) -> None:
    """Utility to assert objects satisfy the backend protocol at runtime."""

    for backend in backends:
        if not isinstance(backend, Backend):  # pragma: no cover - defensive
            raise TypeError(f"{backend!r} does not satisfy Backend protocol")


__all__ = [
    "Backend",
    "Handle",
    "IndexSpec",
    "SerializedIndex",
    "VectorIndexHandle",
    "FloatArray",
    "IntArray",
    "SearchResults",
]

