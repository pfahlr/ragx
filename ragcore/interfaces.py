from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy import ndarray

FloatArray: TypeAlias = ndarray[Any, np.dtype[np.float32]]
IntArray: TypeAlias = ndarray[Any, np.dtype[np.int64]]


@runtime_checkable
class Backend(Protocol):
    """Protocol describing a backend factory registered with ragcore."""

    name: str

    def capabilities(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable description of backend capabilities."""

    def build(self, spec: Mapping[str, Any]) -> Handle:
        """Construct a new handle that implements :class:`Handle`."""


@runtime_checkable
class Handle(Protocol):
    """Protocol describing the concrete index handle exposed to Python."""

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
    ) -> Mapping[str, FloatArray | IntArray]: ...

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
        kind = str(kind_value)
        metric = str(metric_value)
        if not isinstance(dim_value, int) or dim_value <= 0:
            raise ValueError("dim must be a positive integer")
        params = mapping.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping if provided")
        return cls(
            backend=backend,
            kind=kind,
            metric=metric,
            dim=int(dim_value),
            params=dict(params),
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
    """CPU serialised payload for an index."""

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
            "vectors": self.vectors.astype(np.float32).tolist(),
            "ids": self.ids.astype(np.int64).tolist(),
            "metadata": dict(self.metadata),
            "is_trained": bool(self.is_trained),
            "is_gpu": bool(self.is_gpu),
        }


__all__ = [
    "Backend",
    "Handle",
    "IndexSpec",
    "SerializedIndex",
    "FloatArray",
    "IntArray",
]
