from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from ragcore.interfaces import (
    Backend,
    Handle,
    IndexSpec,
    SerializedIndex,
    VectorIndexHandle,
)


class _StubBackend:
    name = "stub"

    def __init__(self) -> None:
        self._built_with: dict[str, Any] | None = None

    def capabilities(self) -> dict[str, Any]:
        return {"name": self.name, "kinds": ["flat"]}

    def build(self, spec: Mapping[str, Any]) -> Handle:
        self._built_with = dict(spec)
        parsed = IndexSpec.from_mapping(spec, default_backend=self.name)
        return VectorIndexHandle(parsed, requires_training=False)


def _make_vectors(*rows: tuple[float, ...]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_index_spec_from_mapping_validates_and_preserves_fields() -> None:
    mapping = {"backend": "dummy", "kind": "flat", "metric": "ip", "dim": 2, "params": {"foo": 1}}
    spec = IndexSpec.from_mapping(mapping)
    assert spec.backend == "dummy"
    assert spec.kind == "flat"
    assert spec.metric == "ip"
    assert spec.dim == 2
    assert spec.params == {"foo": 1}
    assert spec.as_dict() == {
        "backend": "dummy",
        "kind": "flat",
        "metric": "ip",
        "dim": 2,
        "params": {"foo": 1},
    }


def test_index_spec_requires_backend_kind_metric_dim() -> None:
    with pytest.raises(ValueError):
        IndexSpec.from_mapping({"kind": "flat", "metric": "ip", "dim": 2})

    spec = IndexSpec.from_mapping(
        {"kind": "flat", "metric": "ip", "dim": 2},
        default_backend="fallback",
    )
    assert spec.backend == "fallback"

    with pytest.raises(ValueError):
        IndexSpec.from_mapping({"backend": "dummy", "metric": "ip", "dim": 2})

    with pytest.raises(ValueError):
        IndexSpec.from_mapping({"backend": "dummy", "kind": "flat", "dim": 2})

    with pytest.raises(ValueError):
        IndexSpec.from_mapping({"backend": "dummy", "kind": "flat", "metric": "ip", "dim": 0})


def test_vector_index_handle_enforces_training_and_searches() -> None:
    spec = IndexSpec.from_mapping({"backend": "dummy", "kind": "flat", "metric": "l2", "dim": 2})

    handle = VectorIndexHandle(spec, requires_training=True)

    with pytest.raises(RuntimeError):
        handle.add(_make_vectors((0.0, 0.0)))

    handle.train(_make_vectors((0.0, 0.0), (1.0, 1.0)))
    handle.add(_make_vectors((0.0, 0.0), (1.0, 1.0)))

    results = handle.search(_make_vectors((0.0, 0.0), (2.0, 2.0)), k=1)
    assert results["ids"].shape == (2, 1)
    assert results["distances"].shape == (2, 1)
    # First query is identical to the first vector so the distance should be ~0.
    assert results["distances"][0, 0] == pytest.approx(0.0)


def test_vector_index_handle_merge_and_serialize_roundtrip() -> None:
    spec = IndexSpec.from_mapping({"backend": "dummy", "kind": "flat", "metric": "ip", "dim": 2})
    left = VectorIndexHandle(spec, requires_training=False)
    right = VectorIndexHandle(spec, requires_training=False)

    left.add(_make_vectors((1.0, 0.0)))
    right.add(_make_vectors((0.0, 1.0)))

    merged = left.merge_with(right)
    assert merged.ntotal() == 2

    serialised = merged.serialize_cpu()
    assert isinstance(serialised, SerializedIndex)

    payload = serialised.to_dict()
    assert json.loads(json.dumps(payload)) == payload  # JSON friendly
    assert payload["spec"]["backend"] == "dummy"
    assert payload["is_trained"] is True
    assert payload["is_gpu"] is False


def test_backend_protocol_builds_vector_index_handle() -> None:
    backend_impl = _StubBackend()
    backend: Backend = backend_impl
    handle = backend.build({"kind": "flat", "metric": "l2", "dim": 2})
    assert isinstance(handle, VectorIndexHandle)
    assert backend_impl._built_with == {"kind": "flat", "metric": "l2", "dim": 2}
