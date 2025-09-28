from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from ragcore.backends.pyflat import PyFlatBackend, PyFlatHandle
from ragcore.registry import _reset_registry


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterable[None]:
    """Ensure a clean backend registry for each test."""

    _reset_registry()
    yield
    _reset_registry()


def test_capabilities_and_build() -> None:
    backend = PyFlatBackend()

    capabilities = backend.capabilities()
    assert capabilities["name"] == "py_flat"
    assert capabilities["supports_gpu"] is False
    assert capabilities["kinds"]["flat"]["requires_training"] is False
    assert set(capabilities["metrics"]) == {"ip", "l2"}

    handle = backend.build({"kind": "flat", "metric": "ip", "dim": 4})
    assert isinstance(handle, PyFlatHandle)
    assert handle.requires_training() is False
    assert handle.ntotal() == 0


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_add_and_search_deterministic(metric: str) -> None:
    backend = PyFlatBackend()
    handle = backend.build({"kind": "flat", "metric": metric, "dim": 3})

    rng = np.random.default_rng(seed=1234)
    base = rng.normal(size=(6, 3)).astype("float32")
    queries = base[:2]

    handle.add(base)

    results = handle.search(queries, k=3)
    assert results["ids"].shape == (2, 3)
    assert results["distances"].shape == (2, 3)

    if metric == "ip":
        expected = np.argsort(-(queries @ base.T), axis=1)[:, :3]
    else:  # l2
        diffs = queries[:, None, :] - base[None, :, :]
        expected = np.argsort(np.sum(diffs * diffs, axis=2), axis=1)[:, :3]

    assert np.array_equal(results["ids"], expected)


def test_serialize_round_trip_preserves_vectors() -> None:
    backend = PyFlatBackend()
    handle = backend.build({"kind": "flat", "metric": "ip", "dim": 2})

    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    ids = np.array([10, 11], dtype="int64")
    handle.add(vectors, ids=ids)

    serialized = handle.serialize_cpu()
    assert serialized.is_trained is True
    assert serialized.is_gpu is False
    np.testing.assert_allclose(serialized.vectors, vectors)
    np.testing.assert_array_equal(serialized.ids, ids)

    clone = PyFlatHandle.from_serialized(serialized)
    assert clone.ntotal() == 2
    round_trip = clone.search(np.array([[1.0, 0.0]], dtype="float32"), k=1)
    assert np.array_equal(round_trip["ids"], np.array([[10]]))
