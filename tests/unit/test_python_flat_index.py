from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragcore import registry
from ragcore.backends import register_default_backends
from ragcore.backends.pyflat import PyFlatBackend, PyFlatHandle


@pytest.fixture()
def backend() -> PyFlatBackend:
    pytest.importorskip("numpy")
    return PyFlatBackend()


def test_pyflat_capabilities(backend: PyFlatBackend) -> None:
    caps = backend.capabilities()
    assert caps["name"] == "py_flat"
    assert caps["supports_gpu"] is False
    assert caps["kinds"]["flat"]["requires_training"] is False
    assert caps["metrics"] == ["ip", "l2"], "Supported metrics should be sorted"


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_pyflat_build_validates_spec(metric: str, backend: PyFlatBackend) -> None:
    spec = {"backend": "py_flat", "kind": "flat", "metric": metric, "dim": 3}
    handle = backend.build(spec)
    assert isinstance(handle, PyFlatHandle)
    assert handle.requires_training() is False


def test_pyflat_add_and_search_l2(backend: PyFlatBackend) -> None:
    handle = backend.build({"backend": "py_flat", "kind": "flat", "metric": "l2", "dim": 3})
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    handle.add(base)

    queries = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )

    results = handle.search(queries, k=2)
    assert results["ids"].shape == (2, 2)
    assert results["distances"].shape == (2, 2)

    np.testing.assert_allclose(results["ids"], np.array([[1, 0], [2, 0]], dtype=np.int64))
    np.testing.assert_allclose(
        results["distances"],
        np.array([[0.0, 1.0], [0.02, 0.82]], dtype=np.float32),
        atol=1e-6,
    )


def test_pyflat_search_inner_product_orders_descending(backend: PyFlatBackend) -> None:
    handle = backend.build({"backend": "py_flat", "kind": "flat", "metric": "ip", "dim": 2})
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    handle.add(vectors)

    queries = np.array(
        [
            [0.6, 0.8],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = handle.search(queries, k=2)
    np.testing.assert_allclose(result["ids"], np.array([[1, 2], [0, 2]], dtype=np.int64))
    np.testing.assert_allclose(
        result["distances"],
        np.array(
            [
                [-0.8, -0.7],
                [-1.0, -0.5],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_pyflat_serialize_and_deserialize_round_trip(
    tmp_path: Path, backend: PyFlatBackend
) -> None:
    handle = backend.build({"backend": "py_flat", "kind": "flat", "metric": "ip", "dim": 2})
    handle.add(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        ids=np.array([10, 20], dtype=np.int64),
    )

    serialised = handle.serialize_cpu()
    json_path = tmp_path / "index.json"
    json_path.write_text(json.dumps(serialised.to_dict()), encoding="utf-8")

    restored = backend.deserialize_cpu(json.loads(json_path.read_text(encoding="utf-8")))
    assert isinstance(restored, PyFlatHandle)
    assert restored.ntotal() == 2

    query = np.array([[1.0, 0.0]], dtype=np.float32)
    original = handle.search(query, k=2)
    round_tripped = restored.search(query, k=2)
    np.testing.assert_allclose(original["ids"], round_tripped["ids"])
    np.testing.assert_allclose(original["distances"], round_tripped["distances"])


def test_pyflat_registered_with_defaults() -> None:
    registry._reset_registry()
    register_default_backends()
    assert "py_flat" in registry.list_backends()
