from __future__ import annotations

import importlib

import numpy as np
import pytest

from ragcore import registry
from ragcore.backends import register_default_backends
from ragcore.backends.pyflat import PyFlatBackend
from ragcore.interfaces import SerializedIndex


@pytest.fixture(scope="module")
def cpp_module() -> object:
    module = importlib.import_module("ragcore.backends.cpp")
    if not module.is_available():  # type: ignore[attr-defined]
        try:
            module.build_native(force=True)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - environment guard
            pytest.skip(f"cpp backend unavailable: {exc}")
    module.ensure_available()  # type: ignore[attr-defined]
    return module


def _example_vectors() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([10, 11, 12, 13], dtype=np.int64)
    queries = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return base, ids, queries


def test_parity_py_vs_cpp(cpp_module: object) -> None:
    backend = cpp_module.get_backend()  # type: ignore[attr-defined]
    py_backend = PyFlatBackend()

    vectors, ids, queries = _example_vectors()

    for metric in ("l2", "ip"):
        cpp_spec = {"backend": "cpp", "kind": "flat", "metric": metric, "dim": 3}
        py_spec = {"backend": "py_flat", "kind": "flat", "metric": metric, "dim": 3}
        cpp_handle = backend.build(cpp_spec)
        py_handle = py_backend.build(py_spec)
        cpp_handle.add(vectors, ids=ids)
        py_handle.add(vectors, ids=ids)

        cpp_results = cpp_handle.search(queries, k=3)
        py_results = py_handle.search(queries, k=3)

        np.testing.assert_allclose(cpp_results["distances"], py_results["distances"])
        np.testing.assert_array_equal(cpp_results["ids"], py_results["ids"])


def test_cpp_serialize_roundtrip(cpp_module: object) -> None:
    backend = cpp_module.get_backend()  # type: ignore[attr-defined]
    vectors, ids, queries = _example_vectors()
    handle = backend.build({"backend": "cpp", "kind": "flat", "metric": "l2", "dim": 3})
    handle.add(vectors, ids=ids)

    serialized = handle.serialize_cpu()
    assert isinstance(serialized, SerializedIndex)
    assert serialized.vectors.shape == vectors.shape
    assert serialized.ids.shape == ids.shape
    assert serialized.metadata["supports_gpu"] is False

    payload = serialized.to_dict()
    roundtrip = SerializedIndex(
        spec=payload["spec"],
        vectors=np.asarray(payload["vectors"], dtype=np.float32),
        ids=np.asarray(payload["ids"], dtype=np.int64),
        metadata=payload["metadata"],
        is_trained=payload["is_trained"],
        is_gpu=payload["is_gpu"],
    )

    np.testing.assert_allclose(roundtrip.vectors, serialized.vectors)
    np.testing.assert_array_equal(roundtrip.ids, serialized.ids)

    results = handle.search(queries, k=2)
    np.testing.assert_array_equal(results["ids"].shape, (queries.shape[0], 2))


def test_cpp_alias_registration(cpp_module: object) -> None:
    registry._reset_registry()
    register_default_backends()

    available = set(registry.list_backends())
    assert "cpp" in available
    assert "cpp_faiss" in available

    alias_backend = registry.get("cpp_faiss")
    vectors, ids, queries = _example_vectors()
    handle = alias_backend.build(
        {"backend": "cpp_faiss", "kind": "flat", "metric": "l2", "dim": 3}
    )
    handle.add(vectors, ids=ids)
    alias_results = handle.search(queries, k=2)

    cpp_backend = registry.get("cpp")
    cpp_handle = cpp_backend.build({"backend": "cpp", "kind": "flat", "metric": "l2", "dim": 3})
    cpp_handle.add(vectors, ids=ids)
    cpp_results = cpp_handle.search(queries, k=2)

    np.testing.assert_allclose(alias_results["distances"], cpp_results["distances"])
    np.testing.assert_array_equal(alias_results["ids"], cpp_results["ids"])
