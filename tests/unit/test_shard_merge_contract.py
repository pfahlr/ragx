from __future__ import annotations

import numpy as np
import pytest

from ragcore.backends.pyflat import PyFlatBackend

try:
    from ragcore.backends.cpp import CppFaissBackend
except ImportError:  # pragma: no cover - optional
    CppFaissBackend = None


def _build_handle(backend, *, metric: str, dim: int, data: np.ndarray):
    spec = {"backend": getattr(backend, "name", ""), "kind": "flat", "metric": metric, "dim": dim}
    handle = backend.build(spec)
    handle.add(data)
    return handle


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_vector_index_handle_merge_offsets_ids(metric: str) -> None:
    backend = PyFlatBackend()
    dim = 5
    left_data = np.arange(dim * 2, dtype=np.float32).reshape(2, dim)
    right_data = (np.arange(dim * 3, dtype=np.float32).reshape(3, dim) + 10).astype(np.float32)

    left = _build_handle(backend, metric=metric, dim=dim, data=left_data)
    right = _build_handle(backend, metric=metric, dim=dim, data=right_data)

    merged = left.merge_with(right)
    serialized = merged.serialize_cpu()

    assert merged.ntotal() == 5
    np.testing.assert_array_equal(serialized.ids, np.arange(5, dtype=np.int64))
    assert serialized.spec["metric"] == metric


def test_vector_index_handle_merge_rejects_mismatch() -> None:
    backend = PyFlatBackend()
    dim = 3
    left = _build_handle(backend, metric="l2", dim=dim, data=np.eye(dim, dtype=np.float32))
    right = _build_handle(backend, metric="ip", dim=dim, data=np.eye(dim, dtype=np.float32))

    with pytest.raises(ValueError):
        _ = left.merge_with(right)


def test_vector_index_handle_merge_is_immutable() -> None:
    backend = PyFlatBackend()
    dim = 2
    left = _build_handle(backend, metric="l2", dim=dim, data=np.eye(dim, dtype=np.float32))
    right = _build_handle(backend, metric="l2", dim=dim, data=2 * np.eye(dim, dtype=np.float32))

    left_before = left.serialize_cpu()
    right_before = right.serialize_cpu()

    merged = left.merge_with(right)

    # Original handles untouched
    np.testing.assert_array_equal(left.serialize_cpu().vectors, left_before.vectors)
    np.testing.assert_array_equal(left.serialize_cpu().ids, left_before.ids)
    np.testing.assert_array_equal(right.serialize_cpu().vectors, right_before.vectors)
    np.testing.assert_array_equal(right.serialize_cpu().ids, right_before.ids)
    assert merged is not left and merged is not right


def test_vector_index_handle_merge_carries_training_flags() -> None:
    backend = PyFlatBackend()
    dim = 2
    left = _build_handle(backend, metric="l2", dim=dim, data=np.eye(dim, dtype=np.float32))
    right = _build_handle(backend, metric="l2", dim=dim, data=2 * np.eye(dim, dtype=np.float32))

    left._is_trained = False  # type: ignore[attr-defined]
    right._is_trained = True  # type: ignore[attr-defined]
    right.is_gpu = True
    right.device = "cuda:1"

    merged = left.merge_with(right)

    assert merged.serialize_cpu().is_trained is True
    assert merged.is_gpu is True
    assert merged.device == "cuda:1"


@pytest.mark.skipif(CppFaissBackend is None, reason="C++ backend not available")
def test_cpp_stub_merge_matches_pyflat() -> None:
    dim = 4
    rng = np.random.default_rng(123)
    data_a = rng.normal(size=(3, dim)).astype(np.float32)
    data_b = rng.normal(size=(2, dim)).astype(np.float32)

    py_backend = PyFlatBackend()
    cpp_backend = CppFaissBackend()

    py_left = _build_handle(py_backend, metric="l2", dim=dim, data=data_a)
    py_right = _build_handle(py_backend, metric="l2", dim=dim, data=data_b)

    cpp_left = _build_handle(cpp_backend, metric="l2", dim=dim, data=data_a)
    cpp_right = _build_handle(cpp_backend, metric="l2", dim=dim, data=data_b)

    py_serialized = py_left.merge_with(py_right).serialize_cpu()
    cpp_serialized = cpp_left.merge_with(cpp_right).serialize_cpu()

    np.testing.assert_allclose(cpp_serialized.vectors, py_serialized.vectors)
    np.testing.assert_array_equal(cpp_serialized.ids, py_serialized.ids)
