from __future__ import annotations

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pytest.skip("numpy is required for vectordb parity tests", allow_module_level=True)

from ragcore.backends.pyflat import PyFlatBackend

try:
    from ragcore.backends.cpp import CppFaissBackend
except ImportError:  # pragma: no cover - import guard
    pytest.skip("C++ backend unavailable", allow_module_level=True)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_cpp_stub_matches_pyflat(metric: str) -> None:
    dim = 4
    py_spec = {"backend": "py_flat", "kind": "flat", "metric": metric, "dim": dim}
    cpp_spec = {"backend": "cpp_faiss", "kind": "flat", "metric": metric, "dim": dim}

    py_backend = PyFlatBackend()
    cpp_backend = CppFaissBackend()

    py_handle = py_backend.build(py_spec)
    cpp_handle = cpp_backend.build(cpp_spec)

    rng = np.random.default_rng(42)
    base = rng.normal(size=(6, dim)).astype(np.float32)
    queries = rng.normal(size=(3, dim)).astype(np.float32)

    py_handle.add(base)
    cpp_handle.add(base)

    py_result = py_handle.search(queries, k=3)
    cpp_result = cpp_handle.search(queries, k=3)

    np.testing.assert_array_equal(cpp_result["ids"], py_result["ids"])
    np.testing.assert_allclose(
        cpp_result["distances"],
        py_result["distances"],
        rtol=1e-6,
        atol=1e-6,
    )

    py_serialized = py_handle.serialize_cpu()
    cpp_serialized = cpp_handle.serialize_cpu()

    np.testing.assert_allclose(cpp_serialized.vectors, py_serialized.vectors)
    np.testing.assert_array_equal(cpp_serialized.ids, py_serialized.ids)
    assert cpp_serialized.spec["metric"] == metric


def test_cpp_stub_merge_respects_ids() -> None:
    dim = 3
    backend = CppFaissBackend()
    spec = {"backend": "cpp_faiss", "kind": "flat", "metric": "l2", "dim": dim}

    left = backend.build(spec)
    right = backend.build(spec)

    left.add(np.eye(dim, dtype=np.float32))
    right.add((np.eye(dim, dtype=np.float32) * 2).astype(np.float32))

    merged = left.merge_with(right)
    assert merged.ntotal() == 2 * dim
    result = merged.search(np.eye(dim, dtype=np.float32), k=2)
    assert result["ids"].shape == (dim, 2)
