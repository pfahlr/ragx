from __future__ import annotations

import numpy as np
import pytest

from ragcore.backends.pyflat import PyFlatBackend


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_pyflat_add_and_search(metric: str) -> None:
    backend = PyFlatBackend()
    handle = backend.build({
        "backend": "py_flat",
        "kind": "flat",
        "metric": metric,
        "dim": 3,
    })

    assert handle.requires_training() is False

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    handle.add(base)

    queries = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = handle.search(queries, k=2)

    assert result["ids"].shape == (1, 2)
    assert result["distances"].shape == (1, 2)
    assert int(result["ids"][0, 0]) == 1


def test_pyflat_serializes_round_trip() -> None:
    backend = PyFlatBackend()
    spec = {
        "backend": "py_flat",
        "kind": "flat",
        "metric": "l2",
        "dim": 2,
    }
    handle = backend.build(spec)
    vectors = np.array([[0.5, 0.25], [0.1, 0.9]], dtype=np.float32)
    handle.add(vectors)

    serialized = handle.serialize_cpu()
    assert serialized.spec["backend"] == "py_flat"
    np.testing.assert_allclose(serialized.vectors, vectors)
    np.testing.assert_array_equal(serialized.ids, np.array([0, 1], dtype=np.int64))


def test_pyflat_rejects_unknown_metric() -> None:
    backend = PyFlatBackend()
    with pytest.raises(ValueError, match="unsupported PyFlat metric"):
        backend.build(
            {
                "backend": "py_flat",
                "kind": "flat",
                "metric": "cosine",
                "dim": 2,
            }
        )


def test_pyflat_clone_handles_gpu_and_merge() -> None:
    backend = PyFlatBackend()
    spec = {
        "backend": "py_flat",
        "kind": "flat",
        "metric": "l2",
        "dim": 2,
    }
    left = backend.build(spec)
    right = backend.build(spec)

    left.add(np.array([[0.0, 0.0]], dtype=np.float32))
    right.add(np.array([[1.0, 0.0]], dtype=np.float32))

    # Ensure clone uses the same kwargs VectorIndexHandle._clone emits.
    left._factory_kwargs.update({"requires_training": False, "supports_gpu": False})

    gpu_clone = left.to_gpu()
    assert gpu_clone.ntotal() == left.ntotal()
    assert gpu_clone.is_gpu is False
    assert gpu_clone.device is None

    right._factory_kwargs.update({"requires_training": False, "supports_gpu": False})
    merged = left.merge_with(right)
    assert merged.ntotal() == left.ntotal() + right.ntotal()
    np.testing.assert_array_equal(merged._ids, np.arange(merged.ntotal(), dtype=np.int64))
