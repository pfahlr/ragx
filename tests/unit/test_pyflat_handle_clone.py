from __future__ import annotations

import numpy as np

from ragcore.backends.pyflat import PyFlatBackend, PyFlatHandle


def _build_handle(dim: int = 4) -> tuple[PyFlatBackend, PyFlatHandle]:
    backend = PyFlatBackend()
    spec = {"backend": "py_flat", "kind": "flat", "metric": "l2", "dim": dim}
    handle = backend.build(spec)
    return backend, handle


def test_pyflat_handle_to_gpu_clone_roundtrip() -> None:
    _, handle = _build_handle()
    handle.add(np.ones((1, 4), dtype=np.float32))

    gpu_handle = handle.to_gpu()

    assert type(gpu_handle) is type(handle)
    assert gpu_handle.ntotal() == handle.ntotal()


def test_pyflat_handle_merge_with_preserves_vectors() -> None:
    _, first = _build_handle()
    _, second = _build_handle()

    first.add(np.ones((1, 4), dtype=np.float32))
    second.add(np.zeros((2, 4), dtype=np.float32))

    merged = first.merge_with(second)

    assert merged.ntotal() == 3
    assert type(merged) is type(first)
