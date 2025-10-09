from __future__ import annotations

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pytest.skip("numpy is required for pyflat handle tests", allow_module_level=True)

from ragcore.backends.pyflat import PyFlatBackend, PyFlatHandle

SPEC = {"backend": "py_flat", "kind": "flat", "metric": "l2", "dim": 4}


def test_pyflat_handle_to_gpu_clone() -> None:
    backend = PyFlatBackend()
    handle = backend.build(SPEC)
    handle.add(np.zeros((1, SPEC["dim"]), dtype=np.float32))

    clone = handle.to_gpu()

    assert isinstance(clone, PyFlatHandle)
    assert clone.spec() == handle.spec()
    assert clone.ntotal() == handle.ntotal()
    assert clone.is_gpu is False


def test_pyflat_handle_merge_with_preserves_spec() -> None:
    backend = PyFlatBackend()
    left = backend.build(SPEC)
    right = backend.build(SPEC)
    left.add(np.zeros((1, SPEC["dim"]), dtype=np.float32))
    right.add(np.ones((2, SPEC["dim"]), dtype=np.float32))

    merged = left.merge_with(right)

    assert isinstance(merged, PyFlatHandle)
    assert merged.ntotal() == left.ntotal() + right.ntotal()
    assert merged.spec() == left.spec() == right.spec()
