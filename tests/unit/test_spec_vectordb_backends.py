from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from ragcore.backends import register_default_backends
from ragcore.backends.base import SerializedIndex
from ragcore.backends.cuvs import CuVSBackend
from ragcore.backends.faiss import FaissBackend
from ragcore.backends.hnsw import HnswBackend
from ragcore.registry import list_backends, register, _reset_registry


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterable[None]:
    _reset_registry()
    yield
    _reset_registry()


def test_backends_register_and_list() -> None:
    register_default_backends()

    assert set(list_backends()) == {"faiss", "hnsw", "cuvs"}


def test_faiss_ivf_training_and_serialization() -> None:
    backend = FaissBackend()
    register(backend)

    handle = backend.build({
        "backend": "faiss",
        "kind": "ivf_flat",
        "metric": "l2",
        "dim": 4,
        "params": {"nlist": 4, "nprobe": 2},
    })

    assert handle.requires_training()

    training = np.random.RandomState(42).randn(32, 4).astype("float32")
    handle.train(training)

    xb = np.random.RandomState(0).randn(10, 4).astype("float32")
    handle.add(xb)

    assert handle.ntotal() == 10

    search = handle.search(xb[:2], k=3)
    assert search["ids"].shape == (2, 3)
    assert search["distances"].shape == (2, 3)

    serialized = handle.serialize_cpu()
    assert isinstance(serialized, SerializedIndex)
    assert serialized.spec["kind"] == "ivf_flat"
    assert serialized.vectors.shape == (10, 4)
    assert serialized.is_trained is True


def test_faiss_merge_and_gpu_clone() -> None:
    backend = FaissBackend()
    register(backend)

    spec = {"backend": "faiss", "kind": "flat", "metric": "ip", "dim": 3}
    left = backend.build(spec)
    right = backend.build(spec)

    a = np.eye(3, dtype="float32")
    b = (np.eye(3) * 2).astype("float32")

    left.add(a)
    right.add(b)

    merged = left.merge_with(right)
    assert merged.ntotal() == 6

    gpu_clone = merged.to_gpu(device="cuda:0")
    assert gpu_clone.is_gpu
    assert gpu_clone.device == "cuda:0"
    assert gpu_clone.ntotal() == 6


def test_hnsw_roundtrip() -> None:
    backend = HnswBackend()
    register(backend)

    handle = backend.build({
        "backend": "hnsw",
        "kind": "hnsw",
        "metric": "cosine",
        "dim": 5,
        "params": {"m": 8, "ef_construction": 32},
    })

    assert handle.requires_training() is False

    xb = np.random.RandomState(123).randn(12, 5).astype("float32")
    handle.add(xb)

    results = handle.search(xb[:1], k=4)
    assert results["ids"].shape == (1, 4)
    assert np.all(results["distances"][:, 0] <= results["distances"][:, 1])

    serialized = handle.serialize_cpu()
    assert serialized.spec["backend"] == "hnsw"
    assert serialized.ids.shape == (12,)


def test_cuvs_gpu_behaviour_and_merge() -> None:
    backend = CuVSBackend()
    register(backend)

    spec = {
        "backend": "cuvs",
        "kind": "ivf_pq",
        "metric": "l2",
        "dim": 6,
        "params": {"nlist": 2, "m": 2, "nbits": 8},
    }

    handle = backend.build(spec)
    assert handle.requires_training()

    training = np.random.RandomState(9).randn(16, 6).astype("float32")
    handle.train(training)

    xb = np.random.RandomState(7).randn(8, 6).astype("float32")
    handle.add(xb)
    assert handle.ntotal() == 8

    gpu_handle = handle.to_gpu()
    assert gpu_handle.is_gpu
    assert gpu_handle.ntotal() == 8

    clone = backend.build(spec)
    clone.train(training)
    clone.add(xb[:2])

    merged = handle.merge_with(clone)
    assert merged.ntotal() == 10
    assert merged.serialize_cpu().vectors.shape == (10, 6)

