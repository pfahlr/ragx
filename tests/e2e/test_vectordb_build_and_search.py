from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from ragcore.backends.cuvs import CuVSBackend
from ragcore.backends.faiss import FaissBackend
from ragcore.backends.hnsw import HnswBackend
from ragcore.registry import _reset_registry, get, register


@pytest.fixture(autouse=True)
def _clear_registry() -> Iterable[None]:
    _reset_registry()
    yield
    _reset_registry()


@pytest.mark.parametrize(
    "backend, spec",
    [
        (
            FaissBackend(),
            {
                "backend": "faiss",
                "kind": "flat",
                "metric": "l2",
                "dim": 3,
            },
        ),
        (
            FaissBackend(),
            {
                "backend": "faiss",
                "kind": "ivf_flat",
                "metric": "ip",
                "dim": 3,
                "params": {"nlist": 1, "nprobe": 1},
            },
        ),
        (
            HnswBackend(),
            {
                "backend": "hnsw",
                "kind": "hnsw",
                "metric": "cosine",
                "dim": 3,
                "params": {"m": 8, "ef_construction": 24, "ef": 24},
            },
        ),
        (
            CuVSBackend(),
            {
                "backend": "cuvs",
                "kind": "ivf_pq",
                "metric": "l2",
                "dim": 3,
                "params": {"nlist": 1, "m": 1, "nbits": 8},
            },
        ),
    ],
)
def test_small_index_build_and_search(backend, spec) -> None:
    name = spec["backend"]
    register(backend)
    handle = backend.build(spec)

    base = np.eye(3, dtype="float32")
    queries = base.copy()

    if handle.requires_training():
        handle.train(base)

    handle.add(base)

    results = handle.search(queries, k=1)
    assert results["ids"].shape == (3, 1)
    assert np.array_equal(results["ids"].ravel(), np.arange(3))

    serialized = handle.serialize_cpu()
    assert serialized.vectors.shape == (3, 3)

    # Round-trip via registry get
    fetched = get(name)
    assert fetched.capabilities()["name"] == name


def test_merge_shards_across_backends() -> None:
    faiss = FaissBackend()
    hnsw = HnswBackend()
    cuvs = CuVSBackend()

    register(faiss)
    register(hnsw)
    register(cuvs)

    faiss_handle = faiss.build({
        "backend": "faiss",
        "kind": "flat",
        "metric": "l2",
        "dim": 2,
    })
    hnsw_handle = hnsw.build({
        "backend": "hnsw",
        "kind": "hnsw",
        "metric": "l2",
        "dim": 2,
    })
    cuvs_handle = cuvs.build({
        "backend": "cuvs",
        "kind": "ivf_flat",
        "metric": "l2",
        "dim": 2,
        "params": {"nlist": 1},
    })

    for handle in (faiss_handle, hnsw_handle, cuvs_handle):
        if handle.requires_training():
            handle.train(np.eye(2, dtype="float32"))

    shard_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    shard_b = np.array([[2.0, 0.0]], dtype="float32")

    faiss_handle.add(shard_a)
    hnsw_handle.add(shard_a)
    cuvs_handle.add(shard_a)

    faiss_other = faiss.build({
        "backend": "faiss",
        "kind": "flat",
        "metric": "l2",
        "dim": 2,
    })
    hnsw_other = hnsw.build({
        "backend": "hnsw",
        "kind": "hnsw",
        "metric": "l2",
        "dim": 2,
    })
    cuvs_other = cuvs.build({
        "backend": "cuvs",
        "kind": "ivf_flat",
        "metric": "l2",
        "dim": 2,
        "params": {"nlist": 1},
    })

    for other in (faiss_other, hnsw_other, cuvs_other):
        if other.requires_training():
            other.train(np.eye(2, dtype="float32"))
        other.add(shard_b)

    merged_faiss = faiss_handle.merge_with(faiss_other)
    merged_hnsw = hnsw_handle.merge_with(hnsw_other)
    merged_cuvs = cuvs_handle.merge_with(cuvs_other)

    assert merged_faiss.ntotal() == 3
    assert merged_hnsw.ntotal() == 3
    assert merged_cuvs.ntotal() == 3

    for merged in (merged_faiss, merged_hnsw, merged_cuvs):
        search = merged.search(np.array([[1.0, 0.0]], dtype="float32"), k=1)
        assert search["ids"].shape == (1, 1)
        assert search["distances"].shape == (1, 1)

