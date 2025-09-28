import pathlib

import numpy as np
import pytest
from conftest import skip_if_no_eval


def _maybe_import_dummy():
    try:
        from ragcore.backends.dummy import DummyBackend
        from ragcore.registry import get, list_backends, register
        return register, get, list_backends, DummyBackend
    except Exception as e:
        pytest.skip(f"ragcore not available yet: {e}")

def _collect_eval_texts(eval_dir: pathlib.Path, limit: int = 50) -> list[str]:
    texts = []
    # Prefer .md to exercise markdown path; PDFs could be added once loader exists
    for p in eval_dir.rglob("*.md"):
        try:
            texts.append(p.read_text(encoding="utf-8")[:4000])
        except Exception:
            continue
        if len(texts) >= limit:
            break
    return texts

@skip_if_no_eval
def test_dummy_pipeline_index_and_search(eval_dir: pathlib.Path):
    register, get, list_backends, DummyBackend = _maybe_import_dummy()

    # Register dummy backend for tests
    register(DummyBackend())
    backend = get("dummy")

    dim = 64
    spec = {"dim": dim, "metric": "ip", "kind": "flat"}
    handle = backend.build(spec)

    texts = _collect_eval_texts(eval_dir, limit=64)
    if not texts:
        pytest.skip("No .md files in /eval/verification to build a tiny index")

    # Simple deterministic embedder stub: bag-of-chars hashing into dim
    def embed(text: str) -> np.ndarray:
        vec = np.zeros((dim,), dtype=np.float32)
        for ch in text[:2000]:
            i = (ord(ch) + 31) % dim
            vec[i] += 1.0
        # add small deterministic noise for variety
        for i in range(dim):
            vec[i] += ((i * 17) % 5) * 0.01
        # normalize for ip
        n = np.linalg.norm(vec) or 1.0
        return (vec / n).astype(np.float32)

    xb = np.stack([embed(t) for t in texts], axis=0)
    handle.add(xb)
    assert handle.ntotal() == xb.shape[0]

    # Query with the first text (should rank itself highest)
    q = embed(texts[0])[None, :]
    res = handle.search(q, k=min(5, xb.shape[0]))
    ids, dists = res["ids"], res["dists"]
    assert ids.shape == dists.shape
    assert ids.shape[0] == 1
    assert ids.shape[1] >= 1
    # Top-1 should be an in-range index
    top_id = int(ids[0, 0])
    assert 0 <= top_id < xb.shape[0]
    # Serialize (contract)
    _ = handle.serialize_cpu()
