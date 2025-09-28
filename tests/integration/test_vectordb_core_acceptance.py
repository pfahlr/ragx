import numpy as np
import pytest


@pytest.mark.xfail(reason="Vector DB Core not implemented yet", strict=False)
def test_backend_registry_and_dummy_backend():
    # Once ragcore is wired, agents should register a dummy backend and run a simple search
    from ragcore.backends.dummy import DummyBackend  # file exists in the spec pack later
    from ragcore.registry import get, register
    register(DummyBackend())
    b = get("dummy")
    h = b.build({"dim": 4, "metric": "ip", "kind": "flat"})
    xb = np.random.rand(10, 4).astype("float32")
    q = np.random.rand(2, 4).astype("float32")
    h.add(xb)
    res = h.search(q, k=3)
    assert res["ids"].shape == (2, 3)
    assert res["dists"].shape == (2, 3)
