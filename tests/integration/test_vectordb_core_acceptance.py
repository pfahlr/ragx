import pytest

np = pytest.importorskip("numpy")
from numpy.typing import NDArray


@pytest.mark.xfail(reason="Vector DB Core not implemented yet", strict=False)
def test_backend_registry_and_dummy_backend() -> None:
    # Once ragcore is wired, agents should register a dummy backend and run a simple search
    from ragcore.backends.dummy import DummyBackend  # file exists in the spec pack later
    from ragcore.registry import get, register
    register(DummyBackend())
    b = get("dummy")
    h = b.build({"dim": 4, "metric": "ip", "kind": "flat"})
    xb: NDArray[np.float32] = np.random.rand(10, 4).astype("float32")
    q: NDArray[np.float32] = np.random.rand(2, 4).astype("float32")
    h.add(xb)
    res = h.search(q, k=3)
    assert res["ids"].shape == (2, 3)
    assert res["dists"].shape == (2, 3)
