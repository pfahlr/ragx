from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest


@pytest.fixture(name="interfaces")
def _interfaces_module():
    import ragcore.interfaces as interfaces

    return interfaces


def test_backend_protocol_contract(interfaces) -> None:
    Backend = interfaces.Backend

    class FakeHandle:
        def __init__(self) -> None:
            self._trained = False
            self.is_gpu = False
            self.device = None

        def requires_training(self) -> bool:
            return False

        def train(self, vectors: np.ndarray) -> None:
            self._trained = True

        def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
            if ids is not None and ids.shape[0] != vectors.shape[0]:
                raise ValueError("ids/vectors mismatch")

        def search(
            self,
            queries: np.ndarray,
            k: int,
            **kwargs: Any,
        ) -> Mapping[str, np.ndarray]:
            return {
                "ids": np.arange(len(queries))[:, None],
                "distances": np.zeros((len(queries), 1)),
            }

        def ntotal(self) -> int:
            return 0

        def serialize_cpu(self):
            return interfaces.SerializedIndex(
                spec={},
                vectors=np.zeros((0, 1), dtype=np.float32),
                ids=np.zeros((0,), dtype=np.int64),
                metadata={},
                is_trained=True,
                is_gpu=False,
            )

        def to_gpu(self, device: str | None = None):
            return self

        def merge_with(self, other: Any):
            return self

        def spec(self) -> Mapping[str, Any]:
            return {"backend": "fake", "kind": "flat", "metric": "ip", "dim": 1}

    class FakeBackend:
        name = "fake"

        def capabilities(self) -> Mapping[str, Any]:
            return {"name": self.name, "kinds": ["flat"], "metrics": ["ip"]}

        def build(self, spec: Mapping[str, Any]) -> FakeHandle:
            return FakeHandle()

    backend = FakeBackend()
    assert isinstance(backend, Backend)

    capabilities_sig = inspect.signature(Backend.capabilities)
    assert list(capabilities_sig.parameters) == ["self"]

    build_sig = inspect.signature(Backend.build)
    assert list(build_sig.parameters) == ["self", "spec"]
    annotation = build_sig.return_annotation
    if isinstance(annotation, str):
        assert "Handle" in annotation
    else:
        assert getattr(annotation, "__name__", "").startswith("Handle")


def test_handle_protocol_contract(interfaces) -> None:
    Handle = interfaces.Handle

    class ImplementsHandle:
        def __init__(self) -> None:
            self.is_gpu = False
            self.device = None

        def requires_training(self) -> bool:
            return False

        def train(self, vectors: np.ndarray) -> None:  # pragma: no cover - protocol contract stub
            raise NotImplementedError

        def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
            raise NotImplementedError

        def search(self, queries: np.ndarray, k: int, **kwargs: Any) -> Mapping[str, np.ndarray]:
            raise NotImplementedError

        def ntotal(self) -> int:
            return 0

        def serialize_cpu(self) -> interfaces.SerializedIndex:
            return interfaces.SerializedIndex(
                spec={},
                vectors=np.zeros((0, 1), dtype=np.float32),
                ids=np.zeros((0,), dtype=np.int64),
                metadata={},
                is_trained=True,
                is_gpu=False,
            )

        def to_gpu(self, device: str | None = None) -> ImplementsHandle:
            return self

        def merge_with(self, other: ImplementsHandle) -> ImplementsHandle:
            return self

        def spec(self) -> Mapping[str, Any]:
            return {"backend": "fake", "kind": "flat", "metric": "ip", "dim": 1}

    handle = ImplementsHandle()
    assert isinstance(handle, Handle)

    search_sig = inspect.signature(Handle.search)
    param_names = list(search_sig.parameters)
    assert param_names[:3] == ["self", "queries", "k"]
    assert param_names[3] == "kwargs"
    assert search_sig.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD


def test_serialized_index_round_trip(interfaces) -> None:
    vectors = np.array([[1.0, 2.0]], dtype=np.float32)
    ids = np.array([42], dtype=np.int64)
    serialized = interfaces.SerializedIndex(
        spec={"backend": "dummy", "kind": "flat", "metric": "ip", "dim": 2},
        vectors=vectors,
        ids=ids,
        metadata={"note": "unit"},
        is_trained=True,
        is_gpu=False,
    )
    payload = serialized.to_dict()
    assert payload["spec"]["backend"] == "dummy"
    assert payload["vectors"] == [[1.0, 2.0]]
    assert payload["ids"] == [42]
    assert payload["metadata"]["note"] == "unit"
    assert payload["is_trained"] is True
    assert payload["is_gpu"] is False


def test_index_spec_validation(interfaces) -> None:
    spec = interfaces.IndexSpec.from_mapping(
        {
            "backend": "dummy",
            "kind": "flat",
            "metric": "ip",
            "dim": 3,
            "params": {"foo": "bar"},
        }
    )
    assert spec.backend == "dummy"
    assert spec.as_dict()["params"] == {"foo": "bar"}

    with pytest.raises(ValueError, match="index spec must include a backend"):
        interfaces.IndexSpec.from_mapping({"kind": "flat", "metric": "ip", "dim": 1})

    with pytest.raises(ValueError, match="index spec missing required fields"):
        interfaces.IndexSpec.from_mapping({"backend": "dummy", "metric": "ip", "dim": 1})

    with pytest.raises(ValueError, match="dim must be a positive integer"):
        interfaces.IndexSpec.from_mapping(
            {"backend": "dummy", "kind": "flat", "metric": "ip", "dim": 0}
        )
