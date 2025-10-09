from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("numpy")

from ragcore.registry import _reset_registry, get, list_backends, register


class StubBackend:
    name = "stub"

    def __init__(self) -> None:
        self._build_calls: list[Mapping[str, Any]] = []

    def capabilities(self) -> Mapping[str, Any]:
        return {"name": self.name, "kinds": ["flat"], "metrics": ["ip"]}

    def build(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        self._build_calls.append(spec)
        return {"spec": spec}


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    _reset_registry()
    yield
    _reset_registry()


def test_register_and_get_backend() -> None:
    backend = StubBackend()
    register(backend)

    names = list_backends()
    assert list(names) == ["stub"]

    retrieved = get("stub")
    assert retrieved is backend
    assert retrieved.capabilities()["name"] == "stub"


def test_register_rejects_non_backend() -> None:
    class NotABackend:
        name = "broken"

    with pytest.raises(TypeError, match="Backend protocol"):
        register(NotABackend())


def test_get_unknown_backend() -> None:
    with pytest.raises(LookupError, match="not registered"):
        get("missing")


def test_register_prevents_duplicates() -> None:
    register(StubBackend())
    with pytest.raises(ValueError, match="already registered"):
        register(StubBackend())

