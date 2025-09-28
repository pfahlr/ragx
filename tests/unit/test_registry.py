from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from ragcore import registry
from ragcore.interfaces import IndexSpec, VectorIndexHandle


@dataclass
class _ExampleBackend:
    name: str

    def capabilities(self) -> Mapping[str, Any]:
        return {"name": self.name}

    def build(self, spec: Mapping[str, Any]) -> VectorIndexHandle:
        parsed = IndexSpec.from_mapping(spec, default_backend=self.name)
        return VectorIndexHandle(parsed, requires_training=False)


def setup_function() -> None:
    registry._reset_registry()


def test_register_and_get_backend() -> None:
    backend = _ExampleBackend(name="alpha")
    registry.register(backend)

    fetched = registry.get("alpha")
    assert fetched is backend
    assert list(registry.list_backends()) == ["alpha"]


def test_register_rejects_missing_or_duplicate_names() -> None:
    with pytest.raises(ValueError):
        registry.register(_ExampleBackend(name=""))

    backend = _ExampleBackend(name="alpha")
    registry.register(backend)

    with pytest.raises(ValueError):
        registry.register(_ExampleBackend(name="alpha"))


def test_get_unknown_backend_raises() -> None:
    with pytest.raises(KeyError):
        registry.get("missing")
