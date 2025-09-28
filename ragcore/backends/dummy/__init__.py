from __future__ import annotations

from typing import Any, Mapping

from ragcore.interfaces import IndexSpec

from ..base import VectorIndexHandle


class DummyBackend:
    """Pure-python backend suitable for tests and smoke flows."""

    name = "dummy"
    _SUPPORTED_KINDS = {"flat"}
    _SUPPORTED_METRICS = {"ip", "l2"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "kinds": {
                "flat": {
                    "requires_training": False,
                    "supports_merge": True,
                }
            },
            "metrics": sorted(self._SUPPORTED_METRICS),
            "supports_gpu": False,
        }

    def build(self, spec: Mapping[str, Any]) -> "DummyHandle":
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.kind not in self._SUPPORTED_KINDS:
            raise ValueError(f"dummy backend only supports kinds: {sorted(self._SUPPORTED_KINDS)}")
        if config.metric not in self._SUPPORTED_METRICS:
            raise ValueError(f"dummy backend only supports metrics: {sorted(self._SUPPORTED_METRICS)}")
        if config.backend != self.name:
            raise ValueError(f"dummy backend cannot build for '{config.backend}'")
        handle = DummyHandle(config)
        handle._factory_kwargs = {}
        return handle


class DummyHandle(VectorIndexHandle):
    """Vector index handle backed by the in-memory :class:`VectorIndexHandle`."""

    def __init__(self, spec: IndexSpec) -> None:
        super().__init__(spec, requires_training=False, supports_gpu=False)


__all__ = ["DummyBackend", "DummyHandle"]

