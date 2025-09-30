from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragcore.backends.base import VectorIndexHandle
from ragcore.interfaces import IndexSpec


class PyFlatBackend:
    """Pure Python flat backend supporting inner product and L2."""

    name = "py_flat"
    _SUPPORTED_METRICS = {"l2", "ip"}

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "supports_gpu": False,
            "kinds": {
                "flat": {
                    "requires_training": False,
                    "metrics": sorted(self._SUPPORTED_METRICS),
                }
            },
        }

    def build(self, spec: Mapping[str, Any]) -> PyFlatHandle:
        config = IndexSpec.from_mapping(spec, default_backend=self.name)
        if config.kind != "flat":
            raise ValueError("PyFlat backend only supports kind='flat'")
        if config.metric not in self._SUPPORTED_METRICS:
            raise ValueError(f"unsupported PyFlat metric '{config.metric}'")
        if config.backend != self.name:
            raise ValueError(f"PyFlat backend cannot build for '{config.backend}'")
        return PyFlatHandle(config)


class PyFlatHandle(VectorIndexHandle):
    def __init__(
        self,
        spec: IndexSpec,
        *,
        requires_training: bool | None = None,
        supports_gpu: bool | None = None,
    ) -> None:
        super().__init__(
            spec,
            requires_training=False if requires_training is None else requires_training,
            supports_gpu=False if supports_gpu is None else supports_gpu,
        )
        self._factory_kwargs = {
            "requires_training": self._requires_training,
            "supports_gpu": self._supports_gpu,
        }


__all__ = ["PyFlatBackend", "PyFlatHandle"]
