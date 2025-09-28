from __future__ import annotations

from typing import cast

from ragcore.interfaces import Backend, Handle, IndexSpec, SerializedIndex, VectorIndexHandle

from .cuvs import CuVSBackend
from .dummy import DummyBackend
from .faiss import FaissBackend
from .hnsw import HnswBackend
from .pyflat import PyFlatBackend

_backend_classes: list[type[Backend]] = [
    DummyBackend,
    FaissBackend,
    HnswBackend,
    PyFlatBackend,
    CuVSBackend,
]

try:  # pragma: no cover - optional native dependency
    from .cpp import CPPBackend as _CPPBackend
    from .cpp import is_available as _cpp_is_available
except Exception:  # pragma: no cover - extension missing during import
    _CPPBackend = None
    
    def _cpp_is_available() -> bool:  # type: ignore[no-redef]
        return False
else:  # pragma: no cover - importable extension, exercised in unit tests
    if _cpp_is_available():
        _backend_classes.append(cast(type[Backend], _CPPBackend))

DEFAULT_BACKENDS: tuple[type[Backend], ...] = tuple(_backend_classes)


def register_default_backends() -> None:
    """Register the default set of backends with :mod:`ragcore.registry`."""

    from ragcore.registry import list_backends, register

    existing = set(list_backends())
    for backend_cls in DEFAULT_BACKENDS:
        name = getattr(backend_cls, "name", None)
        if not isinstance(name, str):
            continue
        if name in existing:
            continue
        register(backend_cls())
        existing.add(name)


__all__ = [
    "Backend",
    "Handle",
    "IndexSpec",
    "SerializedIndex",
    "VectorIndexHandle",
    "DEFAULT_BACKENDS",
    "register_default_backends",
    "DummyBackend",
    "PyFlatBackend",
    "CuVSBackend",
    "FaissBackend",
    "HnswBackend",
]

try:  # pragma: no cover - optional export
    if _CPPBackend is not None and _cpp_is_available():
        __all__.append("CPPBackend")
except Exception:  # pragma: no cover - guards against runtime errors
    pass
