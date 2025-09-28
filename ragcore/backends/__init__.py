from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

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
    pass

DEFAULT_BACKENDS: tuple[type[Backend], ...] = tuple(_backend_classes)


def _ensure_cpp_backends_registered() -> None:
    """Append the native C++ backends when the extension is available."""

    if _CPPBackend is None or not _cpp_is_available():
        return

    cpp_backend_cls = cast(type[Backend], _CPPBackend)
    names = {
        getattr(cls, "name", None)
        for cls in _backend_classes
        if hasattr(cls, "name")
    }

    cpp_name = getattr(cpp_backend_cls, "name", None)
    if cpp_name not in names and isinstance(cpp_name, str):
        _backend_classes.append(cpp_backend_cls)
        names.add(cpp_name)

    if "cpp_faiss" not in names:

        class _CppFaissBackend:
            """Alias exposing the C++ stub backend under the ``cpp_faiss`` name."""

            name = "cpp_faiss"

            def capabilities(self) -> Mapping[str, Any]:
                backend = cpp_backend_cls()
                capabilities = dict(backend.capabilities())
                capabilities["name"] = self.name
                return capabilities

            def build(self, spec: Mapping[str, Any]) -> Handle:
                backend = cpp_backend_cls()
                adjusted = dict(spec)
                adjusted["backend"] = self.name
                return backend.build(adjusted)

        _backend_classes.append(cast(type[Backend], _CppFaissBackend))

    global DEFAULT_BACKENDS
    DEFAULT_BACKENDS = tuple(_backend_classes)


_ensure_cpp_backends_registered()


def register_default_backends() -> None:
    """Register the default set of backends with :mod:`ragcore.registry`."""

    from ragcore.registry import list_backends, register

    _ensure_cpp_backends_registered()
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
