from ragcore.interfaces import Backend, Handle, IndexSpec, SerializedIndex, VectorIndexHandle

from .cuvs import CuVSBackend
from .dummy import DummyBackend
from .faiss import FaissBackend
from .hnsw import HnswBackend

DEFAULT_BACKENDS = (DummyBackend, FaissBackend, HnswBackend, CuVSBackend)

try:  # pragma: no cover - optional native dependency
    from .cpp import CPPBackend as _CPPBackend
    from .cpp import is_available as _cpp_is_available
except Exception:  # pragma: no cover - extension missing during import
    _CPPBackend = None
else:  # pragma: no cover - importable extension, exercised in unit tests
    if _cpp_is_available():
        DEFAULT_BACKENDS = (*DEFAULT_BACKENDS, _CPPBackend)


def register_default_backends() -> None:
    """Register the default set of backends with :mod:`ragcore.registry`."""

    from ragcore.registry import list_backends, register

    existing = set(list_backends())
    for backend_cls in DEFAULT_BACKENDS:
        name = getattr(backend_cls, "name", None)
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
    "CuVSBackend",
    "FaissBackend",
    "HnswBackend",
]

if '_CPPBackend' in globals() and _CPPBackend is not None and '_cpp_is_available' in globals():
    try:  # pragma: no cover - optional export
        if _cpp_is_available():
            __all__.append("CPPBackend")
    except Exception:  # pragma: no cover - guards against runtime errors
        pass
