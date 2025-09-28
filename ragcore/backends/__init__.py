from .base import (
    Backend,
    FloatArray,
    Handle,
    IndexSpec,
    IntArray,
    SerializedIndex,
    VectorIndexHandle,
)
from .cpp import HAS_CPP_EXTENSION, CppBackend, CppFaissBackend
from .cuvs import CuVSBackend
from .dummy import DummyBackend
from .faiss import FaissBackend
from .hnsw import HnswBackend
from .pyflat import PyFlatBackend

_DEFAULT: list[type[Backend]] = [FaissBackend, HnswBackend, CuVSBackend]
if HAS_CPP_EXTENSION:
    _DEFAULT.append(CppBackend)

DEFAULT_BACKENDS: tuple[type[Backend], ...] = tuple(_DEFAULT)


def register_default_backends() -> None:
    """Register the default set of backends with :mod:`ragcore.registry`."""

    from ragcore.registry import list_backends, register

    existing: set[str] = set(list_backends())
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
    "FloatArray",
    "IntArray",
    "DEFAULT_BACKENDS",
    "register_default_backends",
    "DummyBackend",
    "PyFlatBackend",
    "CuVSBackend",
    "FaissBackend",
    "HnswBackend",
    "CppBackend",
    "CppFaissBackend",
]
