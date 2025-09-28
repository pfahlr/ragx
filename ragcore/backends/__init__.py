from .base import Backend, Handle, IndexSpec, SerializedIndex, VectorIndexHandle
from .cuvs import CuVSBackend
from .faiss import FaissBackend
from .hnsw import HnswBackend


DEFAULT_BACKENDS = (FaissBackend, HnswBackend, CuVSBackend)


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
    "CuVSBackend",
    "FaissBackend",
    "HnswBackend",
]

