from __future__ import annotations

from typing import Dict, Iterable

from ragcore.interfaces import Backend


_REGISTRY: Dict[str, Backend] = {}


def register(backend: Backend) -> None:
    """Register a backend instance by name."""

    if not isinstance(backend, Backend):  # type: ignore[arg-type]
        raise TypeError("Backend protocol not satisfied")

    name = getattr(backend, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("backend must define a non-empty 'name' attribute")
    if name in _REGISTRY:
        raise ValueError(f"backend '{name}' already registered")
    _REGISTRY[name] = backend


def get(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise LookupError(f"backend '{name}' is not registered") from exc


def list_backends() -> Iterable[str]:
    return tuple(sorted(_REGISTRY))


def _reset_registry() -> None:
    _REGISTRY.clear()


__all__ = ["register", "get", "list_backends", "_reset_registry"]

