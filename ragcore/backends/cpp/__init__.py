"""Optional shim for the native C++ backend bindings."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

CPPBackend: type[Any] | None = None
CPPHandle: type[Any] | None = None
_HAS_NATIVE = False
_NATIVE_ERROR: Exception | None = None
__all__ = ("is_available", "ensure_available", "build_native", "get_backend")


def _refresh_exports() -> None:
    global __all__
    exposed = ["is_available", "ensure_available", "build_native", "get_backend"]
    if _HAS_NATIVE:
        exposed.extend(["CPPBackend", "CPPHandle"])
    __all__ = tuple(exposed)


def _load_native() -> None:
    global CPPBackend, CPPHandle, _HAS_NATIVE, _NATIVE_ERROR
    try:
        native = importlib.import_module("ragcore.backends._ragcore_cpp")
    except ModuleNotFoundError as exc:  # pragma: no cover - import path guard
        CPPBackend = None
        CPPHandle = None
        _HAS_NATIVE = False
        _NATIVE_ERROR = exc
    else:
        CPPBackend = native.CPPBackend
        CPPHandle = native.CPPHandle
        _HAS_NATIVE = True
        _NATIVE_ERROR = None
    _refresh_exports()


def is_available() -> bool:
    """Return ``True`` when the native extension is importable."""

    return _HAS_NATIVE


def ensure_available() -> None:
    """Raise a user-friendly error when the native extension is missing."""

    if not _HAS_NATIVE:
        message = (
            "ragcore.backends._ragcore_cpp is not built; run build_native() "
            "to compile the stub backend."
        )
        raise RuntimeError(message) from _NATIVE_ERROR


def get_backend() -> Any:
    """Return a ready-to-use ``CPPBackend`` instance."""

    ensure_available()
    assert CPPBackend is not None  # for type-checkers
    return CPPBackend()


def build_native(*, force: bool = False, verbose: bool = False) -> Path:
    """Compile the native extension in-place and reload the module."""

    from . import _builder

    artifact = _builder.build_extension(force=force, verbose=verbose)
    _load_native()
    return artifact


def __getattr__(name: str) -> Any:  # pragma: no cover - delegated import path
    if name in {"CPPBackend", "CPPHandle"}:
        ensure_available()
        return CPPBackend if name == "CPPBackend" else CPPHandle
    raise AttributeError(name)


_refresh_exports()
_load_native()
