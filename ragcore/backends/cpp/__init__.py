from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from ragcore.backends.pyflat import PyFlatBackend
from ragcore.interfaces import Backend


class _UnavailableBackend(Backend):
    """Fallback backend used when the optional C++ bindings are missing."""

    name = "cpp"

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "available": False,
            "reason": str(_IMPORT_ERROR) if _IMPORT_ERROR else "extension not built",
        }

    def build(self, spec: Mapping[str, Any]) -> Any:  # type: ignore[override]
        raise RuntimeError("ragcore C++ backend is unavailable")


class _UnavailableFaissBackend(PyFlatBackend):
    """PyFlat-backed alias exposed when native bindings are absent."""

    name = "cpp_faiss"


CppHandle: type[Any] | None = None
CppBackend: type[Backend] = _UnavailableBackend
CppFaissBackend: type[Backend] = _UnavailableFaissBackend

_IMPORT_ERROR: Exception | None = None
_HAS_EXTENSION = False

if os.environ.get("RAGCORE_DISABLE_CPP") == "1":
    _IMPORT_ERROR = RuntimeError("disabled via RAGCORE_DISABLE_CPP")
else:
    try:
        from _ragcore_cpp import CppBackend as _CppBackend
        from _ragcore_cpp import CppHandle as _CppHandle

        class _ExtensionBackend(_CppBackend):  # type: ignore[misc]
            name = "cpp"

            def capabilities(self) -> Mapping[str, Any]:
                info = dict(super().capabilities())
                info.setdefault("name", self.name)
                info.setdefault("available", True)
                return info

        class _ExtensionFaissBackend(_ExtensionBackend):  # type: ignore[misc]
            name = "cpp_faiss"

            def capabilities(self) -> Mapping[str, Any]:
                info = dict(super().capabilities())
                info.setdefault("alias", "py_flat parity")
                return info

        CppBackend = _ExtensionBackend
        CppFaissBackend = _ExtensionFaissBackend
        CppHandle = _CppHandle
        _HAS_EXTENSION = True
    except ModuleNotFoundError as exc:  # pragma: no cover - optional extension
        _IMPORT_ERROR = exc


HAS_CPP_EXTENSION = _HAS_EXTENSION


def is_available() -> bool:
    return HAS_CPP_EXTENSION


__all__ = [
    "CppBackend",
    "CppFaissBackend",
    "CppHandle",
    "HAS_CPP_EXTENSION",
    "is_available",
]
