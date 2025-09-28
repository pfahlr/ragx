from __future__ import annotations

from typing import Any, Mapping
import os

from ragcore.interfaces import Backend

_HAS_EXTENSION = False
_IMPORT_ERROR: Exception | None = None

if os.environ.get("RAGCORE_DISABLE_CPP") == "1":
    _IMPORT_ERROR = RuntimeError("disabled via RAGCORE_DISABLE_CPP")
    CppBackend = None  # type: ignore[assignment]
    CppHandle = None  # type: ignore[assignment]
else:
    try:
        from _ragcore_cpp import CppBackend as _CppBackend, CppHandle as CppHandle  # type: ignore[assignment]

        _HAS_EXTENSION = True

        class CppBackend(_CppBackend):  # type: ignore[misc]
            name = "cpp"

            def capabilities(self) -> Mapping[str, Any]:
                info = dict(super().capabilities())
                info.setdefault("name", self.name)
                info.setdefault("available", True)
                return info

    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc
        CppHandle = None  # type: ignore[assignment]


if not _HAS_EXTENSION:

    class CppBackend(Backend):  # type: ignore[misc]
        name = "cpp"

        def capabilities(self) -> Mapping[str, Any]:
            return {
                "name": self.name,
                "available": False,
                "reason": str(_IMPORT_ERROR) if _IMPORT_ERROR else "extension not built",
            }

        def build(self, spec: Mapping[str, Any]):  # type: ignore[override]
            raise RuntimeError("ragcore C++ backend is unavailable")

    class CppHandle:  # pragma: no cover - fallback placeholder
        pass


HAS_CPP_EXTENSION = _HAS_EXTENSION


def is_available() -> bool:
    return HAS_CPP_EXTENSION


__all__ = ["CppBackend", "CppHandle", "HAS_CPP_EXTENSION", "is_available"]

