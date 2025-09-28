"""Tests for the optional C++ backend shim."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
import pytest


def _reload_cpp_module(monkeypatch: pytest.MonkeyPatch, *, force_missing: bool = False):
    """Reload ``ragcore.backends.cpp`` with optional native stub control."""

    target_prefix = "ragcore.backends.cpp"
    for module_name in list(sys.modules):
        if module_name == target_prefix or module_name.startswith(f"{target_prefix}."):
            sys.modules.pop(module_name)

    if force_missing:
        real_import = importlib.import_module

        def fake_import(name: str, package: str | None = None) -> Any:  # pragma: no cover - helper
            if name in {"ragcore.backends._ragcore_cpp", "_ragcore_cpp"}:
                raise ModuleNotFoundError(name)
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", fake_import)

    return importlib.import_module("ragcore.backends.cpp")


def test_optional_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_cpp_module(monkeypatch, force_missing=True)
    assert module.is_available() is False
    with pytest.raises(RuntimeError):
        module.ensure_available()
    with pytest.raises(RuntimeError):
        module.get_backend()


def test_cpp_backend_stub_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_cpp_module(monkeypatch)
    if not module.is_available():
        module.build_native(force=True)

    module.ensure_available()
    backend = module.get_backend()
    caps = backend.capabilities()
    assert caps["name"] == "cpp"
    assert caps["supports_gpu"] is False
    assert "flat" in caps["kinds"]

    spec = {"backend": "cpp", "kind": "flat", "metric": "l2", "dim": 3}
    handle = backend.build(spec)
    assert handle.requires_training() is False

    base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    handle.add(base)
    result = handle.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), k=1)

    assert tuple(result.keys()) == ("ids", "distances")
    assert result["ids"].shape == (1, 1)
    assert result["ids"][0, 0] == 0
    assert result["distances"].shape == (1, 1)
    assert handle.ntotal() == 2
