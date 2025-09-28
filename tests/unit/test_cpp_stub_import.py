from __future__ import annotations

import importlib
import sys
import types

import pytest


def _import_cpp_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_extension: types.ModuleType | None = None,
    disable_env: bool = False,
):
    monkeypatch.delenv("RAGCORE_DISABLE_CPP", raising=False)
    if disable_env:
        monkeypatch.setenv("RAGCORE_DISABLE_CPP", "1")

    monkeypatch.delitem(sys.modules, "ragcore.backends.cpp", raising=False)

    if fake_extension is not None:
        monkeypatch.setitem(sys.modules, "_ragcore_cpp", fake_extension)
    else:
        monkeypatch.delitem(sys.modules, "_ragcore_cpp", raising=False)

    return importlib.import_module("ragcore.backends.cpp")


def test_cpp_backend_import_falls_back_when_extension_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_cpp_module(monkeypatch, disable_env=True)

    assert module.HAS_CPP_EXTENSION is False

    backend = module.CppBackend()
    info = backend.capabilities()
    assert info["name"] == "cpp"
    assert info["available"] is False

    with pytest.raises(RuntimeError):
        backend.build({})


def test_cpp_backend_uses_extension_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_extension = types.ModuleType("_ragcore_cpp")

    class FakeHandle:
        def __init__(self, spec):
            self._spec = spec

        def requires_training(self):
            return False

        def ntotal(self):
            return 0

    class FakeBackend:
        name = "cpp"

        def capabilities(self):
            return {"name": self.name, "available": True, "kinds": ["flat"]}

        def build(self, spec):
            return FakeHandle(spec)

    fake_extension.CppHandle = FakeHandle
    fake_extension.CppBackend = FakeBackend

    module = _import_cpp_module(monkeypatch, fake_extension=fake_extension)

    assert module.HAS_CPP_EXTENSION is True

    backend = module.CppBackend()
    info = backend.capabilities()
    assert info["available"] is True
    assert info["kinds"] == ["flat"]
