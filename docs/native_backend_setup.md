# Native Backend Setup

The optional C++ backend lives in `cpp/` and builds a pybind11 extension
named `_ragcore_cpp`. The Python shim at `ragcore/backends/cpp/__init__.py`
loads the module when available and falls back to a stub implementation when
it cannot be imported.

## Prerequisites

* Python 3.11+
* `pybind11` headers (installed automatically via `pip install -r requirements.txt`)
* A C++17 toolchain and CMake 3.15+

## Building the stub backend

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
```

The build produces `_ragcore_cpp` in the CMake build directory. Copy or add the
build directory to `PYTHONPATH` so that `import _ragcore_cpp` succeeds.

## Optional import behaviour

If the extension cannot be imported, the shim still exposes `CppBackend`, but
`capabilities()` reports `available: False` and `build()` raises
`RuntimeError`. Setting `RAGCORE_DISABLE_CPP=1` forces the fallback path, which
is useful when running unit tests without the compiled module.

When the extension is available, `CppBackend` and `CppHandle` are provided by
pybind11 and store vectors in `std::vector` instances. The capabilities payload
currently reports stub data (`available: True`, `kinds: []`) and will be
extended as feature work continues.

