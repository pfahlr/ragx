# Native C++ Backend Stub

The optional ``ragcore.backends.cpp`` module wraps the stub C++ backend compiled via
`pybind11`. The extension is not required for Python-only development, but building it locally
provides parity with the future FAISS-backed implementation.

## Prerequisites

- A working C++17 compiler (``gcc`` or ``clang``).
- Python development headers (installed with ``python3-dev`` on Debian/Ubuntu).
- ``pybind11`` and ``numpy`` Python packages (already listed in ``requirements.txt``).

## Building the Extension

```bash
pip install -r requirements.txt  # installs pybind11 and numpy
python -m ragcore.backends.cpp build  # builds the extension in-place
```

Alternatively, the unit tests automatically invoke ``ragcore.backends.cpp.build_native()`` when
the extension is missing. The resulting shared object is written to
``ragcore/backends/_ragcore_cpp.*.so``.

To rebuild from scratch:

```bash
python -m ragcore.backends.cpp build --force
```

## Using the Backend

```python
from ragcore.backends import register_default_backends
from ragcore.backends.cpp import get_backend, is_available

if is_available():
    backend = get_backend()
    handle = backend.build({
        "backend": "cpp",
        "kind": "flat",
        "metric": "l2",
        "dim": 128,
    })
    # add vectors, search, etc.
```

If the native module is not available, importing ``ragcore.backends.cpp`` still succeeds, but
``ensure_available()`` raises an informative error until the extension is built.
