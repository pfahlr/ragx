# Vector DB Interfaces

The `vector_db_core` component defines the canonical Python surface for working
with vector index backends. This document summarises the protocols, registry,
and CLI behaviour implemented in `ragcore` so other packages can consume the
APIs without reaching into backend internals.

## Protocols

Protocols are defined in `ragcore.interfaces` and mirror the requirements in
`codex/specs/ragx_master_spec.yaml`.

### Backend

```python
@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Mapping[str, Any]:
        ...

    def build(self, spec: Mapping[str, Any]) -> Handle:
        ...
```

* `name` is the registry key.
* `capabilities` returns a JSON-friendly description (kinds, metrics,
  GPU support, etc.).
* `build` receives an index specification (see below) and returns a `Handle`
  implementation ready for training/adding vectors.

### Handle

```python
@runtime_checkable
class Handle(Protocol):
    is_gpu: bool
    device: str | None

    def requires_training(self) -> bool: ...
    def train(self, vectors: FloatArray) -> None: ...
    def add(self, vectors: FloatArray, ids: IntArray | None = None) -> None: ...
    def search(
        self,
        queries: FloatArray,
        k: int,
        **kwargs: Any,
    ) -> Mapping[str, FloatArray | IntArray]: ...
    def ntotal(self) -> int: ...
    def serialize_cpu(self) -> SerializedIndex: ...
    def to_gpu(self, device: str | None = None) -> Handle: ...
    def merge_with(self, other: Handle) -> Handle: ...
    def spec(self) -> Mapping[str, Any]: ...
```

The handle tracks whether it currently resides on GPU (`is_gpu`, `device`) and
manages the standard FAISS-style lifecycle: train → add → search. All search
results return NumPy arrays for `ids` and `distances`.

### IndexSpec and SerializedIndex

`IndexSpec` validates inbound configuration and normalises it into a
`{backend, kind, metric, dim, params}` mapping. `SerializedIndex` stores the CPU
payload produced after `serialize_cpu()` with helper `to_dict()` for JSON
serialisation. Both live in `ragcore.interfaces` and are used by every backend
(including the dummy backend introduced for tests).

## Registry

`ragcore.registry` maintains a global mapping from backend name → instance.

```python
register(backend: Backend) -> None
get(name: str) -> Backend
list_backends() -> Tuple[str, ...]
_reset_registry() -> None  # tests only
```

* Registration enforces the `Backend` protocol at runtime and rejects duplicate
  names.
* `get()` raises `LookupError` with a descriptive message when the backend does
  not exist.
* `list_backends()` returns a sorted tuple of registered names for deterministic
  output.

`ragcore.backends.register_default_backends()` wires up the default set (dummy,
FAISS, HNSW, cuVS) so callers usually need to register only custom plug-ins.

## Dummy backend

`ragcore.backends.dummy.DummyBackend` is a pure-Python backend that implements
`kind=flat` with default metrics and exists only for smoke tests.

`ragcore.backends.pyflat.PyFlatBackend` is the baseline production-ready Python
implementation. It uses `VectorIndexHandle` under the hood, offering exact `L2`
and inner-product search, deterministic serialization, and merge support without
any native dependencies. The backend is registered by default so `vectordb-
builder` can target `--backend py_flat` out of the box.

## CLI (`vectordb-builder`)

`ragcore.cli` exposes two sub-commands:

* `list` – prints JSON describing all registered backends (name + capabilities).
* `build` – ingests a corpus, embeds documents with a deterministic hashing
  projection, and writes the CPU artefacts required by the spec.

Accepted builder flags are sourced from the spec (`--index-kind`, `--metric`,
`--dim`, HNSW/IVF parameters, `--accept-format`, etc.) plus `--backend` to choose
which registered backend to invoke.

Running `build` produces the index layout from the spec:

```
<out>/docmap.json
<out>/index_spec.json
<out>/index.bin
<out>/shards/shard_0.jsonl
```

* `docmap.json` – ingested documents with relative paths and metadata.
* `index_spec.json` – merged `IndexSpec` plus runtime metadata (`ntotal`,
  `is_trained`, `is_gpu`).
* `index.bin` – NumPy `.npz` archive containing `vectors` and `ids` arrays.
* `shards/` – deterministic shard manifest (currently a single JSONL file).

The build step can leverage any backend implementer because it only interacts
through the shared `Backend`/`Handle` protocols and the registry.
