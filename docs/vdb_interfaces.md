# Vector DB Interfaces

The vector database core exposes a stable Python surface described in
[`codex/specs/ragx_master_spec.yaml`](../codex/specs/ragx_master_spec.yaml).
This module documents the canonical protocols, registry helpers, and CLI
behaviour implemented in `ragcore`.

## Python Protocols

All backends implement the runtime-checkable `Backend` protocol defined in
`ragcore.interfaces`. A backend is responsible for reporting its
capabilities and constructing handles:

```python
from ragcore.interfaces import Backend, Handle, IndexSpec

class ExampleBackend:
    name = "example"

    def capabilities(self) -> Mapping[str, Any]:
        return {"name": self.name, "supports_gpu": False}

    def build(self, spec: Mapping[str, Any]) -> Handle:
        parsed = IndexSpec.from_mapping(spec, default_backend=self.name)
        return ExampleHandle(parsed)
```

The returned handle must satisfy the `Handle` protocol, which mirrors the
spec requirements: `train`, `add`, `search`, `ntotal`, `serialize_cpu`,
`to_gpu`, `merge_with`, and `spec`. The shared
`VectorIndexHandle` implementation in `ragcore.interfaces` provides an
in-memory reference suitable for pure-Python backends such as the
`dummy` backend used in tests.

## Registry

The registry (`ragcore.registry`) offers a lightweight discovery mechanism:

* `register(backend)`: add a backend instance (protocol-checked; duplicate
  names are rejected).
* `get(name)`: fetch a registered backend by name.
* `list_backends()`: enumerate registered backend names in sorted order.

`ragcore.backends.register_default_backends()` installs the built-in
backends (currently `dummy`, `faiss`, `hnsw`, and `cuvs`). The CLI always
invokes this helper before parsing commands to guarantee a predictable
environment for tools.

## CLI (`vectordb-builder`)

The CLI surfaces two sub-commands:

* `vectordb-builder list` – print a JSON array of registered backends and
  their `capabilities()` payloads.
* `vectordb-builder build` – scan a corpus directory, write `docmap.json`,
  materialise `index_spec.json`, and emit a serialised index payload
  (`index.bin`) using the selected backend. Flags map directly to the
  `vectordb_builder` entries in the spec, including `--backend` to choose
  the implementation and `--accept-format` to filter corpus ingestion.

The build output conforms to the `index_layout` contract in the spec: the
CLI writes `docmap.json`, `index_spec.json`, and an empty `shards/`
directory. The serialised index is stored as JSON for now; native
backends can replace this stub with a binary payload in later phases.

