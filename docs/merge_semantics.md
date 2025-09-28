# Merge Semantics

Vector indexes built with `vectordb-builder` can be merged deterministically as
long as the source shards share the same backend, kind, metric, and dimension.
This document captures the contract that the registry, handles, and CLI must
respect.

## Handle contract

* `merge_with(other)` returns a **new** handle whose vectors are the
  concatenation of `self` followed by `other`.
* The merged handle reassigns vector identifiers starting at zero and increasing
  by one (`ids == range(ntotal)`), regardless of the original offsets.
* Merging handles with mismatched specs (backend/kind/metric/dim) MUST raise
  `ValueError`.
* Merging honours metadata flags: `is_trained` is preserved if either shard was
  trained, and `is_gpu` remains `False` for the stub implementations.

The Python baseline (`py_flat`) and the C++ stub (`cpp_faiss`) both delegate to
this shared implementation and therefore have identical behaviour.

## Docmap offsets

Each document entry emitted by `vectordb-builder` now includes:

```json
{
  "id": "doc-id",
  "path": "relative/path.md",
  "format": "md",
  "metadata": { ... },
  "text": "…",
  "vector_offset": 12,
  "vector_count": 1
}
```

* `vector_offset` indicates the starting index (into `index.bin:vectors`) for
  the vectors derived from that document.
* `vector_count` is currently `1` for the flat baseline but must be preserved so
  future backends can emit multiple vectors per document.

During merge we adjust `vector_offset` by the cumulative size of previous
shards, ensuring offsets remain strictly increasing.

## CLI merge command

`vectordb-builder merge` consumes one or more shard directories (the outputs of
`vectordb-builder build`) and produces a single index directory containing:

* `index.bin` – stacked vectors with contiguous ids.
* `index_spec.json` – the spec from the first shard plus metadata about the
  source shard paths.
* `docmap.json` – a merged docmap with adjusted `vector_offset` values.
* `shards/shard_0.jsonl` – a manifest of document ids and relative paths.

Example invocation:

```bash
vectordb-builder merge \
  --merge /tmp/shard_a \
  --merge /tmp/shard_b \
  --out /tmp/merged_index
```

The command performs the following checks:

1. Each shard must contain `index_spec.json`, `index.bin`, and `docmap.json`.
2. All shard specs must match on backend/kind/metric/dim.
3. The backend is instantiated via the registry, so all registered aliases
   (`py_flat`, `cpp_faiss`, etc.) are supported.

The merged index is equivalent to constructing a fresh handle from the shared
spec and adding `np.concatenate` of all shard vectors.
