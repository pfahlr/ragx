# Vector Database Backends

This document outlines the simulated vector database backends provided by `ragcore`. The
implementations adhere to the `ragcore.backends.base.Handle` protocol so they can be used
interchangeably by higher-level orchestration code and tests.

## Common Capabilities

All backends rely on a shared in-memory implementation that stores vectors in NumPy arrays and
supports deterministic CPU serialisation. The handles expose:

- `requires_training()` – indicates whether training data must be supplied before ingesting
  vectors.
- `train(vectors)` – validates the training data and toggles the trained state.
- `add(vectors, ids=None)` – appends vectors to the index while optionally accepting external
  identifiers.
- `search(queries, k)` – performs exact nearest-neighbour search for the requested metric.
- `serialize_cpu()` – returns a `SerializedIndex` payload suitable for persistence.
- `merge_with(other)` – merges shards that share the same `IndexSpec`.
- `to_gpu(device=None)` – clones the handle and marks it as GPU-backed when supported.

### Distance Metrics

The following metrics are supported across every backend:

- `l2` – squared L2 distance.
- `ip` – inner product; search results are ordered by highest similarity.
- `cosine` – cosine distance, normalised and returned as `1 - cosine_similarity`.

## FAISS Backend

The FAISS backend (`FaissBackend`) simulates the behaviour of FAISS indexes. It supports three
index kinds:

| Kind       | Training Required | Notes                          |
|------------|-------------------|--------------------------------|
| `flat`     | No                | Brute-force exact search.      |
| `ivf_flat` | Yes               | Simulates centroid-based IVF.  |
| `ivf_pq`   | Yes               | Simulates IVF with PQ codebooks.|

The backend advertises GPU support; calling `to_gpu()` on a handle returns a clone flagged for a
specific CUDA device.

## HNSW Backend

`HnswBackend` exposes a graph-based index with deterministic reinsertion semantics. The Python
implementation keeps behaviour simple by reusing the shared in-memory store while preserving the
handle contract. Merging shards concatenates stored vectors and reassigns contiguous identifiers.

## cuVS Backend

`CuVSBackend` represents the GPU-first RAFT/cuVS indexes. Handles require training and expose the
same API as FAISS, including GPU promotion via `to_gpu()`. Serialisation always uses the CPU code
path so higher-level tooling can persist indexes without GPU dependencies.

## Registry

Backends register with `ragcore.registry`, which maintains a mapping from backend names to
instances. Call `ragcore.backends.register_default_backends()` to populate the registry with the
FAISS, HNSW, and cuVS implementations. The registry exposes `register`, `get`, and
`list_backends` helpers used by unit and end-to-end tests to orchestrate builds and searches.

