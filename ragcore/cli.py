from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from collections.abc import Sequence as SeqType
from pathlib import Path
from typing import Any

import numpy as np

from ragcore.backends import register_default_backends
from ragcore.backends.dummy import DummyBackend
from ragcore.backends.pyflat import PyFlatBackend

try:  # Optional C++ alias
    from ragcore.backends.cpp import CppFaissBackend
except ImportError:  # pragma: no cover - optional dependency
    CppFaissBackend = None  # type: ignore[assignment]

from ragcore.ingest.scanner import IngestedDocument, scan_corpus
from ragcore.interfaces import Backend, SerializedIndex
from ragcore.registry import get, list_backends, register


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectordb-builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list",
        help="List registered vector backends.",
    ).set_defaults(func=_cmd_list)

    build = subparsers.add_parser(
        "build",
        help="Build vector index artifacts from a corpus.",
    )
    build.add_argument(
        "--backend",
        required=True,
        help="Backend name to build the index with.",
    )
    build.add_argument(
        "--index-kind",
        default="ivf_flat",
        help="Index kind as defined by the backend.",
    )
    build.add_argument("--metric", default="ip", help="Similarity metric.")
    build.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Embedded vector dimension.",
    )
    build.add_argument("--nlist", type=int, help="IVF list count.")
    build.add_argument("--nprobe", type=int, help="IVF probe count.")
    build.add_argument("--pq-m", dest="pq_m", type=int, help="Product quantiser m parameter.")
    build.add_argument(
        "--pq-bits",
        dest="pq_bits",
        type=int,
        help="Product quantiser bits per sub-vector.",
    )
    build.add_argument("--M", dest="hnsw_M", type=int, help="HNSW M parameter.")
    build.add_argument(
        "--ef-construction",
        dest="hnsw_ef_construction",
        type=int,
        help="HNSW construction ef.",
    )
    build.add_argument(
        "--ef-search",
        dest="hnsw_ef_search",
        type=int,
        help="HNSW search ef.",
    )
    build.add_argument(
        "--train-vectors",
        type=Path,
        help="Optional path to training vectors (npy/npz).",
    )
    build.add_argument(
        "--add-vectors",
        type=Path,
        help="Optional path to vectors to add (npy/npz).",
    )
    build.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Path to corpus directory.",
    )
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for build artifacts.",
    )
    build.add_argument(
        "--accept-format",
        dest="accept_format",
        action="append",
        choices=["pdf", "md"],
        help="Accepted source formats when scanning --corpus-dir.",
    )
    build.set_defaults(func=_cmd_build)

    merge = subparsers.add_parser("merge", help="Merge shard directories into a single index.")
    merge.add_argument(
        "--merge",
        dest="merge",
        action="append",
        required=True,
        help="Path to a shard directory produced by 'build'.",
    )
    merge.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for merged artifacts.",
    )
    merge.set_defaults(func=_cmd_merge)

    return parser


def _cmd_list(_: argparse.Namespace) -> int:
    entries = []
    for name in list_backends():
        backend = get(name)
        _ensure_backend_protocol(name, backend)
        entries.append({"name": name, "capabilities": backend.capabilities()})
    print(json.dumps({"backends": entries}))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    backend = get(args.backend)
    _ensure_backend_protocol(args.backend, backend)

    spec: dict[str, Any] = {
        "backend": args.backend,
        "kind": args.index_kind,
        "metric": args.metric,
        "dim": args.dim,
    }

    params = _collect_params(args)
    if params:
        spec["params"] = params

    handle = backend.build(spec)

    documents = scan_corpus(args.corpus_dir, accept_formats=args.accept_format)
    docmap = _write_docmap(args.corpus_dir, documents, args.out)

    vectors = _embed_documents(documents, dim=args.dim)
    ids = np.arange(vectors.shape[0], dtype=np.int64)

    if handle.requires_training() and vectors.size:
        handle.train(vectors)

    if vectors.size:
        handle.add(vectors, ids)

    serialized = handle.serialize_cpu()
    _write_index_payload(serialized, args.out)
    _write_shards(docmap, args.out)

    summary = {
        "documents": len(documents),
        "docmap_path": str((args.out / "docmap.json").resolve()),
    }
    print(json.dumps(summary))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    shard_paths = [Path(p) for p in args.merge]
    if not shard_paths:
        raise SystemExit("merge requires at least one --merge shard path")

    shard_payloads = [_load_shard(path) for path in shard_paths]
    base_spec = shard_payloads[0]["spec"]
    backend_name = base_spec["backend"]

    backend = get(backend_name)
    _ensure_backend_protocol(backend_name, backend)

    merged_vectors: list[np.ndarray] = []
    merged_entries: list[dict[str, Any]] = []
    id_counts: dict[str, int] = {}
    vector_base = 0
    for _shard, payload in zip(shard_paths, shard_payloads, strict=False):
        spec = payload["spec"]
        if not _compatible_specs(base_spec, spec):
            raise ValueError("all shards must share backend, kind, metric, and dim")

        vectors = payload["vectors"]
        merged_vectors.append(vectors)

        for entry in payload["docmap"]:
            new_entry = dict(entry)
            base_id = new_entry.get("id", "")
            count = id_counts.get(base_id, 0)
            if count:
                new_entry["id"] = f"{base_id}-{count}"
            id_counts[base_id] = count + 1
            new_entry["vector_offset"] = vector_base + int(new_entry.get("vector_offset", 0))
            merged_entries.append(new_entry)

        vector_base += vectors.shape[0]

    if merged_vectors:
        all_vectors = np.concatenate(merged_vectors, axis=0)
    else:
        all_vectors = np.empty((0, base_spec["dim"]), dtype=np.float32)

    handle = backend.build({
        "backend": base_spec["backend"],
        "kind": base_spec["kind"],
        "metric": base_spec["metric"],
        "dim": base_spec["dim"],
    })
    if handle.requires_training() and all_vectors.size:
        handle.train(all_vectors)
    if all_vectors.size:
        handle.add(all_vectors)

    serialized = handle.serialize_cpu()
    metadata = dict(serialized.metadata)
    metadata.setdefault("merged_from", [str(path.resolve()) for path in shard_paths])
    serialized = SerializedIndex(
        spec=serialized.spec,
        vectors=serialized.vectors,
        ids=serialized.ids,
        metadata=metadata,
        is_trained=serialized.is_trained,
        is_gpu=serialized.is_gpu,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    _write_index_payload(serialized, args.out)
    _write_docmap_entries(merged_entries, args.out)
    _write_shards({"documents": merged_entries}, args.out)

    summary = {
        "documents": len(merged_entries),
        "docmap_path": str((args.out / "docmap.json").resolve()),
        "ntotal": int(serialized.vectors.shape[0]),
    }
    print(json.dumps(summary))
    return 0


def _collect_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if args.nlist is not None:
        params["nlist"] = args.nlist
    if args.nprobe is not None:
        params["nprobe"] = args.nprobe
    if args.pq_m is not None:
        params["m"] = args.pq_m
    if args.pq_bits is not None:
        params["nbits"] = args.pq_bits
    if args.hnsw_M is not None:
        params["M"] = args.hnsw_M
    if args.hnsw_ef_construction is not None:
        params["ef_construction"] = args.hnsw_ef_construction
    if args.hnsw_ef_search is not None:
        params["ef_search"] = args.hnsw_ef_search
    return params


def _build_docmap(corpus_dir: Path, documents: Sequence[IngestedDocument]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    offset = 0

    for record in documents:
        metadata = dict(record.metadata)
        doc_id_source = metadata.get("id") or metadata.get("slug") or record.path.stem
        doc_id = str(doc_id_source)
        counter = seen.get(doc_id)
        if counter is None:
            seen[doc_id] = 0
        else:
            seen[doc_id] = counter + 1
            doc_id = f"{doc_id}-{seen[doc_id]}"

        try:
            rel_path = record.path.relative_to(corpus_dir).as_posix()
        except ValueError:
            rel_path = record.path.resolve().as_posix()

        entries.append(
            {
                "id": doc_id,
                "path": rel_path,
                "format": record.format,
                "metadata": metadata,
                "text": record.text,
                "vector_offset": offset,
                "vector_count": 1,
            }
        )
        offset += 1

    return {"documents": entries}


def _write_docmap(
    corpus_dir: Path,
    documents: Sequence[IngestedDocument],
    out_dir: Path,
) -> dict[str, Any]:
    docmap = _build_docmap(corpus_dir, documents)
    _write_docmap_entries(docmap["documents"], out_dir)
    return docmap


def _write_docmap_entries(entries: SeqType[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"documents": list(entries)}
    docmap_path = out_dir / "docmap.json"
    docmap_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _embed_documents(documents: Sequence[IngestedDocument], *, dim: int) -> np.ndarray:
    if not documents:
        return np.empty((0, dim), dtype=np.float32)
    vectors = np.zeros((len(documents), dim), dtype=np.float32)
    for row, doc in enumerate(documents):
        buffer = doc.text.encode("utf-8")
        if not buffer:
            continue
        for idx, byte in enumerate(buffer):
            vectors[row, idx % dim] += byte / 255.0
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors


def _write_index_payload(serialized: SerializedIndex, out_dir: Path) -> None:
    spec_path = out_dir / "index_spec.json"
    spec_payload = dict(serialized.spec)
    spec_payload.update(
        {
            "is_trained": bool(serialized.is_trained),
            "is_gpu": bool(serialized.is_gpu),
            "metadata": dict(serialized.metadata),
            "ntotal": int(serialized.vectors.shape[0]),
        }
    )
    spec_path.write_text(json.dumps(spec_payload, indent=2, sort_keys=True), encoding="utf-8")

    index_bin_path = out_dir / "index.bin"
    with index_bin_path.open("wb") as buffer:
        np.savez(buffer, vectors=serialized.vectors, ids=serialized.ids)


def _write_shards(docmap: dict[str, Any], out_dir: Path) -> None:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shards_dir / "shard_0.jsonl"
    with shard_path.open("w", encoding="utf-8") as handle:
        for entry in docmap["documents"]:
            handle.write(json.dumps({"id": entry["id"], "path": entry["path"]}) + "\n")


def _load_shard(path: Path) -> dict[str, Any]:
    spec_path = path / "index_spec.json"
    docmap_path = path / "docmap.json"
    index_path = path / "index.bin"
    if not spec_path.exists() or not docmap_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"shard missing required files: {path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))["documents"]
    payload = np.load(index_path)
    vectors = payload["vectors"].astype(np.float32)
    return {"spec": spec, "docmap": docmap, "vectors": vectors}


def _compatible_specs(base: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    keys = ("backend", "kind", "metric", "dim")
    return all(base.get(key) == other.get(key) for key in keys)


def _ensure_backend_protocol(name: str, backend: Backend) -> None:
    if not isinstance(backend, Backend):  # type: ignore[arg-type]
        raise TypeError(f"backend '{name}' does not implement Backend protocol")


def main(argv: Sequence[str] | None = None) -> int:
    register_default_backends()
    if "py_flat" not in list_backends():
        register(PyFlatBackend())
    if "dummy" not in list_backends():
        register(DummyBackend())
    if CppFaissBackend is not None and "cpp_faiss" not in list_backends():
        register(CppFaissBackend())
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
