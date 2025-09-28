"""Command-line interface for the vector database builder."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from ragcore import registry
from ragcore.backends import DEFAULT_BACKENDS, register_default_backends
from ragcore.interfaces import IndexSpec, SerializedIndex

from .ingest.scanner import IngestedDocument, scan_corpus


def _available_backend_names() -> list[str]:
    return [backend_cls.name for backend_cls in DEFAULT_BACKENDS]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vectordb-builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available backends and their capabilities.")
    list_parser.set_defaults(func=_cmd_list)

    build = subparsers.add_parser("build", help="Build vector index artifacts from a corpus.")
    build.add_argument("--backend", choices=_available_backend_names(), default="dummy")
    build.add_argument("--corpus-dir", type=Path, required=True, help="Path to corpus directory.")
    build.add_argument("--out", type=Path, required=True, help="Output directory for build artifacts.")
    build.add_argument("--index-kind", dest="index_kind", default="ivf_flat")
    build.add_argument("--metric", choices=["l2", "ip"], default="ip")
    build.add_argument("--dim", type=int, default=384)
    build.add_argument("--nlist", type=int, default=2048)
    build.add_argument("--nprobe", type=int, default=32)
    build.add_argument("--pq-m", dest="pq_m", type=int, default=16)
    build.add_argument("--pq-bits", dest="pq_bits", type=int, default=8)
    build.add_argument("--M", dest="hnsw_M", type=int, default=32)
    build.add_argument("--ef-construction", dest="hnsw_ef_construction", type=int, default=200)
    build.add_argument("--ef-search", dest="hnsw_ef_search", type=int, default=64)
    build.add_argument("--train-vectors", dest="train_vectors", type=Path)
    build.add_argument("--add-vectors", dest="add_vectors", type=Path)
    build.add_argument("--merge", dest="merge", action="append", type=Path)
    build.add_argument("--faiss-threads", dest="faiss_threads", type=int, default=0)
    build.add_argument("--to-gpu", dest="to_gpu", type=int)
    build.add_argument(
        "--accept-format",
        dest="accept_format",
        action="append",
        choices=["pdf", "md"],
        help="Accepted source formats when scanning --corpus-dir.",
    )
    build.set_defaults(func=_cmd_build)

    return parser


def _cmd_list(_: argparse.Namespace) -> int:
    names = registry.list_backends()
    payload = []
    for name in names:
        backend = registry.get(name)
        payload.append({"name": name, "capabilities": backend.capabilities()})
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    backend = registry.get(args.backend)
    formats: Sequence[str] | None = args.accept_format
    if formats is None:
        formats = ["pdf", "md"]

    documents = scan_corpus(args.corpus_dir, accept_formats=formats)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    docmap = _build_docmap(args.corpus_dir, documents)
    docmap_path = out_dir / "docmap.json"
    docmap_path.write_text(json.dumps(docmap, indent=2, sort_keys=True), encoding="utf-8")

    index_spec = _construct_index_spec(args)
    handle = backend.build(index_spec)
    spec = IndexSpec.from_mapping(handle.spec())

    spec_path = out_dir / "index_spec.json"
    spec_path.write_text(json.dumps(spec.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    serialised = handle.serialize_cpu()
    _write_serialized_index(serialised, out_dir / "index.bin")
    (out_dir / "shards").mkdir(exist_ok=True)

    summary = {
        "documents": len(documents),
        "backend": args.backend,
        "docmap_path": str(docmap_path),
        "index_spec_path": str(spec_path),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _construct_index_spec(args: argparse.Namespace) -> Mapping[str, Any]:
    params: dict[str, Any] = {}
    for field in [
        ("nlist", args.nlist),
        ("nprobe", args.nprobe),
        ("pq_m", args.pq_m),
        ("pq_bits", args.pq_bits),
        ("hnsw_M", args.hnsw_M),
        ("hnsw_ef_construction", args.hnsw_ef_construction),
        ("hnsw_ef_search", args.hnsw_ef_search),
        ("faiss_threads", args.faiss_threads),
        ("to_gpu", args.to_gpu),
    ]:
        if field[1] is not None:
            params[field[0]] = field[1]

    spec: dict[str, Any] = {
        "backend": args.backend,
        "kind": args.index_kind,
        "metric": args.metric,
        "dim": args.dim,
    }
    if params:
        spec["params"] = params
    return spec


def _write_serialized_index(serialised: SerializedIndex, path: Path) -> None:
    payload = serialised.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_docmap(corpus_dir: Path, documents: Sequence[IngestedDocument]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for record in documents:
        metadata = dict(record.metadata)
        doc_id = str(metadata.get("id") or record.path.stem)
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
            }
        )

    return {"documents": entries}


def main(argv: Sequence[str] | None = None) -> int:
    register_default_backends()
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

