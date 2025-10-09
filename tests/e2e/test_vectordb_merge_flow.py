from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

main = import_module("ragcore.cli").main
get = import_module("ragcore.registry").get


@pytest.mark.parametrize("backend_name", ["py_flat"])
def test_vectordb_merge_flow(tmp_path: Path, backend_name: str) -> None:
    shard_dirs: list[Path] = []
    for shard_idx in range(2):
        corpus = tmp_path / f"corpus_{shard_idx}"
        corpus.mkdir()
        (corpus / "doc.md").write_text(f"Doc {shard_idx}", encoding="utf-8")

        out_dir = tmp_path / f"out_{shard_idx}"
        args = [
            "build",
            "--backend",
            backend_name,
            "--index-kind",
            "flat",
            "--metric",
            "ip",
            "--dim",
            "4",
            "--corpus-dir",
            str(corpus),
            "--out",
            str(out_dir),
        ]
        assert main(args) == 0
        shard_dirs.append(out_dir)

    merged_out = tmp_path / "merged"
    merge_args = [
        "merge",
        "--out",
        str(merged_out),
    ]
    for shard in shard_dirs:
        merge_args.extend(["--merge", str(shard)])

    assert main(merge_args) == 0

    index_spec_path = merged_out / "index_spec.json"
    docmap_path = merged_out / "docmap.json"
    index_bin_path = merged_out / "index.bin"
    manifest_path = merged_out / "shards" / "shard_0.jsonl"

    assert index_spec_path.exists()
    assert docmap_path.exists()
    assert index_bin_path.exists()
    assert manifest_path.exists()

    spec = json.loads(index_spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == backend_name
    assert spec["kind"] == "flat"
    metadata = spec.get("metadata", {})
    merged_from = metadata.get("merged_from", [])
    assert len(merged_from) == len(shard_dirs)

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))
    documents = docmap["documents"]
    assert len(documents) == 2
    offsets = [entry["vector_offset"] for entry in documents]
    counts = [entry["vector_count"] for entry in documents]
    ids = [entry["id"] for entry in documents]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert offsets[1] == offsets[0] + counts[0]
    assert counts == [1, 1]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("doc") and ids[1].startswith("doc")
    assert ids[0] != ids[1]

    payload = np.load(index_bin_path)
    vectors = payload["vectors"]
    assert vectors.shape[0] == 2

    backend = get(backend_name)
    handle = backend.build({
        "backend": backend_name,
        "kind": spec["kind"],
        "metric": spec["metric"],
        "dim": spec["dim"],
    })
    handle.add(vectors)
    search = handle.search(vectors[:1], k=1)
    assert search["ids"][0, 0] == 0
