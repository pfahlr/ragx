from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

PyFlatBackend = import_module("ragcore.backends.pyflat").PyFlatBackend
main = import_module("ragcore.cli").main


@pytest.mark.parametrize("backend_name", ["py_flat", "cpp_faiss"])
def test_cli_build_with_pyflat_backend(tmp_path: Path, backend_name: str) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text("# Doc 1\n\nalpha beta", encoding="utf-8")
    (corpus / "doc2.md").write_text("Second document gamma", encoding="utf-8")

    out_dir = tmp_path / "out"

    exit_code = main(
        [
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
    )
    assert exit_code == 0

    index_spec_path = out_dir / "index_spec.json"
    docmap_path = out_dir / "docmap.json"
    index_bin_path = out_dir / "index.bin"

    assert index_spec_path.is_file()
    assert docmap_path.is_file()
    assert index_bin_path.is_file()

    spec = json.loads(index_spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == backend_name
    assert spec["metric"] == "ip"
    assert spec["kind"] == "flat"

    payload = np.load(index_bin_path)
    vectors = payload["vectors"].astype(np.float32)

    backend = PyFlatBackend()
    handle = backend.build(
        {
            "backend": "py_flat",
            "kind": "flat",
            "metric": spec["metric"],
            "dim": spec["dim"],
        }
    )
    handle.add(vectors)

    search = handle.search(vectors[:1], k=1)
    assert search["ids"][0, 0] == 0

    serialized = handle.serialize_cpu()
    np.testing.assert_allclose(serialized.vectors, vectors)
    np.testing.assert_array_equal(serialized.ids, np.array([0, 1], dtype=np.int64))

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))["documents"]
    assert all("vector_offset" in entry for entry in docmap)
