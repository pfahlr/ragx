from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragcore import registry
from ragcore.backends import register_default_backends
from ragcore.backends.pyflat import PyFlatHandle
from ragcore.cli import main as cli_main


def setup_module() -> None:
    registry._reset_registry()


def teardown_module() -> None:
    registry._reset_registry()


def test_pyflat_cli_build_and_search(tmp_path: Path) -> None:
    register_default_backends()

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.md").write_text("# Title\nBody text.", encoding="utf-8")

    out_dir = tmp_path / "out"

    exit_code = cli_main(
        [
            "build",
            "--backend",
            "py_flat",
            "--corpus-dir",
            str(corpus_dir),
            "--out",
            str(out_dir),
            "--index-kind",
            "flat",
            "--metric",
            "ip",
            "--dim",
            "3",
        ]
    )

    assert exit_code == 0

    spec_path = out_dir / "index_spec.json"
    docmap_path = out_dir / "docmap.json"
    assert spec_path.exists()
    assert docmap_path.exists()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == "py_flat"
    assert spec["kind"] == "flat"
    assert spec["metric"] == "ip"

    backend = registry.get("py_flat")
    handle = backend.build(spec)

    base = np.eye(3, dtype="float32")
    handle.add(base)

    results = handle.search(base, k=1)
    assert np.array_equal(results["ids"].ravel(), np.arange(3))

    serialized = handle.serialize_cpu()
    clone = PyFlatHandle.from_serialized(serialized)
    round_trip = clone.search(np.array([[0.0, 1.0, 0.0]], dtype="float32"), k=1)
    assert np.array_equal(round_trip["ids"], np.array([[1]]))
