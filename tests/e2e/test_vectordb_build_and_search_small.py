from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ragcore.backends.pyflat import PyFlatBackend


@pytest.fixture(scope="module")
def _ensure_numpy() -> None:
    pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def ensure_cpp_backend() -> None:
    module = importlib.import_module("ragcore.backends.cpp")
    if not module.is_available():  # type: ignore[attr-defined]
        module.build_native(force=True)  # type: ignore[attr-defined]
    module.ensure_available()  # type: ignore[attr-defined]


def _write(tmp_dir: Path, relative: str, content: str) -> Path:
    path = tmp_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_cli_lists_backends(_ensure_numpy: None, ensure_cpp_backend: None) -> None:
    cmd = [sys.executable, "-m", "ragcore.cli", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    names = {entry["name"] for entry in payload}
    assert "py_flat" in names
    assert "cpp_faiss" in names


def test_build_and_search_round_trip(tmp_path: Path, _ensure_numpy: None) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    _write(
        corpus_dir,
        "doc.md",
        """title: Example\n---\nExample body\n""",
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "ragcore.cli",
        "build",
        "--backend",
        "py_flat",
        "--corpus-dir",
        str(corpus_dir),
        "--out",
        str(out_dir),
        "--accept-format",
        "md",
        "--index-kind",
        "flat",
        "--metric",
        "l2",
        "--dim",
        "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    spec_path = out_dir / "index_spec.json"
    assert spec_path.exists(), "index_spec.json should be produced"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == "py_flat"
    assert spec["kind"] == "flat"

    backend = PyFlatBackend()
    handle = backend.build(spec)
    handle.add(
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        ids=np.array([100, 200, 300], dtype=np.int64),
    )

    query = np.array([[0.1, 0.9]], dtype=np.float32)
    results = handle.search(query, k=2)
    np.testing.assert_allclose(results["ids"], np.array([[200, 100]], dtype=np.int64))

    index_payload = json.loads((out_dir / "index.bin").read_text(encoding="utf-8"))
    restored = backend.deserialize_cpu(index_payload)
    assert restored.ntotal() == 0
    with pytest.raises(RuntimeError):
        restored.search(query, k=1)


def test_build_and_search_round_trip_cpp_backend(
    tmp_path: Path, _ensure_numpy: None, ensure_cpp_backend: None
) -> None:
    corpus_dir = tmp_path / "corpus_cpp"
    corpus_dir.mkdir()

    _write(
        corpus_dir,
        "doc.md",
        """title: Example\n---\nExample body\n""",
    )

    out_dir = tmp_path / "out_cpp"
    cmd = [
        sys.executable,
        "-m",
        "ragcore.cli",
        "build",
        "--backend",
        "cpp_faiss",
        "--corpus-dir",
        str(corpus_dir),
        "--out",
        str(out_dir),
        "--accept-format",
        "md",
        "--index-kind",
        "flat",
        "--metric",
        "l2",
        "--dim",
        "3",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    spec_path = out_dir / "index_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["backend"] == "cpp_faiss"
    assert spec["kind"] == "flat"

    cpp_module = importlib.import_module("ragcore.backends.cpp")
    backend = cpp_module.get_backend()  # type: ignore[attr-defined]
    handle = backend.build(spec)
    vectors = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    ids = np.array([1, 2, 3], dtype=np.int64)
    handle.add(vectors, ids=ids)

    query = np.array([[0.2, 0.8, 0.0]], dtype=np.float32)
    results = handle.search(query, k=2)
    assert results["ids"].shape == (1, 2)

    index_payload = json.loads((out_dir / "index.bin").read_text(encoding="utf-8"))
    assert index_payload["spec"]["backend"] == "cpp_faiss"
    assert isinstance(index_payload["metadata"], dict)
    assert index_payload["vectors"] == []
