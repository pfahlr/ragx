from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragcore.cli import main
from ragcore.registry import _reset_registry, register


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    _reset_registry()
    yield
    _reset_registry()


@pytest.fixture(autouse=True)
def register_dummy_backend() -> None:
    from ragcore.backends.dummy import DummyBackend

    register(DummyBackend())


def test_list_command_outputs_registered_backends(capsys) -> None:
    exit_code = main(["list"])
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    names = [entry["name"] for entry in payload["backends"]]
    assert "dummy" in names


def test_build_command_creates_artifacts(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text("# Title\n\nContent one", encoding="utf-8")
    (corpus / "doc2.md").write_text("Second document", encoding="utf-8")

    out_dir = tmp_path / "out"

    args = [
        "build",
        "--backend",
        "dummy",
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

    exit_code = main(args)
    assert exit_code == 0

    stdout = capsys.readouterr().out.strip()
    summary = json.loads(stdout)
    assert summary["documents"] == 2

    docmap_path = out_dir / "docmap.json"
    index_spec_path = out_dir / "index_spec.json"
    index_bin_path = out_dir / "index.bin"
    shards_dir = out_dir / "shards"

    assert docmap_path.exists()
    assert index_spec_path.exists()
    assert index_bin_path.exists()
    assert shards_dir.is_dir()

    docmap = json.loads(docmap_path.read_text(encoding="utf-8"))
    assert len(docmap["documents"]) == 2

    index_spec = json.loads(index_spec_path.read_text(encoding="utf-8"))
    assert index_spec["backend"] == "dummy"
    assert index_spec["dim"] == 4

    payload = np.load(index_bin_path)
    assert payload["vectors"].shape == (2, 4)
    assert payload["ids"].tolist() == [0, 1]

