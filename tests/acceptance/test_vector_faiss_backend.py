import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.xfail(reason="FAISS backend not wired yet")

def test_vdb_build_and_search_smoke(tmp_path: Path) -> None:
    # Prepare spec JSON
    spec = {
        "dim": 8,
        "metric": "ip",
        "kind": "flat",
        "params": {},
        "version": "v1",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    # Prepare random vectors
    xb = np.random.RandomState(0).rand(32, 8).astype("float32")
    add_path = tmp_path / "xb.npy"
    np.save(add_path, xb)
    # Run CLI (placeholder path ragcore.cli: vectordb-builder)
    cmd = [
        "python",
        "-m",
        "ragcore.cli",
        "build",
        "--backend",
        "dummy",
        "--spec",
        str(spec_path),
        "--add",
        str(add_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["ntotal"] == 32
    assert out["spec"]["dim"] == 8
