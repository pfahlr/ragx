import os
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval" / "verification"

def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip() in {"1", "true", "yes", "on"}

NATIVE_AVAILABLE = _env_true("RAGX_NATIVE_OK")
GPU_AVAILABLE = _env_true("RAGX_GPU_OK")

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return REPO_ROOT

@pytest.fixture(scope="session")
def eval_dir() -> pathlib.Path:
    return EVAL_DIR

skip_if_no_eval = pytest.mark.skipif(
    not EVAL_DIR.exists(),
    reason="/eval/verification not found; provide gold corpus to run this test",
)

skip_if_no_native = pytest.mark.skipif(
    not NATIVE_AVAILABLE,
    reason=(
        "native toolchain/backends not available "
        "(set RAGX_NATIVE_OK=1 to enable)"
    ),
)

skip_if_no_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason=(
        "GPU runtime not available "
        "(set RAGX_GPU_OK=1 to enable)"
    ),
)
