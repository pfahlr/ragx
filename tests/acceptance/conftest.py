from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(".").resolve()


@pytest.fixture(scope="session")
def spec_path(repo_root: Path) -> Path:
    path = repo_root / "codex" / "specs" / "ragx_master_spec.yaml"
    if not path.exists():
        pytest.skip("Master spec missing at codex/specs/ragx_master_spec.yaml")
    return path
