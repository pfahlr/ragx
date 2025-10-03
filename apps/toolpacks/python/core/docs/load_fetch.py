"""Local filesystem loader for docs.load.fetch."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def load_fetch(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(payload["path"]).expanduser()
    encoding = str(payload.get("encoding", "utf-8"))
    document = _read_text(path, encoding)

    metadata_path = payload.get("metadata_path")
    metadata: dict[str, Any] = {}
    if metadata_path:
        metadata_path = Path(str(metadata_path)).expanduser()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata.setdefault("source_path", str(path))
    checksum = hashlib.sha256(document.encode(encoding)).hexdigest()
    return {
        "document": document,
        "metadata": metadata,
        "checksum": checksum,
    }
