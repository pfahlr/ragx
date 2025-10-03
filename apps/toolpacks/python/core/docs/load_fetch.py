"""Local document loader for docs.load.fetch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def _read_metadata(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("metadata must decode to a mapping")
    return dict(data)


def load_fetch(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Load a document from disk with optional metadata."""

    target = Path(str(payload["path"])).expanduser()
    encoding = str(payload.get("encoding", "utf-8"))
    if not target.exists():
        raise FileNotFoundError(target)

    content = _read_text(target, encoding)
    line_count = content.count("\n") + (0 if content.endswith("\n") else 1 if content else 0)
    checksum = hashlib.sha256(content.encode(encoding)).hexdigest()

    metadata_path_raw = payload.get("metadata_path")
    metadata_path = Path(str(metadata_path_raw)).expanduser() if metadata_path_raw else None
    metadata = _read_metadata(metadata_path) if metadata_path else {}
    metadata.setdefault("path", str(target))

    document = {
        "path": str(target),
        "content": content,
        "encoding": encoding,
        "line_count": line_count,
        "checksum": checksum,
    }

    return {
        "document": document,
        "metadata": metadata,
    }


__all__ = ["load_fetch"]

