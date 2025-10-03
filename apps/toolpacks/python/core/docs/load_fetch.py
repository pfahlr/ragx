from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("metadataPath must contain a JSON object")
    return data


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(payload["path"]).expanduser().resolve()
    encoding = str(payload.get("encoding", "utf-8"))
    metadata_path_value = payload.get("metadataPath")
    metadata_path = None
    if metadata_path_value:
        metadata_path = Path(metadata_path_value).expanduser().resolve()

    content = _read_text(path, encoding)
    metadata = _load_metadata(metadata_path)

    import hashlib

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return {
        "document": {
            "path": str(path),
            "content": content,
            "sha256": digest,
        },
        "metadata": metadata,
        "stats": {
            "bytes": len(content.encode("utf-8")),
        },
    }
