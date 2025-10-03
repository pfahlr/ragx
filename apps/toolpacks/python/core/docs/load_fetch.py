from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def _parse_front_matter(document: str) -> tuple[dict[str, Any], str]:
    if not document.startswith("---\n"):
        return {}, document

    try:
        _, _, remainder = document.partition("---\n")
        front_matter_raw, _, body = remainder.partition("\n---\n")
        data = yaml.safe_load(front_matter_raw) or {}
        if not isinstance(data, dict):
            data = {}
        return data, body
    except Exception:  # pragma: no cover - defensive fallback
        return {}, document


def load_fetch(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(payload["path"]).expanduser().resolve()
    metadata_path = payload.get("metadata_path")

    document_text = path.read_text(encoding="utf-8")
    extracted_metadata, _ = _parse_front_matter(document_text)

    metadata: dict[str, Any] = {}
    if metadata_path:
        metadata_content = Path(str(metadata_path)).expanduser().resolve().read_text(encoding="utf-8")
        metadata = json.loads(metadata_content)
    metadata.update(extracted_metadata)

    metadata.setdefault("title", path.stem.replace("_", " ").title())
    metadata.setdefault("tags", [])
    metadata.setdefault("authors", [])
    metadata["source_path"] = str(path)
    metadata["characters"] = len(document_text)
    metadata["checksum"] = f"sha256:{hashlib.sha256(document_text.encode('utf-8')).hexdigest()}"

    return {
        "document": document_text,
        "metadata": metadata,
    }
