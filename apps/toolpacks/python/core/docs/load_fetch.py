from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_fetch(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Load a markdown document and optional metadata from disk."""

    path = Path(str(payload.get("path")))
    metadata_path = Path(str(payload.get("metadata_path")))

    document_contents = path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "document": {"path": str(path), "contents": document_contents},
        "metadata": metadata,
    }
