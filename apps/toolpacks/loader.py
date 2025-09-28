from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Toolpack:
    """Resolved toolpack configuration."""

    id: str
    version: str
    config: dict[str, Any]
    path: Path

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.config.get("input_schema", {})

    @property
    def output_schema(self) -> dict[str, Any]:
        return self.config.get("output_schema", {})


class ToolpackLoader:
    """Load and cache toolpack definitions from disk."""

    def __init__(self, *, toolpacks: dict[str, Toolpack], root: Path) -> None:
        self._toolpacks = toolpacks
        self._root = root

    @classmethod
    def load_dir(cls, directory: Path | str) -> ToolpackLoader:
        base_dir = Path(directory).expanduser().resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"Toolpacks directory does not exist: {base_dir}")

        toolpacks: dict[str, Toolpack] = {}
        schema_cache: dict[tuple[Path, str], Any] = {}

        for file_path in sorted(base_dir.rglob("*.tool.yaml")):
            config = _load_yaml(file_path)
            if not isinstance(config, dict):
                raise ValueError(f"Toolpack file must define a mapping: {file_path}")

            resolved = _resolve_refs(config, file_path.parent, schema_cache)

            tool_id = resolved.get("id")
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise ValueError(f"Toolpack {file_path} missing required field 'id'")

            version = resolved.get("version")
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"Toolpack {file_path} missing required field 'version'")

            if tool_id in toolpacks:
                raise ValueError(f"duplicate toolpack id '{tool_id}'")

            toolpacks[tool_id] = Toolpack(
                id=tool_id,
                version=version,
                config=resolved,
                path=file_path.resolve(),
            )

        return cls(toolpacks=toolpacks, root=base_dir)

    def list(self) -> list[Toolpack]:
        return [self._toolpacks[key] for key in sorted(self._toolpacks)]

    def get(self, tool_id: str) -> Toolpack:
        try:
            return self._toolpacks[tool_id]
        except KeyError as exc:
            raise KeyError(f"toolpack '{tool_id}' is not loaded") from exc

    def __contains__(self, tool_id: str) -> bool:  # pragma: no cover - convenience
        return tool_id in self._toolpacks


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_refs(node: Any, base_dir: Path, cache: dict[tuple[Path, str], Any]) -> Any:
    if isinstance(node, dict):
        if set(node.keys()) == {"$ref"} and isinstance(node.get("$ref"), str):
            ref = node["$ref"]
            return _load_ref(ref, base_dir, cache)
        return {key: _resolve_refs(value, base_dir, cache) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, base_dir, cache) for item in node]
    return node


def _load_ref(reference: str, base_dir: Path, cache: dict[tuple[Path, str], Any]) -> Any:
    path_part, fragment = _split_reference(reference)
    target_path = Path(path_part) if path_part else Path()
    if not target_path.is_absolute():
        target_path = (base_dir / target_path).resolve()

    cache_key = (target_path, fragment or "")
    if cache_key in cache:
        return copy.deepcopy(cache[cache_key])

    if not target_path.exists():
        raise FileNotFoundError(f"Referenced schema not found: {reference}")

    with target_path.open("r", encoding="utf-8") as handle:
        if target_path.suffix in {".yaml", ".yml"}:
            document = yaml.safe_load(handle) or {}
        else:
            document = json.load(handle)

    fragment_data = _apply_json_pointer(document, fragment) if fragment else document
    resolved = _resolve_refs(fragment_data, target_path.parent, cache)
    cache[cache_key] = copy.deepcopy(resolved)
    return copy.deepcopy(resolved)


def _split_reference(reference: str) -> tuple[str, str | None]:
    if "#" not in reference:
        return reference, None
    path_part, fragment = reference.split("#", 1)
    if not fragment:
        return path_part, None
    if not fragment.startswith("/"):
        fragment = "/" + fragment
    return path_part, fragment


def _apply_json_pointer(document: Any, pointer: str | None) -> Any:
    if not pointer:
        return document
    parts = pointer.lstrip("/").split("/") if pointer != "/" else []
    current = document
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            index = int(part)
            current = current[index]
        elif isinstance(current, Mapping):
            current = current[part]
        else:  # pragma: no cover - defensive
            raise KeyError(f"Cannot apply pointer '{pointer}' to non-container value")
    return current
