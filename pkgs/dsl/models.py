from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Minimal representation of a tool defined in the DSL globals section.

    The master specification only requires tags for policy resolution in this
    phase, so the descriptor intentionally stays small.  Additional fields can
    be added later without breaking callers.
    """

    name: str
    tags: tuple[str, ...]

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, object]) -> ToolDescriptor:
        """Construct a descriptor from a mapping loaded from YAML.

        Missing or non-iterable ``tags`` entries default to an empty tuple.
        """

        raw_tags = data.get("tags", ()) if isinstance(data, dict) else ()
        if isinstance(raw_tags, str):
            tags_iter: Iterable[str] = (raw_tags,)
        elif isinstance(raw_tags, Iterable):
            tags_iter = (str(tag) for tag in raw_tags)
        else:  # pragma: no cover - defensive fallback
            tags_iter = ()
        tags: tuple[str, ...] = tuple(tags_iter)
        return cls(name=name, tags=tags)
