"""CLI entry-point for building the C++ backend stub."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import build_native


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Build the ragcore C++ backend stub")
    parser.add_argument("build", nargs="?", default="build", help=argparse.SUPPRESS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation even if up-to-date",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose build output")
    args = parser.parse_args(argv)

    artifact = build_native(force=args.force, verbose=args.verbose)
    print(f"Built {artifact}")
    return artifact


if __name__ == "__main__":  # pragma: no cover - exercised via tests/CLI
    main()
