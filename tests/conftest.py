from __future__ import annotations

import functools
import importlib.util
import inspect
import itertools
import os
import pathlib
import sys
import types
from collections.abc import Callable, Iterable, Sequence

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval" / "verification"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        "native toolchain/backends not available (set RAGX_NATIVE_OK=1 to enable)"
    ),
)

skip_if_no_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="GPU runtime not available (set RAGX_GPU_OK=1 to enable)"
)


# ---------------------------------------------------------------------------
# Hypothesis fallback stub
# ---------------------------------------------------------------------------
if importlib.util.find_spec("hypothesis") is None:  # pragma: no cover - fallback only

    class _Strategy:
        def __init__(self, samples: Sequence[str]) -> None:
            self._samples = list(samples)

        def filter(self, predicate: Callable[[str], bool]) -> _Strategy:
            filtered = [item for item in self._samples if predicate(item)]
            if not filtered:
                filtered = self._samples[:1] or [""]
            return _Strategy(filtered)

        def samples(self) -> list[str]:
            return list(self._samples)

    def _text(*, min_size: int = 0, max_size: int | None = None) -> _Strategy:
        base: list[str] = [
            "custom",
            "invalid-type",
            "foo",
            "bar",
            "null",
            "boolean",
            "object",
            "array",
            "number",
            "string",
            "integer",
        ]
        candidates = [s for s in base if len(s) >= min_size]
        if max_size is not None:
            candidates = [s for s in candidates if len(s) <= max_size]
        if not candidates:
            candidates = ["stub"]
        return _Strategy(candidates)

    def _product_samples(strategies: Sequence[_Strategy]) -> Iterable[tuple[str, ...]]:
        pools = [strategy.samples() for strategy in strategies]
        return itertools.product(*pools) if pools else [tuple()]

    def _given(*strategies: _Strategy) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def decorator(test_fn: Callable[..., None]) -> Callable[..., None]:
            @functools.wraps(test_fn)
            def wrapper(*args, **kwargs):
                for sample_values in _product_samples(strategies):
                    test_fn(*sample_values, *args, **kwargs)

            original_params = list(inspect.signature(test_fn).parameters.values())
            retained_params = original_params[len(strategies) :]
            wrapper.__signature__ = inspect.Signature(parameters=retained_params)
            return wrapper

        return decorator

    strategies_module = types.ModuleType("hypothesis.strategies")
    strategies_module.text = _text  # type: ignore[attr-defined]

    hypothesis_stub = types.ModuleType("hypothesis")
    hypothesis_stub.given = _given  # type: ignore[attr-defined]
    hypothesis_stub.strategies = strategies_module  # type: ignore[attr-defined]

    sys.modules.setdefault("hypothesis", hypothesis_stub)
    sys.modules.setdefault("hypothesis.strategies", strategies_module)
