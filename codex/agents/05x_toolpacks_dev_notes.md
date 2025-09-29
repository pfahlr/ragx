# 05x Feature Branch - features/05-toolpacks(-local)


## New CLI Operations, Class interfaces, 

These operations cover the capabilities delivered across tasks 05a–05j on the current feature branch.

### Toolpack Executor

- **Execute a python toolpack end-to-end:**

```python  
from pathlib import Path
from apps.toolpacks.loader import ToolpackLoader
    # Confirms schema validation, async support, deterministic caching (mutate result and re-run), and error handling
loader = ToolpackLoader(); loader.load_dir(Path("apps/mcp_server/toolpacks"))
executor = Executor()
result = executor.run_toolpack(loader.get("tool.echo"), {"text": "hello"})
```
    
- **Trigger execution errors:**

   run the executor on a **non-python toolpack** or supply **invalid input/output payloads** to observe `ToolpackExecutionError`

### Docs & Diagnostics

  - Review docs/toolpacks_loader.md and docs/toolpacks_executor.md for the consolidated invariants, failure semantics, and usage snippets—all synced with the spec.
  - Check logs for legacy shim warnings (Toolpack ... used legacy snake_case keys) or validation errors while loading malformed toolpacks.

### CLI / Scripts

  - Use `./scripts/ensure_green.sh` for the full lint/type/test gate; focus on unit suites covering loader (tests/unit/test_toolpacks_loader.py) and executor (tests/unit/
  test_toolpacks_exec_python.py).
  - Run targeted tests interactively, e.g.
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/pytest tests/unit/test_toolpacks_loader.py -k execution.

  These operations cover the capabilities delivered across tasks `05a–05j` on the current feature branch.

  
## Test Coverage 
Python scripts that require test coverage are located in `./ragcore` and `./apps`, running **pytest-cov** for both of these shows us we have 88% coverage for `./ragcore` and 98% coverage for `./apps`:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=<source directory> --cov-report=term-missing`


```bash
⬢ [rag] ❯ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=ragcore --cov-report=term-missing
xxx.sx.......................Xx.......................sssssss.........................x.....xx.............................................................................. [ 94%]
..........                                                                                                                                                                   [100%]
================================================================================== tests coverage ==================================================================================
_________________________________________________________________ coverage: platform linux, python 3.13.3-final-0 __________________________________________________________________

Name                                  Stmts   Miss  Cover   Missing
-------------------------------------------------------------------
ragcore/__init__.py                       2      0   100%
ragcore/backends/__init__.py             23      2    91%   19, 33
ragcore/backends/base.py                115     10    91%   21, 23, 32, 49, 78-79, 85, 97, 134, 147
ragcore/backends/cpp/__init__.py         45      4    91%   60-62, 76
ragcore/backends/cuvs.py                 23      3    87%   31, 33, 35
ragcore/backends/dummy/__init__.py       28      5    82%   34-35, 37-38, 40
ragcore/backends/faiss.py                25      3    88%   32, 34, 36
ragcore/backends/hnsw.py                 23      3    87%   30, 32, 34
ragcore/backends/pyflat/__init__.py      25      2    92%   31, 35
ragcore/cli.py                          225     22    90%   151, 162, 182, 198, 218, 227, 260, 262, 264, 266, 268, 270, 272, 288-289, 293-294, 331, 336, 377, 393, 410
ragcore/ingest/__init__.py                5      0   100%
ragcore/ingest/md_parser.py              63      7    89%   59, 68, 73-74, 77, 91, 98
ragcore/ingest/pdf_parser.py             25     18    28%   21-46
ragcore/ingest/scanner.py                50      8    84%   31, 36, 42, 73-75, 81-82
ragcore/interfaces.py                    56      1    98%   91
ragcore/registry.py                      23      1    96%   18
-------------------------------------------------------------------
TOTAL                                   756     89    88%
165 passed, 8 skipped, 8 xfailed, 1 xpassed in 12.56s


⬢ [rag] ❯ PYTEPYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=apps --cov-report=term-missing

xxx.sx.......................Xx.......................sssssss.........................x.....xx.............................................................................. [ 94%]
..........                                                                                                                                                                   [100%]
================================================================================== tests coverage ==================================================================================
_________________________________________________________________ coverage: platform linux, python 3.13.3-final-0 __________________________________________________________________

Name                         Stmts   Miss  Cover   Missing
----------------------------------------------------------
apps/__init__.py                 1      0   100%
apps/toolpacks/__init__.py       3      0   100%
apps/toolpacks/executor.py      76      8    89%   80-84, 101-102, 107, 121-122, 130
apps/toolpacks/loader.py       378      0   100%
----------------------------------------------------------
TOTAL                          458      8    98%
165 passed, 8 skipped, 8 xfailed, 1 xpassed in 4.71s

```
