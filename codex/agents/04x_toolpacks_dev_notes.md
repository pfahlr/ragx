# 00 - 04x Feature Branch - (00-04 scaffolding, development environment, tests, and vectordb)  features/05-vector-db(-local)

## New CLI Operations 

- Confirms the scaffolded directory layout from 00_scaffold_directories (apps/mcp_server, apps/toolpacks, etc.).
`tree -F -L 2 apps`

- One-shot CI gate wired up in 01/02: runs ruff, mypy, yamllint, and the entire pytest suite.
`./scripts/ensure_green.sh`

-Validates the glued Makefile targets (02_glue_makefile_and_scripts).
`make lint && make typecheck && make test`

- Exercises markdown ingestion utilities from 03_vectordb_md_ingest.
`python -m pytest tests/unit/test_md_front_matter_parse.py`

- CLI ingestion (04f) over the sample markdown corpus; /tmp/ragx-md-build gets index_spec.json, docmap.json, etc.
`python -m ragcore.cli build --backend py_flat --index-kind flat --metric ip --dim 4 --corpus-dir eval/verification/rag_gold_corpus_neuroplasticity  --out /tmp/ragx-md-build --accept-format md`

- Validates the shard merge workflow (04e); the merged directory will contain index.bin, docmap.json, and the shard manifest.
`python -m ragcore.cli merge --out /tmp/ragx-merged --merge fixtures/vector_shards/shard_a --merge fixtures/vector_shards/shard_b`

- Verifies protocol and registry behavior from 04a.
`python -m pytest tests/unit/test_vectordb_protocols.py tests/unit/test_registry.py`

- Flat index baseline plus C++ stub parity tests (04c / 04d).
`python -m pytest tests/unit/test_python_flat_index.py tests/unit/test_cpp_flat_parity.py`

- Confirms the optional C++ backend scaffold import/fallback logic (04b).
`python -m pytest tests/unit/test_cpp_stub_import.py`

- End-to-end CLI coverage for build/ingest/merge (03–04f).
`python -m pytest tests/e2e/test_vectordb_build_and_search_small.py tests/e2e/test_vectordb_build_md_fixture.py tests/e2e/test_vectordb_merge_flow.py`

- Toolpack loader

```python  
from pathlib import Path
from apps.toolpacks.loader import ToolpackLoader
loader = ToolpackLoader.load_dir(Path('fixtures/toolpacks'))
print([tp.id for tp in loader.list()])
```

- The first vector-database model has been implemented:  partition document collection / corpus into text "shards"

```python
# PARTITION THE CORPUS INTO SHARDS (store them in /tmp/corpus_split)
import shutil
from pathlib import Path

corpus = Path("./eval/verification/rag_gold_corpus_neuroplasticity/docs")
split_root = Path("/tmp/corpus_split")
chunk = 5

docs = [p for p in corpus.rglob("*") if p.is_file()]
docs.sort()

for idx in range(0, len(docs), chunk):
    shard_dir = split_root / f"shard_{idx // chunk:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for src in docs[idx:idx + chunk]:
        dest = shard_dir / src.relative_to(corpus)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
```
- process the text shards, embedding them into a vector database

```python
# BUILD THE SHARDS (for shard in /tmp/corpus_split run `ragcore.cli build` write them to /tmp/vector_shards)
for shard in "/tmp/corpus_split/shard_*; do
out="/tmp/vector_shards/$(basename "$shard")"
python -m ragcore.cli build \
    --backend py_flat \
    --index-kind flat \
    --metric ip \
    --dim 768 \
    --corpus-dir "$shard" \
    --out "$out"
done
```
- merge the vectorized shards into a single vector index

```bash
#!/bin/bash
python -m ragcore.cli merge \
    --out "/tmp/ragx-merged" \
    $(printf -- '--merge %s ' "/tmp/vector_shards/shard_*)
```
 - at this point  `/tmp/ragx-merged`  will contain a consolidated vectordb index `index.bin`.   additional metadata is left in `index_spec.json`,` docmap.json` and `shards/shard_0.jsonl`, with document offsets adjusted across shards.
