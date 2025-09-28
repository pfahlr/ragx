#!/bin/bash
python - <<'PY'
try:
  import sys
  import shutil
  from pathlib import Path
  corpus = Path("../../../eval/verification/rag_gold_corpus_neuroplasticity/docs")
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
except Exception as e: 
  print("in exception handler")
  #sys.stderr.write(e)
  sys.exit(1)
PY
exit $?
echo $?
