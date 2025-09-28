#!/bin/bash
for shard in /tmp/corpus_split/shard_*; do
out="/tmp/vector_shards/$(basename "$shard")"
python -m ragcore.cli build \
    --backend py_flat \
    --index-kind flat \
    --metric ip \
    --dim 768 \
    --corpus-dir "$shard" \
    --out "$out"
done
