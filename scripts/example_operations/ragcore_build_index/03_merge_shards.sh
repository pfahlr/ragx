#!/bin/bash
python -m ragcore.cli merge \
  --out "/tmp/ragx-merged" \
  $(printf -- '--merge %s ' /tmp/vector_shards/shard_*)
