import json
from hashlib import sha256

VOLATILE_FIELDS = {
    "ts", "timestamp", "durationMs", "runId", "traceId", "spanId", "requestId",
    "attempt", "attemptId", "sessionId", "processId", "pid", "tid"
}

def _mask(obj):
    if isinstance(obj, dict):
        return {k: ("<VOLATILE>" if k in VOLATILE_FIELDS else _mask(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    return obj

def canonicalize_events(events):
    masked = [_mask(e) for e in events]
    lines = [json.dumps(e, sort_keys=True, separators=(",", ":")) for e in masked]
    return lines

def hash_events(events) -> str:
    lines = canonicalize_events(events)
    h = sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
