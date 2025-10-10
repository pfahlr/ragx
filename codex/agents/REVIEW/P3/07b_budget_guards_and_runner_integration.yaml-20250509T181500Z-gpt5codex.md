# Phase 3 Review – Budget guards and runner integration

## Checklist
- [x] Loop scopes stop on loop-specific budget breaches without aborting the run.
- [x] Policy denials emit `policy_violation` traces via the shared emitter and do not charge budgets.
- [x] Trace payloads remain immutable mapping proxies (verified through TraceEventEmitter semantics).
- [x] Legacy run/node budget behaviour stays green after loop enhancements.
- [x] New tests fail on the pre-change runner and pass with the implemented loop orchestration.

## Notes
- Bridged PolicyStack event sink by attaching a lightweight adapter (`_PolicyEventBridge`) so policy events reuse the shared trace channel.
- Loop summaries now include stop reason, iteration count, and optional breach metadata for downstream diagnostics.
