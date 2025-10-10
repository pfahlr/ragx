# Phase 3 Review – Budget guards integrated with FlowRunner

## Checklist
- [x] Policy trace bridge preserves downstream sinks and emits immutable payloads.
- [x] FlowRunner loop execution halts only loop scopes on stop decisions and surfaces `loop_summary` payloads.
- [x] Run/node policy push/pop balanced under success and exception paths.
- [x] Budget breach exceptions expose `decision` metadata for diagnostics and testing.
- [x] Branch-scoped tests cover loop stop, policy traces, and regression suites remain deterministic.

## Notes for Reviewers
- `PolicyTraceBridge` forwards to any pre-existing `PolicyStack` sink to avoid breaking external telemetry wiring.
- Loop handling commits run/node budgets only after loop preview indicates execution will proceed, preventing phantom spend.
- `loop_summary` events include the blocking outcome payload when applicable; see tests for expectations.
- Consider future follow-up to support additional loop termination criteria (e.g., until LLM response) if required by spec.
