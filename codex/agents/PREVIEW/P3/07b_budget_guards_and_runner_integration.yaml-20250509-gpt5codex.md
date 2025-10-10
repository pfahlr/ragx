# Phase 3 Preview – Budget guards integrated with FlowRunner

## Scope
- Add a reusable `PolicyTraceBridge` so PolicyStack trace events flow through the shared `TraceEventEmitter`.
- Extend `FlowRunner` with loop orchestration, loop-aware budget charging, and loop summary trace emission.
- Preserve adapter-driven execution while layering run/node policy push/pop and policy resolution traces.
- Attach preview `BudgetDecision` metadata to `BudgetBreachError` for downstream diagnostics.

## Test Strategy
- New branch-scoped tests under `codex/code/work/tests/work/` cover loop budget stops and policy trace propagation.
- Existing budget manager and integration tests gain assertions around attached decisions and remain deterministic.
- All tests will be executed with `pytest codex/code/work/tests -q` during implementation.

## Open Questions / Risks
- Loop termination currently relies on `max_iterations` or budget stops; additional stop criteria (LLM feedback) remain future work.
- Policy trace bridging assumes ownership of `PolicyStack._event_sink`; callers supplying custom sinks should continue working via downstream forwarding, but additional validation in integration environments is recommended.
