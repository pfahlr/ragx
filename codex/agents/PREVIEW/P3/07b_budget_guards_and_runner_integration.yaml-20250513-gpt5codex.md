# Phase 3 Preview – Budget guards and runner integration

## Key Enhancements
- Loop-aware FlowRunner execution that enters loop scopes, emits iteration traces, and halts gracefully on budget stop outcomes.
- Validator-backed `TraceEventEmitter` so schema requirements can be enforced in tests before events are persisted.
- Hardened policy/budget interplay ensuring policy denials raise before any budget charge is committed.

## Planned Validation
- Unit tests verifying trace validator success/failure paths.
- Loop scope regression covering stop-on-breach behaviour, iteration metadata, and trace ordering.
- Policy violation regression ensuring run/node budgets remain uncharged and traces capture the violation event.
