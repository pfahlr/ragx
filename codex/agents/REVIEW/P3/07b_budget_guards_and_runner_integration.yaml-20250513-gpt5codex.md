# Phase 3 Review Checklist – Budget guards and runner integration

## Functional Verification
- [ ] Loop scopes are entered/exited once and emit `loop_start`, iteration, and stop/complete events in deterministic order.
- [ ] Loop budget breaches surface as `loop_stop` traces without aborting the surrounding run.
- [ ] Policy violations raise `PolicyViolationError` before any budget commit and leave manager history at zero.
- [ ] `TraceEventEmitter.attach_validator` rejects malformed events prior to persistence and sink forwarding.

## Test Coverage
- [ ] `test_trace_emitter_validation.py` exercises validator success/failure paths.
- [ ] `test_flow_runner_loop_policy.py` covers loop stop semantics and policy/budget interplay.
- [ ] Existing integration tests continue to pass with iteration metadata added to `NodeExecution`.

## Code Quality
- [ ] `_run_unit_node` previews budgets before committing to avoid partial spend on loop stops.
- [ ] Trace payloads include iteration and loop identifiers when available.
- [ ] New helpers remain synchronous and deterministic for reproducible tests.
