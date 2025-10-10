# Phase 3 Review — Task 07b Budget Guards & Runner Integration

## Checklist
- [x] Budget decisions honour mode + breach_action combinations in unit tests.
- [x] FlowRunner emits policy decisions before budget commits; traces assert ordering.
- [x] Policy denials halt execution prior to budget commit (`test_flow_runner_halts_on_policy_violation_before_commit`).
- [x] Soft budgets log warnings without stopping loops and propagate to results.
- [x] Public APIs summarised in PREVIEW artifact.

## Verification
- Unit tests: `pytest codex/code/task_07b_budget_guards_and_runner_integration/tests -q`.
- Manual spot-check of trace ordering and warning payloads in emitted events.

## Known Issues / Follow-ups
- Trace payload schema validation deferred; consider integrating jsonschema checks later.
- Runner currently registers scopes defensively; upstream manager could expose introspection helper.
