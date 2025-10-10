# Phase 3 Review Notes — Budget Guards and Runner Integration

## Checklist
- [x] BudgetManager preview/commit keeps state immutable until commit; duplicate scope registration guarded.
- [x] TraceEventEmitter emits `budget_charge`, `budget_breach`, and `policy_resolved` with mapping proxies and loop metadata.
- [x] FlowRunner clears scope caches per run, registers run/node/loop/spec budgets, and halts on hard breaches.
- [x] Soft node budgets accumulate warnings without halting execution.
- [x] Tests cover hard + soft scenarios and trace payload validation.

## Review Comments
- Consider deduplicating breach events for charges emitted during commit if preview already warned; current behaviour is acceptable but may double log warnings.
- Future enhancement: surface policy stack push/pop traces through the shared emitter for parity with budget events.
