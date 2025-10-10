# Phase 3 Preview — Budget Guards & FlowRunner Integration

## Overview
- Normalised budget domain objects (`BudgetSpec`, `CostSnapshot`, `BudgetChargeOutcome`) with deterministic math and immutable payloads.
- Centralised `BudgetManager` orchestrating run/loop/node/spec scopes with preview + charge lifecycles and stop/warn semantics.
- FlowRunner wiring adapters, budget manager, and trace emitter to halt loops on stop budgets, warn on soft breaches, and emit structured events.

## Key Design Decisions
- Treated `breach_action: stop` as inclusive (`remaining <= 0`) to stop loops exactly at the limit.
- Propagated immutable trace payloads via `TraceEventEmitter` to satisfy JSONL schema expectations.
- Captured hard-breach outcomes in `BudgetBreachError` to allow trace emission prior to aborting runs.

## Test Strategy
- Unit tests assert hard vs soft enforcement and preview immutability for `BudgetManager`.
- Integration test runs FlowRunner through loop iterations verifying stop reasons, warnings, adapter execution counts, and trace events.
