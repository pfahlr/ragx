# Phase 3 Preview — Task 07b_budget_guards_and_runner_integration

## Overview
- Established immutable budget domain models (`CostSnapshot`, `BudgetSpec`, `BudgetBreach`) and a nested `BudgetManager` capable of propagating charges across run/loop/node scopes.
- Delivered a `TraceEventEmitter` that formats `budget_charge`, `budget_breach`, and `loop_summary` events through immutable payloads for downstream policy/budget sinks.
- Wired a synchronous `FlowRunner` that coordinates adapters, policy enforcement, and budget charging, halting loops deterministically on hard breaches.

## Scope & Goals
- Support soft vs hard breach semantics with structured stop reasons surfaced via `BudgetChargeResult`.
- Emit deterministic trace payloads consumable by PolicyStack and future telemetry pipelines.
- Exercise integration of PolicyStack checks with budget guards using deterministic adapter doubles.

## Key Modules
- `codex/code/07b_budget_guards_and_runner_integration.yaml/budget.py`
- `codex/code/07b_budget_guards_and_runner_integration.yaml/trace.py`
- `codex/code/07b_budget_guards_and_runner_integration.yaml/runner.py`
- `codex/code/07b_budget_guards_and_runner_integration.yaml/adapters.py`

## Test Plan
- Unit coverage for budget arithmetic and soft/hard breaches: `test_budget_manager.py`.
- Trace emitter payload immutability: `test_trace_emitter.py`.
- Runner integration with policy + budget stop conditions: `test_flow_runner_budget_integration.py`.
