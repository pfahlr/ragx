# Phase 3 Preview — Budget Guards & Runner Integration

## Overview
- Introduced immutable budget domain models (`CostSnapshot`, `BudgetSpec`, `BudgetChargeOutcome`) with milliseconds normalization and arithmetic helpers.
- Implemented `BudgetManager` coordinating run/loop/node scopes, supporting preview/commit phases, warnings, and deterministic trace emission.
- Added `TraceEventEmitter` bridging policy and budget events through immutable payloads and optional recorders/sinks.
- Delivered a `FlowRunner` that orchestrates PolicyStack enforcement, ToolAdapters, and BudgetManager cooperation while emitting loop summaries and stop reasons.

## Test Coverage Targets
- `tests/test_budget_manager.py`: scope arithmetic, hard vs soft breaches, trace emission.
- `tests/test_trace_emitter.py`: payload immutability and policy bridge semantics.
- `tests/test_flow_runner.py`: adapter execution flow, policy enforcement, loop summaries, and run-stop behavior.
