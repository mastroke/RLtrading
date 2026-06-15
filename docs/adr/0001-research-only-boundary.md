# ADR 0001: Keep Trading Execution Out Of Scope

## Status

Accepted

## Context

AI trading projects can create reputational and compliance risk if they blur the line between research, paper trading and live execution.

## Decision

This project produces research reports and paper-trading candidates only. It does not connect to broker APIs or execute live orders.

## Consequences

- The repository demonstrates quant systems thinking without encouraging unsafe deployment.
- Risk guardrails are first-class and visible in tests.
- Live execution can be designed later as a separate, permissioned system.

