# Contributing

Thanks for contributing to `drift-models`.

## Scope and contribution standards

This repository prioritizes:
- semantic correctness of drifting objective implementations,
- reproducibility and evidence-backed claims,
- test-backed changes over undocumented behavior shifts.

Before opening a PR, read:
- `docs/reproduction_report.md`
- `docs/eval_contract.md`
- `docs/release_gate_checklist.md`
- `docs/faithfulness_status.md`

## Development setup

```bash
uv sync --extra dev --extra eval --extra sdvae
uv run pytest -q
```

## Pull request requirements

1. Keep PRs focused; avoid unrelated refactors.
2. Add or update tests for behavioral changes.
3. Update affected docs when claim-scope or protocol changes.
4. Use clear PR descriptions with:
   - problem statement,
   - implementation summary,
   - validation commands + results.

For parity-critical changes (queue/drift/eval semantics), update:
- `docs/decision_log.md`
- `docs/reproduction_report.md`
- `docs/claim_to_evidence_matrix.md` (if claim mapping changes)

## Coding conventions

- Follow existing code style and naming patterns.
- Prefer explicit checks and informative error messages for invalid runtime configs.
- Do not introduce hidden behavior changes without tests.

## Issue reporting

For bugs, include:
- exact command,
- full error output,
- `python --version`, `torch.__version__`, device info,
- config path and modified flags,
- relevant run artifacts (`RUN.md`, summary JSON, eval JSON).

## Claim integrity

Do not claim paper-level faithfulness or paper-level metrics without explicit evidence artifacts and matching eval contracts.
