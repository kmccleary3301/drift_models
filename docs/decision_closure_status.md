# Decision Closure Status (Open / Ambiguous Items)

This file tracks closure status for remaining open/ambiguous decisions that affect paper-faithfulness scope.

## Status legend
- `closed`: implemented + tested + documented.
- `deferred-with-impact`: not yet closed; scope impact is explicit and claim-limited.

## Current status

| Decision topic | Source | Status | Scope impact |
| --- | --- | --- | --- |
| Pixel-space generator dimensional specifics vs paper tables | `docs/IMPLEMENTATION_PLAN.md` Open Decision #1 | deferred-with-impact | Pixel paper-faithful claims remain gated; pixel path stays experimental/closest-feasible. |
| MAE encoder architecture inconsistency across paper sections | `docs/IMPLEMENTATION_PLAN.md` Open Decision #2 | closed | `paper_resnet34_unet` path implemented and contract-pinned for faithful templates. |
| Interpretation of vanilla drifting term usage in high-D feature runs | `docs/IMPLEMENTATION_PLAN.md` Open Decision #3 | closed | Faithful Table-8 latent templates now pin `feature-include-raw-drift-loss: true` with `feature-raw-drift-loss-weight: 1.0`; non-faithful tiers may still ablate explicitly. |
| Table-8 latent MAE width specification by column | P4 faithfulness remediation | closed | Faithful templates now pin width-parity fields (`feature-base-channels`: `256`/`640`) and matching MAE export-path expectations (`w256`/`w640`) via contract tests. |
| Feature encoder freeze policy (full freeze vs selective gradients) | `docs/IMPLEMENTATION_PLAN.md` Open Decision #4 | closed | Current training paths use explicit frozen feature encoders for latent/pixel feature losses. |
| Exact alpha embedding parameterization in generator conditioning | `docs/IMPLEMENTATION_PLAN.md` Open Decision #5 | deferred-with-impact | Absolute parity claims on conditioning micro-details remain gated; alpha sampling/weighting remains explicitly logged. |
| Exact RoPE/QK-Norm micro-variant details | `docs/IMPLEMENTATION_PLAN.md` Open Decision #6 | deferred-with-impact | Generator details are tracked as inferred parity with residual risk until appendix-level confirmation. |

## Policy

- Any `deferred-with-impact` row must be reflected in:
  - `docs/reproduction_report.md` deviations/claim scope wording.
  - `docs/paper_to_code_map.md` status rows where relevant.
  - `docs/release_gate_checklist.md` (Tier-A claim gating).
