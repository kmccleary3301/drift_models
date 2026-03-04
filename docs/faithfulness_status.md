# Faithfulness Status

> Claim boundaries and paper-faithfulness posture for this reproduction.

---

## Current Status

| Indicator | Status | Description |
|-------------|-----------|----------------|
| Repository Type | Community reproduction | Not official author code |
| Latent Pipeline | Active hardening | Parity work in progress |
| Pixel Pipeline | Config parity | Paper-scale evidence pending |
| Full Results | Pending | Long-horizon runs required |

---

## Allowed Claims

| Claim | Rationale |
|----------|-------------|
| "Community PyTorch reproduction in progress" | Factually accurate |
| "Includes drifting objective, queue pipeline, and eval tooling" | Implemented and tested |
| "Closest-feasible single-GPU protocols" | Hardware-adapted but faithful |
| "Extensive smoke/testing coverage" | CI across platforms |
| "PyPI installable package" | `pip install drift-models` |

## Not Allowed Claims

| Claim | Why |
|----------|--------|
| "Paper-level metric reproduction is complete" | Training runs still in progress |
| "Paper-faithful parity is closed" | Known gaps documented |
| "Official implementation" | Not from paper authors |
| "Production-ready model" | Research code, not product |

---

## Evidence Sources

| Document | Link | Contains |
|-------------|---------|-------------|
| Reproduction Report | [reproduction_report.md](reproduction_report.md) | Current results vs. paper |
| Experiment Log | [experiment_log.md](experiment_log.md) | Training run history |
| Eval Contract | [eval_contract.md](eval_contract.md) | Measurement methodology |
| Claim Matrix | [claim_to_evidence_matrix.md](claim_to_evidence_matrix.md) | Specific claim mapping |

---

## Pipeline Status

| Pipeline | Status | Details |
|-------------|:---------:|------------|
| Latent | Active | Primary pipeline, under parity hardening |
| Pixel | Experimental | Config-parity pinned, evidence pending |
| MAE | Stable | Feature encoder working |
| Toy | Stable | Sanity check pipeline |

---

## Release Gates

| Checklist | Link | Purpose |
|-------------|---------|------------|
| Release Gate | [release_gate_checklist.md](release_gate_checklist.md) | Pre-release requirements |
| Evidence Requirements | [faithfulness_evidence_requirements.md](faithfulness_evidence_requirements.md) | What evidence is needed |

---

## Important Notes

> This repository tracks **mechanical faithfulness** — we implement the same algorithms, but results may differ due to:
> 
> - Hardware differences (single GPU vs. paper's TPU pod)
> - Training duration (shorter runs for validation)
> - Hyperparameter adaptations

---

<div align="center">

**Transparency is our policy** — see [claim_to_evidence_matrix.md](claim_to_evidence_matrix.md) for detailed mapping.

</div>
