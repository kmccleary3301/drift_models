# Paper → Code Map (Generative Modeling via Drifting)

This document maps the paper’s core symbols/equations to their concrete implementations in this repo.
It is meant to be used as a “where is *that* in code?” index while validating faithfulness.

Source paper scan: `Drift_Models/Drift_Models.md`.

## 1) Key objects and notation

| Paper symbol | Meaning | Repo implementation |
| --- | --- | --- |
| \(f_\theta\) | generator / pushforward map | `drifting_models/models/` (e.g. `DiTLikeGenerator`) |
| \(\epsilon \sim p_\epsilon\) | prior noise input | `scripts/train_latent.py` / `drifting_models/train/stage2.py` noise sampling (`torch.randn`) |
| \(q = f_\# p_\epsilon\) | pushforward distribution | implicit via generator forward outputs (training and sampling scripts) |
| \(p_{\text{data}}\) | data distribution | `drifting_models/data/RealBatchProvider` + queue paths in `scripts/train_latent.py` |
| \(\phi(\cdot)\) | feature encoder for feature-space drifting | `drifting_models/features/` + feature extraction/vectorization in `drifting_models/features/vectorize.py` |
| \(\mathbf{V}_{p,q}(\cdot)\) | drifting field | `drifting_models/drift_field.compute_v()` |
| \(k(\cdot,\cdot)\) | kernel over samples | implemented implicitly as `softmax(-cdist/tau)` in `drifting_models/drift_field.compute_affinity_matrices()` |
| `stopgrad(·)` | stop-gradient target | `.detach()` in `drifting_models/drift_loss.drifting_stopgrad_loss()` and feature variant |

## 2) Equation-level map (core sections)

### Eq. (1) — Pushforward distribution
\(q = f_\# p_\epsilon\).

- Generator forward pass: `drifting_models/models/` (DiT-like model).
- Training-time sampling of \(\epsilon\): `scripts/train_latent.py` (noise_grouped) and `drifting_models/train/stage2.py` (flattening and generator call).

### Eq. (2) — Drifted sample evolution
\(\mathbf{x}_{i+1}=\mathbf{x}_i + \mathbf{V}_{p,q_i}(\mathbf{x}_i)\).

- Realized as a *target* in the stopgrad loss (Eq. (6)); we do not explicitly update samples in-place.
- Target construction: `drifting_models/drift_loss.drifting_stopgrad_loss()` sets `target = x + drift`.

### Prop. 3.1 / Eq. (3) — Anti-symmetry
\(\mathbf{V}_{p,q}(\mathbf{x})=-\mathbf{V}_{q,p}(\mathbf{x})\).

- The repo’s anti-symmetric construction is the “attraction minus repulsion” form:
  - `drifting_models/drift_field.compute_v()` returns `drift_pos - drift_neg`.
  - `drifting_models/drift_field.compute_drift_components()` computes weighted positive and negative contributions.
- Self-negative masking (to avoid degenerate self-repulsion within the generated batch) is implemented via:
  - `DriftFieldConfig.mask_self_negatives` and diagonal bump in `compute_affinity_matrices()`.

### Eq. (6) — Stop-gradient fixed-point loss
\(\mathcal{L}=\mathbb{E}[\|f_\theta(\epsilon) - \text{stopgrad}(f_\theta(\epsilon)+\mathbf{V}(\cdot))\|^2]\).

- Implemented in `drifting_models/drift_loss.drifting_stopgrad_loss()`:
  - computes drift components (via drift field),
  - constructs target `x + drift`,
  - detaches target if `stopgrad_target=True`,
  - uses `torch.nn.functional.mse_loss(x, target)`.

### Eq. (10)–(12) — Kernelized drifting field
The paper defines \(\mathbf{V}_{p,q} = \mathbf{V}^+_p - \mathbf{V}^-_q\) and uses a kernel
\(k(\mathbf{x},\mathbf{y}) = \exp(-\|\mathbf{x}-\mathbf{y}\|/\tau)\).

- Distances: `torch.cdist(x, y)` in `drifting_models/drift_field.compute_affinity_matrices()`.
- Kernel logits: `-(dist / temperature)` (softmax over logits forms normalized weights).
- Normalization:
  - row-normalization: `softmax(logits, dim=-1)` (over y-samples for each x).
  - optional “normalize over x”: `softmax(logits, dim=-2)` plus geometric mean `sqrt(row*col)`.
  - controlled by `DriftFieldConfig.normalize_over_x`.
- Drift assembly corresponds to the paper’s “normalized-kernel weighted” attraction/repulsion:
  - `weight_pos = affinity_pos * affinity_neg.sum(dim=1, keepdim=True)`
  - `weight_neg = affinity_neg * affinity_pos.sum(dim=1, keepdim=True)`
  - then `drift_pos = weight_pos @ y_pos`, `drift_neg = weight_neg @ y_neg`.

### Eq. (13)–(14) — Feature-space drifting loss
The paper applies drifting in feature space, optionally at multiple scales \(\phi_j\).

- Feature extraction: `drifting_models/features/vectorize.py` (`extract_feature_maps`, `vectorize_feature_maps`).
- Feature-space drifting and multi-stage aggregation:
  - `drifting_models/drift_loss.feature_space_drifting_loss()`
  - per-stage keys correspond to feature-map stages returned by the encoder adapter.
- Feature normalization scale \(S\) is computed in `drifting_models/drift_loss._normalize_features()` using
  positive distances plus negative distances (generated + optional unconditional) with optional log-weighting
  from CFG (`build_negative_log_weights`), then detached before scaling.
- Important repo knobs that must match paper intent:
  - feature normalization and drift normalization
  - temperature scaling by \(\sqrt{C}\) (config `scale_temperature_by_sqrt_channels`)
  - detach policy for positive/negative features

### Eq. (15)–(16) — CFG as training-time mixture
Paper describes preparing negatives using a mixture of generated negatives and unconditional (data) negatives.

- Unconditional negative *count* is controlled by:
  - `--unconditional-per-group` and queue sampling in `scripts/train_latent.py`.
- Unconditional negative *weighting* (mapping \(\alpha\) to an unconditional weight) is in:
  - `drifting_models/drift_field.cfg_alpha_to_unconditional_weight()`
  - `drifting_models.drift_field.build_negative_log_weights()` adds log-weights to the negative logits.
- Application site:
  - `drifting_models/train/stage2.py` builds `negative_log_weights` when unconditional negatives exist.

## 3) Algorithm-level map (training step)

### “Alg. 1 training step” (paper) → repo
- Primary step implementation: `drifting_models/train/stage2.py::grouped_drift_training_step()`.
- Batch construction, queue, and alpha sampling orchestration:
  - `scripts/train_latent.py` (grouped noise, positives via queue/provider, alpha sampling, unconditional mixing).
- Drift-field evaluation is invoked via:
  - `drifting_models/drift_loss` → `drifting_models.drift_field`.

### “Alg. 2 compute V” (paper) → repo
- Implemented by:
  - `drifting_models/drift_field.compute_affinity_matrices()`
  - `drifting_models/drift_field.compute_drift_components()`
  - `drifting_models/drift_field.compute_v()`

## 4) Validation hooks (where to look when checking parity)

- Drift invariants and anti-symmetry regressions: `tests/` (integration + unit tests).
- Resume semantics (model-only vs optimizer restore/reset): `tests/integration/`.
- Alpha/queue behavior and unconditional mixing: `scripts/train_latent.py` logs and `latent_summary.json`.
- Comparable eval protocol (pretrained Inception + cached ImageNet stats):
  - `scripts/eval_fid_is.py`, `scripts/cache_reference_stats.py`, `outputs/datasets/*reference_stats*.pt`.

## 5) Known paper→repo deltas to explicitly track

- The repo’s “normalize over x” softmax (geometric mean of row/col) should be treated as a paper-variant knob:
  - `DriftFieldConfig.normalize_over_x`.
- Self-negative masking is a stabilization choice:
  - `DriftFieldConfig.mask_self_negatives` and diagonal bump.
- CFG implementation is realized as weighted unconditional negatives rather than explicit sampling from an analytic mixture; ensure this matches the paper’s intended estimator (Appendix references).

## 6) Fidelity status matrix (critique-driven)

| Critique item | Status | Primary evidence |
| --- | --- | --- |
| P0.1 queue label-integrity semantics | complete | `tests/unit/test_queue_label_integrity.py`, `scripts/train_latent.py` |
| P0.2 feature-loss reduction semantics (`sum` vs `mean`) | complete | `tests/unit/test_feature_drift_loss.py`, `drifting_models/drift_loss.py` |
| P0.3 normalization semantics include negatives + unconditional weighting | complete | `tests/unit/test_feature_drift_loss.py`, `drifting_models/drift_loss.py` |
| P0.4 faithful feature inventory + raw-drift composition + MAE width parity flags | complete | `configs/latent/imagenet1k_sdvae_latents_table8_ablation_default_template.yaml`, `configs/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml`, `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`, `docs/table8_config_map.md`, `docs/decision_log.md` (DEC-0040) |
| P0.5 pixel-path claim discipline | complete | `README.md`, `docs/reproduction_report.md`, `docs/table8_config_map.md` |
| P3.1 MAE paper encoder/decoder semantics (A.3) | complete | `drifting_models/features/mae.py`, `tests/unit/test_paper_mae_arch.py` |
| P3.2 feature taps every-2-block + stage-final (A.5) | complete | `drifting_models/features/mae.py`, `tests/unit/test_paper_mae_arch.py` |
| P3.3 optional strict queue no-replacement guard | complete | `drifting_models/data/queue.py`, `tests/unit/test_data_queue.py`, `tests/integration/test_queue_strict_mode.py` |
| P3.4 dataset/tokenizer/eval provenance contract | complete | `docs/provenance_contract.md`, `tests/unit/test_provenance_contract.py`, `outputs/ops/provenance_bundle_20260220_001234.json` |
| P3.5 claim-facing eval contract gate (pretrained only) | complete | `scripts/audit_claim_eval_contract.py`, `docs/eval_contract.md`, `outputs/ops/claim_eval_contract_audit_20260220/claim_eval_contract_audit.md` |
| D2 rung 1–3 fast-signal ladder | complete | `outputs/stage4_signal_ladder/rung_summary.json`, `docs/experiment_log.md` |
| D2 rung 4 mini-long run + scheduled proxy eval | complete | `outputs/stage4_signal_ladder/rung4_completion.json`, `docs/experiment_log.md` |
| Long-run faithfulness evidence gate | gated/pending faithful-template horizon | `outputs/imagenet/paperscale_b2_closest_feasible_run_20260215_005727/PRE_PARITY_BASELINE.md`, `outputs/baselines/pre_parity_b2_20260219_144046/artifacts_manifest.md` |
