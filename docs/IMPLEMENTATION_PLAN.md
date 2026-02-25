# Drifting Models Reproduction Plan (PyTorch)

## 0) Objective and Success Criteria

### Primary objective
Reproduce the **Drifting Models** architecture and training pipeline from *Generative Modeling via Drifting* in PyTorch, with a codebase that is:
- faithful to the paper’s core algorithmic behavior,
- auditable (clear separation of paper-backed vs inferred decisions),
- scalable from toy to ImageNet-level experiments.

### Success criteria (tiered)

#### Tier A: Core correctness (must pass)
1. `compute_V` implementation satisfies required invariants (anti-symmetry structure, self-mask behavior, normalization behavior).
2. Stop-gradient drifting objective reproduces expected gradient direction and loss identities.
3. 2D toy experiments converge qualitatively as in paper (distribution evolution, anti-symmetry ablation failure).

#### Tier B: Architecture and training fidelity (must pass)
1. DiT-like generator with documented architecture deltas (conditioning, register tokens, style embeddings, RoPE/QK-norm/RMSNorm/SwiGLU).
2. Feature-space drifting pipeline with multi-feature extraction + normalization stack (A.5/A.6 behavior).
3. Training-time CFG mechanism via weighted unconditional negatives with alpha conditioning.

#### Tier C: Experimental reproduction (stretch)
1. ImageNet latent-space ablation trends qualitatively match paper tables (not necessarily exact numbers initially).
2. Pixel-space pipeline runs end-to-end with stable optimization.
3. System-level FID/IS approaches published ranges subject to compute/resource constraints.

---

## 1) Scope Boundaries

### In scope
- PyTorch implementation of generator, drifting field, loss stack, and training loops.
- Feature-encoder integration (pretrained and custom MAE paths).
- Reproducible configs + deterministic controls + evaluation scripts.
- Engineering for scale (DDP/FSDP-ready interfaces, mixed precision, memory-safe chunking).

### Out of scope (initially)
- Exact TPU-stack matching of the original implementation.
- Immediate claim of exact FID/IS parity without extensive hyperparameter sweeps.
- Production-level deployment tooling.

---

## 2) Evidence Hygiene: Requirement Classification

Every implementation decision must be tagged as one of:

- **PAPER-BACKED**: explicit in scanned paper/equations/tables.
- **INFERRED**: strongly implied but not explicit.
- **OPEN**: ambiguous/inconsistent in source; requires controlled decision + ablation.

A decision log (`docs/decision_log.md`) is mandatory before scaling past Stage 2.

---

## 3) Workstreams

### WS-A: Drift Core (algorithmic kernel)
Deliverables:
- `compute_V` implementation matching Algorithm 2 semantics.
- optional switches for:
  - y-only normalization vs x+y normalization,
  - diagonal self-masking,
  - multi-temperature aggregation,
  - weighted unconditional negatives.

Acceptance tests:
1. Anti-symmetry structural test (at formula-level decomposition).
2. Equilibrium sanity test under matched empirical sets.
3. Self-mask test (diagonal negative suppression).
4. Numeric stability under bf16/fp16/fp32.

### WS-B: Objective + Gradient Path
Deliverables:
- Stop-grad drifting regression objective module.
- Explicit graph policy:
  - gradients through generated branch,
  - detached target branch,
  - configurable feature-encoder gradient mode.

Acceptance tests:
1. Batch loss identity checks (`mse(x, sg(x+V))` behavior).
2. Finite-difference directional checks around drift direction.
3. Memory profiling to confirm graph detachment policy.

### WS-C: Generator Architecture
Deliverables:
- DiT-like generator with configurable variants for latent/pixel protocols.
- conditioning stack (class, alpha, optional style embeddings).
- register/in-context token path.

Acceptance tests:
1. Shape and patchify/unpatchify consistency tests.
2. Conditioning-path ablations (disable each component).
3. Deterministic seed reproducibility across 2+ runs.

### WS-D: Feature Stack
Deliverables:
- multi-scale/multi-location vectorization from feature maps.
- feature normalization (`S_j`) and drift normalization (`lambda_j`).
- multi-temperature merge behavior.

Acceptance tests:
1. Scale invariance sanity under affine feature scaling.
2. Per-feature normalization stats logging.
3. Runtime/memory benchmark across feature subsets.

### WS-E: CFG-by-Negative-Mixing
Deliverables:
- alpha sampling distribution and alpha->w conversion.
- unconditional negative injection + weighting in logits and normalization.

Acceptance tests:
1. Effective-mixture consistency checks vs formula.
2. Alpha sweep smoke test for FID/IS trend direction.
3. Edge-case tests (`alpha=1`, high-alpha caps).

### WS-F: Data, Queueing, and Training Orchestration
Deliverables:
- class-conditional sampling with positive/unconditional pools.
- queue-based sampler fallback (A.8-style) + direct dataloader mode.
- train loop with grouped per-class drift computation.

Acceptance tests:
1. Sampling correctness (class purity, no replacement constraints).
2. Throughput/latency budget checks.
3. Distributed consistency tests for queue updates.

### WS-G: Feature Encoder Reproduction
Deliverables:
- plug-in API for external SSL encoders.
- custom ResNet-style MAE pretraining path (latent/pixel).
- optional classifier fine-tuning stage.

Acceptance tests:
1. MAE pretrain reconstruction learning curve sanity.
2. Encoder feature quality probes (distance distributions, separability).
3. Downstream impact in controlled ablations.

### WS-H: Evaluation + Reproducibility Ops
Deliverables:
- FID/IS generation pipeline (50k protocol configurable).
- qualitative sample dumps + nearest-neighbor audit tooling.
- experiment registry (config hash, git hash, env fingerprint).

Acceptance tests:
1. Eval determinism checks.
2. Metric script cross-validation against known baselines.
3. Artifact integrity checks (resume/reload consistency).

---

## 4) Stage Plan (Execution Order)

### Stage 0: Project Foundation and Environment
Goals:
- Establish reproducible Python environment with `uv`.
- Install latest torch builds and baseline dependencies.
- Initialize skeleton repo structure for modular implementation.

Exit criteria:
- `uv` venv active and lockable dependency state recorded.
- `torch` import + CUDA/MPS detection checks pass.

### Stage 1: Drift Core + Toy Validation
Goals:
- Implement `compute_V`, drifting objective, and toy trainer.
- Reproduce toy convergence and anti-symmetry ablation failure.

Exit criteria:
- Unit tests pass for core invariants.
- Toy artifacts generated + qualitative convergence verified.

### Stage 2: Generator Integration (Small Scale)
Goals:
- Implement DiT-like generator and integrate with drift loss.
- Train on a small image benchmark (or downscaled ImageNet subset).

Exit criteria:
- Stable training for fixed wallclock budget.
- Sanity samples improve over checkpoints.

### Stage 3: Feature-Space Drifting Stack
Goals:
- Implement multi-feature extraction/normalization pipeline.
- Integrate multi-temperature losses and feature-level logging.

Exit criteria:
- Feature stack tests pass.
- Ablation confirms richer feature sets improve proxy quality.

### Stage 4: CFG Mechanism + Grouped Batching + Queueing
Goals:
- Implement alpha-conditioned training-time CFG via weighted negatives.
- Grouped class-conditional batch semantics and queue sampling.

Exit criteria:
- Formula-level consistency checks pass.
- Alpha sweep exhibits expected qualitative behavior.

### Stage 5: Latent-Space ImageNet Pipeline
Goals:
- Integrate SD-VAE latent data path.
- Train ablation-scale and then scaled latent models.

Exit criteria:
- End-to-end latent training/evaluation runs complete.
- Ablation trend alignment vs paper tables.

### Stage 6: MAE Encoder Reproduction
Goals:
- Pretrain latent-MAE, optional classifier fine-tune.
- Replace generic feature extractor with custom encoder.

Exit criteria:
- Encoder pretrain metrics stable.
- Downstream drifting quality improves in controlled experiments.

### Stage 7: Pixel-Space Pipeline
Goals:
- Enable pixel-space generator + feature stack.
- Integrate stronger feature encoders (e.g., ConvNeXt-V2 path if available).

Exit criteria:
- Pixel pipeline stable and measurable.
- System-level results tracked vs baseline references.

### Stage 8: Scale, Harden, and Report
Goals:
- Resolve open decisions with targeted ablations.
- Produce reproducible report package and runbook.

Exit criteria:
- Decision log fully closed or explicitly deferred.
- Reproduction report generated with transparent deltas.

---

## 5) Open Decisions (Known Ambiguities)

Each item must be resolved by decision record + experiment ID:

1. Pixel-space generator dimensional specifics vs table inconsistencies.
2. MAE encoder block-type inconsistency across sections/tables.
3. Exact interpretation/utility of “vanilla drifting loss without phi” in high-dimensional regimes.
4. Whether feature encoder weights are frozen during generator training (input-grad-only route).
5. Exact alpha embedding parameterization in generator conditioning.
6. Exact implementation variant of RoPE/QK-norm details (if under-specified).

Decision template fields:
- Context
- Candidate options
- Chosen option
- Why this option
- Validation evidence
- Residual risk

---

## 6) Test Strategy

### Unit tests
- math-level checks for `compute_V` and normalizations.
- tensor-shape tests for patch/token pipelines.
- deterministic checks for sampling utilities.

### Integration tests
- one-step train smoke tests (toy and mini image dataset).
- checkpoint save/resume exactness.
- distributed compatibility smoke test (single-node multi-GPU where possible).

### Regression tests
- fixed seed + fixed config quality proxy trend monitoring.
- performance regression thresholds (throughput/memory).

---

## 7) Instrumentation and Logging

Must log per-step/interval:
- drift norms (raw and normalized) per feature family and temperature.
- kernel entropy / effective neighborhood statistics.
- ratio of unconditional vs generated negative contributions.
- alpha samples distribution realized vs target.
- gradient norms per module.

Must log per-run:
- git hash, config hash, dependency snapshot, hardware summary.
- random seed and determinism flags.

---

## 8) Reliability and Failure Handling

### Expected failure modes
1. Flat kernels and vanishing drift in poor feature spaces.
2. Memory blowups from multi-location feature expansion.
3. Divergence from incorrect CFG weighting pipeline.
4. Silent anti-symmetry break due to asymmetric preprocessing.

### Mitigations
- runtime assertions for symmetry-sensitive transforms.
- chunked pairwise-distance computation path.
- strict config schema validation.
- pre-flight invariant checks before long runs.

---

## 9) Repository Layout (Target)

```text
.
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── toy/
│   ├── latent/
│   └── pixel/
├── docs/
│   ├── decision_log.md
│   ├── experiment_log.md
│   └── reproduction_report_template.md
├── drifting_models/
│   ├── __init__.py
│   ├── drift_field.py
│   ├── drift_loss.py
│   ├── features/
│   ├── models/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── utils/
├── scripts/
│   ├── setup_env.sh
│   ├── train_toy.py
│   ├── train_latent.py
│   ├── train_pixel.py
│   └── eval_fid_is.py
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

---

## 10) Immediate Execution Checklist (Now)

1. Create local `uv` venv.
2. Install latest PyTorch builds (and verify device backend visibility).
3. Initialize project skeleton directories/files.
4. Add decision and experiment log templates.
5. Run import/CLI sanity checks and capture environment fingerprint.

---

## 11) Definition of Done (Current Milestone)

Current milestone is complete when:
- this plan exists and is versioned in repo,
- local `uv` environment is live,
- latest torch packages installed and verified,
- skeleton for Stage 1 implementation is present,
- short execution report is produced.
