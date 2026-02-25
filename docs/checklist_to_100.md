# Checklist To 100% Overall Completion

This checklist is the “full closure” path for the reproduction plan in `docs/IMPLEMENTATION_PLAN.md` and the progress rubric in `docs/reproduction_report.md`.

Completion means:
- Latent and pixel protocols are reproduced at paper-relevant scale (or the closest feasible approximation, with explicit deltas).
- All paper-backed vs inferred vs open decisions are either closed with evidence or explicitly deferred with impact.
- End-to-end runs are reproducible from documented commands and artifacts.

## Stage 0: Foundation + environment (keep it 100%)
- [ ] Verify `uv sync --extra dev --extra sdvae --extra imagenet` is reproducible on a clean machine.
- [ ] Add per-run environment snapshot artifacts (GPU/driver/torch/CUDA/RAM/disk).
- [ ] Add codebase fingerprint artifacts for non-git environments.
- [ ] Document minimum disk requirements for:
  - latent-only workflow
  - pixel workflow
  - full 50k sample evaluation

## Stage 1: Drift core + toy validation (90% -> 100%)
- [ ] Create a longer-horizon toy config and store qualitative convergence artifacts.
- [ ] Add deterministic toy replay (fixed seed -> identical artifacts/hashes).
- [ ] Save toy baseline vs anti-symmetry ablation trajectories as artifacts.
- [ ] Add precision stability checks for `compute_V` across `fp32/bf16/fp16` (finite + bounded).
- [ ] Add stress tests for extreme kernel temperatures (NaN/inf guard).
- [ ] Add an “equilibrium sanity” test (positives == negatives -> drift ~0).

## Stage 2: Generator integration (85% -> 100%)
- [ ] Audit generator architecture vs paper appendix; tag each component as paper-backed/inferred/open.
- [ ] Implement any missing paper-backed generator components and add unit tests.
- [ ] Enforce checkpoint-sampling config parity (no hidden CLI overrides required).
- [ ] Add tiny-set overfit mode (ensure generator can overfit; sanity).
- [ ] Run a small ablation matrix toggling RoPE/QK-norm/RMSNorm/style/register tokens and log stability/throughput impacts.

## Stage 3: Feature-space drifting stack (80% -> 100%)
- [ ] Enumerate full feature inventory required by the paper (locations/scales/normalization).
- [ ] Implement missing paper-backed feature vectorization pieces.
- [ ] Add feature extraction shape tests per feature family and scale.
- [ ] Add feature normalization invariance tests (affine scaling behavior).
- [ ] Add a feature ablation runner and record comparable summaries.
- [ ] Ensure `S_j` and `lambda_j` per feature/temp are logged and non-degenerate.

## Stage 4: CFG + grouping + queueing (85% -> 100%)
- [ ] Map CFG math in the paper to exact code symbols and logged quantities.
- [ ] Add numeric test for alpha->unconditional weighting across multiple alpha values.
- [ ] Add grouped-batch semantics tests (positives class-consistent, negatives generated, unconditional unconditioned).
- [ ] Add queue correctness tests: warmup reaches capacity targets; underflows bounded.
- [ ] Add queue determinism test (single-process): fixed seed -> identical warmup report.
- [ ] Define a DDP queue update strategy (or explicit guard against DDP usage).
- [ ] Implement DDP support or enforce a clear single-process constraint for paper-scale runs.

## Stage 5: Latent-space ImageNet protocol (smoke complete -> paper-scale complete)
### Dataset provenance + protocol pinning
- [ ] Record MD5 for `data/ILSVRC2012_img_train.tar.1` and `data/ILSVRC2012_img_val.tar` in an artifact JSON.
- [ ] Pin SD-VAE provenance (repo + revision/commit) in run artifacts.
- [ ] Record Inception weight provenance (torchvision version + weight enum) in eval artifacts.

### Ablation-scale training
- [ ] Run ImageNet latent ablation-scale training at a meaningful horizon with checkpoints and evals.
- [ ] Run alpha sweep on a fixed checkpoint; document trend alignment or divergence.
- [ ] Run last-K checkpoint evaluation; document trend stability.

### Table 8 scale readiness
- [ ] Create exact Table 8 configs (per column) with explicit mapping to table entries.
- [ ] Add a throughput benchmark script per Table 8 config (step time, VRAM, RAM, disk, queue underflows).
- [ ] Choose and document the scaling strategy (single GPU vs multi-GPU vs DDP).

### Paper-scale latent run
- [ ] Train paper-scale (or closest feasible) L/2 latent model with correct alpha sampling and multi-temperature semantics.
- [ ] Generate 50k samples with SD-VAE decode (256x256).
- [ ] Evaluate FID/IS with pretrained Inception and cached reference stats.
- [ ] Record final metrics and artifacts in `docs/reproduction_report.md`.

### Audit tooling
- [ ] Add nearest-neighbor audit utility for generated ImageNet samples.
- [ ] Add qualitative fixed-seed grid sampler across checkpoints.

## Stage 6: MAE encoder reproduction (85% -> 100%)
- [ ] Implement MAE pretraining hyperparameters per paper appendix (batch, lr, EMA, aug, length).
- [ ] Ensure augmentations happen pre-VAE encode (latent MAE), and document it.
- [ ] Train multiple MAE widths and epoch counts; export each with embedded strict config metadata.
- [ ] Implement classifier fine-tuning (“cls ft”) and export the fine-tuned encoder variant.
- [ ] Quantify downstream impact of MAE variants on ImageNet latent generation quality (controlled runs).

## Stage 7: Pixel-space pipeline (85% -> 100%)
- [ ] Decide on disk-efficient ImageNet pixel data format (ImageFolder vs WebDataset) and document it.
- [ ] Implement a pixel data lifecycle pipeline (extract/shard/cache/cleanup).
- [ ] Implement paper-described pixel feature encoder paths (MAE and/or ConvNeXt-V2 if required).
- [ ] Validate pixel encoder grad flow and multi-temperature semantics.
- [ ] Run pixel ablation-scale training with checkpoints and evals.
- [ ] Train pixel paper-scale (or closest feasible) model and evaluate on 50k samples.

## Stage 8: Scale, harden, and report (70% -> 100%)
### Close open decisions
- [ ] For each open ambiguity: write a decision record + run targeted ablations, or explicitly defer with impact.

### Reproducibility package
- [ ] Write an end-to-end reproduction document (clean-room) covering env->data->train->sample->eval.
- [ ] Ensure every major entrypoint writes a complete `RUN.md` (commands, hashes, key paths).

### Claim-to-evidence hardening
- [ ] Build a claim-to-evidence table mapping paper claims/tables to exact artifacts.
- [ ] Add a deviations table with impact assessment.
- [ ] Add regression tests for any discovered “footguns” during scaling.

### Automation
- [ ] Add resume-friendly orchestrators for paper-scale latent and pixel protocols.
- [ ] Add a standard status/reporting script for progress + artifact presence.

## Final 100% gate
- [ ] Mark all stages as 100% in `docs/reproduction_report.md` with artifact-backed justification.
- [ ] Run a clean-room reproduction drill and record outcome.
- [ ] Publish the final report package (metrics + artifacts + decisions + runbooks).
