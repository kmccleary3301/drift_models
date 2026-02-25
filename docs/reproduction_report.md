# Drifting Models Reproduction Report (Work-in-Progress)

This report documents the current state of the PyTorch reproduction effort for *Generative Modeling via Drifting*.
It is intentionally conservative: it separates engineering scaffolding and smoke validation from any claim of paper-level metric parity.

## 0. Progress Accounting (Explicit Rubric)

### Stage weights (sum to 100)
These weights are used to compute the single “Overall completion %” reported below.

| Stage | Name | Weight |
| ---: | :--- | ---: |
| 0 | Foundation + environment | 5 |
| 1 | Drift core + toy validation | 10 |
| 2 | Generator integration (small scale) | 10 |
| 3 | Feature-space drifting stack | 10 |
| 4 | CFG + grouping + queueing | 15 |
| 5 | Latent-space ImageNet pipeline | 20 |
| 6 | MAE encoder reproduction | 10 |
| 7 | Pixel-space pipeline | 10 |
| 8 | Scale, harden, and report | 10 |

### “Done for now” criteria (per stage)
- Stage 0: `uv` env + torch import + tests pass.
- Stage 1: toy training runs + drift invariants tested.
- Stage 2: generator trains stably + deterministic behavior tested.
- Stage 3: feature loss path works + vectorization tested.
- Stage 4: queue + CFG mixing used + alpha controls logged.
- Stage 5: real latent protocol established + evaluated with pretrained Inception.
- Stage 6: MAE pretrain/export + downstream ablation evidence.
- Stage 7: experimental pixel protocol path established + evaluated with pretrained Inception.
- Stage 8: reproducible runbooks + report maps claims to artifacts.

### Historical completion snapshot (2026-02-14)
This table is retained as a historical scaffold snapshot and is **not** the current faithfulness gate status.

| Stage | Completion | Notes |
| ---: | ---: | :--- |
| 0 | 100% | env + CI-style tests passing |
| 1 | 100% | core invariants + toy validation + deterministic replay/anti-symmetry regression gates + long-horizon toy artifact bundle |
| 2 | 95% | generator stability run completed + parity audits/param-count verification + tiny overfit sanity mode (ablation matrix follow-up still pending) |
| 3 | 95% | feature stack integrated; strict-parity knobs added (raw+feature composition + temperature aggregation) and test-covered |
| 4 | 90% | queue + alpha controls validated; alpha→weight numeric coverage expanded; grouped sampling class-consistency test added |
| 5 | 100% | ImageNet train/val extracted; reference stats cached; train/val SD-VAE latents sharded; latent smoke + checkpointed ablation trend eval completed |
| 6 | 95% | MAE recipe knobs + classifier fine-tune hook implemented; short and longer-horizon ImageNet MAE variant impact runs completed |
| 7 | 85% | experimental pixel protocol (CIFAR-10) evaluated with pretrained Inception; paper encoder parity still open |
| 8 | 90% | runbooks + provenance pinning + pipeline runner + readiness benchmark + NN/grid audits + post-step400 eval bundle + disk-hygiene cleanup tooling + “golden artifacts” index |

**Historical overall completion (weighted)**: **94.5%**

> Note: this percentage reflects an older engineering/repro scaffolding snapshot, not a finalized paper-faithfulness claim.

### Current gate status (2026-02-21)
- P3 semantic blockers (MAE paper arch/taps + strict queue contract + provenance/eval contract hardening): **closed**.
- Long-horizon paper-facing latent evidence package: **in progress** (active long run at `59450/200000` in `outputs/ops/longrun_status_20260221_1710.json`; final claim artifacts still pending).
- Pixel paper-faithfulness path: **not closed** (pre-production package, scale2/scale3 paper-facing packages, and ablation-scale pretrained-cached package are complete, but paper-scale parity/evidence gates remain open; see `docs/pixel_scope_status.md`).

## 1. Scope
- Target claims reproduced (current):
  - End-to-end train -> sample -> eval plumbing for latent protocol and an experimental pixel protocol path.
  - Deterministic, resumable artifact generation (sampling outputs, cached reference stats).
  - Checkpoint metadata parity needed for reliable sampling/evaluation (config hash + model config).
- Scope exclusions (current):
  - Full-scale ImageNet runs and matching of paper FID/IS tables.
  - Exhaustive closure of open paper ambiguities via ablations.

## 2. Environment
- Python environment: `uv`-managed local `.venv` (see `uv.lock`)
- Torch stack: `torch 2.10.0+cu128` (queried 2026-02-13; see per-run `env_snapshot.json` artifacts for exact torchvision/torchaudio versions)
- Accelerator visibility: CUDA available in this environment (`torch.cuda.is_available() == True`)

## 3. Implementation Summary
- Drift field core: `drifting_models/drift_field.py` (`compute_v`, affinity construction, optional x-normalization, self-negative masking)
- Stop-grad objective / feature drifting: `drifting_models/drift_loss.py` and `drifting_models/train/`
- Generator: `drifting_models/models/` (DiT-like generator + config)
- Feature stack: `drifting_models/features/` (tiny encoders, latent decoder utilities)
- Sampling + postprocessing:
  - Pixel: `scripts/sample_pixel.py` + `drifting_models/sampling/`
  - Latent: `scripts/sample_latent.py` + optional ImageFolder decode path
- Evaluation: `scripts/eval_fid_is.py`
  - Supports ImageFolder and tensor-file sources, optional reference-stats caching, and stable init for `--inception-weights none`

## 4. Reproducibility Ops
- End-to-end runners:
  - Pixel: `scripts/run_end_to_end_pixel_eval.py`
  - Latent: `scripts/run_end_to_end_latent_eval.py`
- Alpha sweeps: `scripts/eval_alpha_sweep.py` (per-alpha sample+eval with shared reference cache)
- Last-K checkpoint evaluation: `scripts/eval_last_k_checkpoints.py`
- Runbooks:
  - Each run writes `RUN.md` alongside JSON summaries (`drifting_models/utils/run_md.py`)
  - Optional experiment-log appenders:
    - Script: `scripts/append_experiment_log.py`
    - Library: `drifting_models/utils/experiment_log.py`

## 4.1 CIFAR-10 Reference Protocol (Current)
- Layout: ImageFolder with class directories `0..9` containing `.png` images.
- Expected image size: `32x32` RGB.
- Canonical val reference export: `outputs/datasets/cifar10_val` (10k images).
- Quick-iteration subset: `outputs/stage8_cifar10/reference` (500 images; useful for fast iteration only).
- Evaluation notes:
  - “Comparable” FID/IS runs should use `--inception-weights pretrained`.
  - Reference-stat caching should be used (`--cache-reference-stats` / `--load-reference-stats`) to keep comparisons consistent across sweeps.

## 4.2 Metric Comparability Rules
- **Non-comparable (pipeline/CI only)**:
  - `scripts/eval_fid_is.py --inception-weights none` (randomly initialized Inception; useful for smoke tests and caching logic validation only).
- **Comparable within this repo**:
  - `scripts/eval_fid_is.py --inception-weights pretrained` with a documented reference dataset export and (ideally) cached reference stats reused across all comparisons.
- **Dataset protocol matters**:
  - report FID/IS alongside the reference dataset path, sample count, and whether reference stats were cached or recomputed.

### 4.3 Claim-facing metric gate
- Only artifacts with `inception_weights=pretrained` and `metrics_validity=standard` are eligible for claim-facing summaries.
- Proxy/smoke artifacts (for example Stage-4 rung proxy evals using `inception_weights=none`) are explicitly excluded from claim-facing comparisons.
- Audit artifact:
  - `outputs/ops/claim_eval_contract_audit_20260220/claim_eval_contract_audit.md`

## 5. Experimental Results (Smoke-Level)
- Validation coverage is primarily via `pytest` integration tests under `tests/integration/`.
- Metric outputs produced with `--inception-weights none` are for pipeline validation only and are not comparable to published FID/IS.
- For a chronological log of pipeline validations, see `docs/experiment_log.md`.

### 5.1 Comparable CIFAR-10 pixel runs (pretrained Inception)
All results below use the canonical CIFAR-10 val ImageFolder export (`outputs/datasets/cifar10_val`, 10k images) and `--inception-weights pretrained`.

- **Pixel real-queue short run (500 steps, sample 5k)**:
  - Generated: `outputs/milestone70/pixel_cifar_short_samples/images`
  - Eval: `outputs/milestone70/pixel_cifar_short_eval/eval_pretrained.json` (`fid≈479.26`, `IS≈1.0249`)
- **Alpha sweep (step 500 checkpoint, sample 5k each)**:
  - Summary: `outputs/milestone70/pixel_cifar_short_alpha_sweep_pretrained/alpha_sweep_summary.json`
  - `alpha=1`: `fid≈479.04`, `IS≈1.0250`
  - `alpha=2`: `fid≈479.47`, `IS≈1.0249`
  - `alpha=3`: `fid≈479.88`, `IS≈1.0248`
- **Last-K checkpoint eval (K=5, sample 5k each, alpha=1)**:
  - Summary: `outputs/milestone70/pixel_cifar_short_lastk_pretrained/last_k_summary.json`
  - Steps: 300/350/400/450/500 with FID in ~`432..479` range (still very early-training quality)

### 5.2 Comparable CIFAR-10 SD-VAE latent runs (pretrained Inception)
These runs validate the paper-targeted SD-VAE tokenizer interface (32x32x4 latents) and decode-to-pixels evaluation path. CIFAR-10 images are upscaled to 256x256 before SD-VAE encoding for protocol wiring.

- **Reference export (256x256)**: `outputs/datasets/cifar10_val_256` (10k images)
- **SD-VAE latent export (32x32x4)**: `outputs/datasets/cifar10_val_sdvae_latents.pt`
- **Latent short run (200 steps, queue, tensor_file)**:
  - Train artifacts: `outputs/milestone80/latent_cifar_sdvae_short/latent_summary.json`
  - Checkpoints: `outputs/milestone80/latent_cifar_sdvae_short/checkpoints/`
- **Sample + eval (step 200 checkpoint, sample 5k)**:
  - Samples: `outputs/milestone80/latent_cifar_sdvae_short_samples_sdvae/images`
  - Eval: `outputs/milestone80/latent_cifar_sdvae_short_eval/eval_pretrained.json` (`fid≈368.24`, `IS≈1.2562`)
- **Alpha sweep (step 200 checkpoint, sample 2k each)**:
  - Summary: `outputs/milestone80/latent_cifar_sdvae_short_alpha_sweep_pretrained/alpha_sweep_summary.json`
  - `alpha=1`: `fid≈369.91`, `IS≈1.2534`
  - `alpha=2`: `fid≈369.61`, `IS≈1.2532`
  - `alpha=3`: `fid≈369.35`, `IS≈1.2532`
- **Last-K checkpoint eval (K=4, sample 2k each, alpha=1)**:
  - Summary: `outputs/milestone80/latent_cifar_sdvae_short_lastk_pretrained/last_k_summary.json`
  - Steps 50/100/150/200 show FID trending down from ~`529 -> 370` (short horizon; interpret cautiously)

### 5.3 MAE pretraining on real SD-VAE latents (scaffold-level)
- Training run: `outputs/milestone80/mae_cifar_sdvae_short/mae_summary.json`
- Encoder export: `outputs/milestone80/mae_cifar_sdvae_short/mae_encoder.pt`
- Latent drifting smoke with MAE features: `outputs/milestone80/latent_cifar_sdvae_smoke_mae_feature_eval/eval_pretrained.json` (`fid≈358.96`, `IS≈1.2559`)

### 5.4 ImageNet-1k latent smoke (pretrained Inception)
This run establishes paper-targeted ImageNet latent protocol wiring end-to-end (smoke horizon), including SD-VAE decode and comparable pretrained-Inception evaluation.

- Train artifacts: `outputs/imagenet/latent_smoke_mae/latent_summary.json` + `outputs/imagenet/latent_smoke_mae/checkpoint.pt`
- Samples (5k, SD-VAE decode to 256x256 JPG): `outputs/imagenet/latent_smoke_mae_samples_2026-02-10_183641/sample_summary.json`
- Eval (pretrained Inception, cached ImageNet val reference stats): `outputs/imagenet/latent_smoke_mae_eval_2026-02-10_183641/eval_pretrained.json` (`fid≈429.31`, `IS≈1.4032`)

### 5.5 ImageNet-1k latent ablation follow-up (checkpointed trend runs)
To move beyond pure smoke validation, we added a checkpointed ablation-horizon run and trend evaluation package:

- **Ablation-horizon train run (step 100)**:
  - Config: `configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2.yaml`
  - Checkpoints: `outputs/imagenet/latent_ablation_b2_300/checkpoints/checkpoint_step_00000050.pt`,
    `outputs/imagenet/latent_ablation_b2_300/checkpoints/checkpoint_step_00000100.pt`
- **Alpha sweep (step 50, 1k samples/alpha)**:
  - Summary: `outputs/imagenet/latent_ablation_b2_300_alpha_sweep_step50/alpha_sweep_summary.json`
  - `alpha=1.0`: `fid≈476.83`, `IS≈1.1261`
  - `alpha=1.5`: `fid≈477.03`, `IS≈1.1262`
  - `alpha=2.0`: `fid≈476.72`, `IS≈1.1264`
- **Last-K checkpoint eval (K=2, 1k samples/checkpoint)**:
  - Summary: `outputs/imagenet/latent_ablation_b2_300_lastk_step_eval/last_k_summary.json`
  - `step=50`: `fid≈476.83`, `IS≈1.1261`
  - `step=100`: `fid≈280.64`, `IS≈1.2375`
- **Qualitative + audit tooling artifacts**:
  - fixed-seed grids: `outputs/imagenet/latent_ablation_b2_300_fixed_seed_grids/fixed_seed_grid_summary.json`
  - nearest-neighbor audit: `outputs/imagenet/latent_smoke_mae_nn_audit.json`

### 5.6 ImageNet latent MAE classifier fine-tune + variant impact (controlled short-run)
Stage-6 follow-up added a classifier fine-tuning hook and measured how MAE encoder variants affect downstream latent generation quality under a fixed short protocol (30 train steps, 1k decoded samples, pretrained-Inception eval).

- **Classifier fine-tune run (`w64` MAE resume, `cls ft` enabled)**:
  - Artifacts: `outputs/imagenet/mae_variant_a_w64_clsft/mae_summary.json`,
    `outputs/imagenet/mae_variant_a_w64_clsft/mae_encoder_clsft.pt`
- **Controlled MAE-variant matrix**:
  - Summary: `outputs/imagenet/latent_mae_variant_impact/mae_variant_impact_summary.json`
  - Ranked by FID (lower is better):
    - `w96`: `FID≈362.33`, `IS≈1.3464`
    - `w64`: `FID≈370.28`, `IS≈1.0047`
    - `w64_clsft`: `FID≈378.39`, `IS≈1.0000`
- **Interpretation**:
  - At this short horizon, wider MAE features (`w96`) help most.
  - The current `cls ft` recipe did not improve short-run generation quality and should be treated as an ablation branch requiring longer-horizon re-evaluation.

### 5.7 ImageNet latent MAE variant impact (longer horizon, pinned `cuda:1`)
We re-ran the MAE-variant comparison at a longer horizon and larger eval set under fixed hardware placement (`--device cuda:1`) for consistency:

- **Protocol**:
  - train: `steps=100` per variant
  - sample: `5000` decoded samples per variant
  - eval: pretrained Inception with cached ImageNet val reference stats
  - artifacts root: `outputs/imagenet/latent_mae_variant_impact_long100_s5k_cuda1`
- **Results (ranked by FID)**:
  - `w64`: `FID≈369.78`, `IS≈1.0000` (**best**)
  - `w64_clsft`: `FID≈398.03`, `IS≈1.0026`
  - `w96`: `FID≈475.05`, `IS≈1.0024`
- **Summary artifacts**:
  - `outputs/imagenet/latent_mae_variant_impact_long100_s5k_cuda1/mae_variant_impact_summary.json`
  - `outputs/imagenet/latent_mae_variant_impact_long100_s5k_cuda1/mae_variant_impact_summary.md`
- **Interpretation**:
  - The short-horizon winner (`w96`) did **not** hold at this longer horizon.
  - Current evidence supports `w64` as the safer default MAE feature encoder for next latent scaling runs.

### 5.8 ImageNet latent post-run package (step-400 continuation, `cuda:1`)
After the continuation run completed to `step=400` (`outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt`), we generated a post-run evaluation package with consistent sampling/eval settings:

- **Last-K checkpoint eval (K=3, 2k samples/checkpoint)**:
  - Summary: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_lastk_eval_k3_s2k/last_k_summary.json`
  - `step=300`: `fid≈318.13`, `IS≈1.1796`
  - `step=350`: `fid≈342.08`, `IS≈1.2085`
  - `step=400`: `fid≈293.57`, `IS≈1.4601`
- **Alpha sweep at step 400 (2k samples/alpha)**:
  - Summary: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_alpha_sweep_s2k/alpha_sweep_summary.json`
  - `alpha=1.0`: `fid≈293.57`, `IS≈1.4601`
  - `alpha=1.5`: `fid≈293.53`, `IS≈1.4607` (best FID in this sweep)
  - `alpha=2.0`: `fid≈293.53`, `IS≈1.4616`
  - `alpha=2.5`: `fid≈293.58`, `IS≈1.4625`
  - `alpha=3.0`: `fid≈293.62`, `IS≈1.4631`
- **Qualitative + nearest-neighbor audit additions**:
  - fixed-seed grids (post-cleanup checkpoints): `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_fixed_seed_grids_postcleanup/fixed_seed_grid_summary.json`
  - nearest-neighbor audit (best-alpha branch): `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_alpha_sweep_s2k/alpha_1p5/nn_audit.json`
    - `mean_cosine≈0.3413`, `median≈0.3399`, `p95≈0.3919`, `label_match_rate≈0.0020`
- **Disk hygiene follow-up**:
  - pruned stale relaunch/debug artifacts + redundant early continuation checkpoints (150/200/250)
  - report: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_cleanup_20260211.md`
  - run dir reduced from `~17G` to `~9.4G`; `/mnt/drive_4` free space increased by ~`7G`

### 5.9 Stage-2 generator closure addendum (stability + architecture + overfit, `cuda:1`)
To tighten Stage-2 implementation confidence, we added three targeted artifacts after the step-400 latent bundle:

- **120-step generator stability run**:
  - summary: `outputs/stage2_generator_stability_120/latent_summary.json`
  - checkpoints: `outputs/stage2_generator_stability_120/checkpoints/checkpoint_step_00000040.pt`, `outputs/stage2_generator_stability_120/checkpoints/checkpoint_step_00000080.pt`, `outputs/stage2_generator_stability_120/checkpoints/checkpoint_step_00000120.pt`
  - integrity signal: queue underflows remained zero, metrics finite through step 120
- **Generator architecture ablation matrix (6 variants)**:
  - summary: `outputs/stage2_generator_arch_ablations_cuda1/generator_arch_ablation_summary.json`
  - markdown table: `outputs/stage2_generator_arch_ablations_cuda1/generator_arch_ablation_summary.md`
  - toggles covered: RoPE, QK-Norm, RMSNorm/LayerNorm, register tokens, style tokens
  - all variants converged without NaN/inf and logged comparable throughput/VRAM profiles
- **Tiny-set overfit sanity mode + gate**:
  - config: `configs/latent/stage2_tiny_overfit.yaml`
  - run summary: `outputs/stage2_tiny_overfit_cuda1/latent_summary.json`
  - executable check: `outputs/stage2_tiny_overfit_cuda1/overfit_check.json` (`passed=true`)
  - implementation hook: `scripts/train_latent.py --overfit-fixed-batch`

### 5.10 Table-8 latent proxy benchmark (post-restart readiness, `cuda:1`)
After host restart validation, we executed a short proxy benchmark pass for both paper Table-8 latent templates to refresh launch envelopes:

- **Benchmark artifacts**:
  - `outputs/imagenet/benchmarks/table8_latent_proxy_20260212_restart/table8_latent_benchmark_summary.json`
  - `outputs/imagenet/benchmarks/table8_latent_proxy_20260212_restart/table8_latent_benchmark_summary.md`
- **Protocol**:
  - `bench_steps=8`
  - proxy batch: `groups=4`, `neg=4`, `pos=4`, `unc=2`
  - queue-backed latent tensor shards (`real-batch-source=tensor_shards`)
- **Results**:
  - `DiT-B/2 latent template`: `mean_step_time_s≈3.5066`, `img/s≈5.0781`, `peak_vram≈7.63 GB`
  - `DiT-L/2 latent template`: `mean_step_time_s≈3.0938`, `img/s≈5.1850`, `peak_vram≈22.79 GB`
- **Interpretation**:
  - restart did not break training stack or data-path assumptions
  - current `L/2` proxy peak VRAM leaves substantial headroom on 49 GB GPUs, supporting next-step horizon expansion


### 5.11 Step-600 latent closure package (`cuda:1`, 2k eval protocol)
After the `step=600` ablation-horizon run completed, we executed a full post-run closure bundle using pretrained Inception and cached ImageNet val reference stats.

- **Last-K checkpoint eval (`K=4`)**:
  - summary: `outputs/imagenet/latent_ablation_b2_600_cuda1_w64_lastk_eval_k4_s2k/last_k_summary.json`
  - `step=450`: `FID≈323.95`, `IS≈1.2830`
  - `step=500`: `FID≈333.58`, `IS≈1.1352`
  - `step=550`: `FID≈286.16`, `IS≈1.3934` (best in this window)
  - `step=600`: `FID≈390.61`, `IS≈1.0288` (late degradation)
- **Alpha sweep at `step=600` (`2k` samples/alpha)**:
  - summary: `outputs/imagenet/latent_ablation_b2_600_cuda1_w64_alpha_sweep_s2k/alpha_sweep_summary.json`
  - `alpha=1.0`: `FID≈390.6109`, `IS≈1.02878`
  - `alpha=1.5`: `FID≈390.5757`, `IS≈1.02876`
  - `alpha=2.0`: `FID≈390.5074`, `IS≈1.02877`
  - `alpha=2.5`: `FID≈390.4180`, `IS≈1.02879`
  - `alpha=3.0`: `FID≈390.3497`, `IS≈1.02881` (best in sweep, negligible delta)
- **Fixed-seed checkpoint grids**:
  - summary: `outputs/imagenet/latent_ablation_b2_600_cuda1_w64_fixed_seed_grids/fixed_seed_grid_summary.json`
  - grids: `grid_00000450.png`, `grid_00000500.png`, `grid_00000550.png`, `grid_00000600.png`
- **Nearest-neighbor audit (`alpha=1.5`, 512 vs 10k)**:
  - report: `outputs/imagenet/latent_ablation_b2_600_cuda1_w64_alpha_sweep_s2k/alpha_1p5/nn_audit.json`
  - `mean_cosine≈0.2806`, `median≈0.2778`, `p95≈0.3090`, `label_match_rate≈0.0059`
- **Interpretation**:
  - latest-checkpoint quality regressed materially from `step=550` to `step=600`
  - alpha retuning in the tested range does not recover quality at `step=600`
  - duplicate-risk indicators remain low in this audit scale, but absolute generation quality is still far from paper-level


### 5.12 Recovery continuation from step-550 (`cuda:1`, step 700)
To investigate the quality collapse seen by `step=600`, we ran a continuation from `checkpoint_step_00000550.pt` to `step=700`, then executed a clean alpha-sweep rerun against the immutable final checkpoint artifact.

- **Training continuation artifacts**:
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64/latent_summary.json`
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64/checkpoints/checkpoint_step_00000700.pt`
- **Training telemetry**:
  - `mean_step_time_s≈6.90`, `img/s≈9.55`, `peak_vram≈23.70 GB`
  - terminal step log: `lr≈2.0e-4`, `grad_norm≈1.41`, `mean_drift_norm≈11.84`
- **Alpha sweep rerun at `step=700` (`2k` samples/alpha)**:
  - summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_alpha_sweep_s2k_rerun/alpha_sweep_summary.json`
  - `alpha=1.0`: `FID≈354.08`, `IS≈1.00838` (best FID in sweep)
  - `alpha=1.5`: `FID≈354.20`, `IS≈1.00843`
  - `alpha=2.0`: `FID≈354.31`, `IS≈1.00858`
  - `alpha=2.5`: `FID≈354.40`, `IS≈1.00864`
  - `alpha=3.0`: `FID≈354.51`, `IS≈1.00868`
- **Nearest-neighbor audit (`alpha=1.5`, rerun branch)**:
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_alpha_sweep_s2k_rerun/alpha_1p5/nn_audit.json`
  - `mean_cosine≈0.3274`, `median≈0.3276`, `p95≈0.3342`, `label_match_rate≈0.0078`
- **Implementation notes**:
  - initial chained post-run attempt hit a checkpoint write/read race when it targeted mutable `checkpoint.pt`; rerun was executed against immutable `checkpoint_step_00000700.pt`
  - this run used `--allow-resume-config-mismatch`, but optimizer state restore kept effective LR at `2e-4`; a new `--resume-model-only` mode was added to support true optimizer-reset ablations
- **Interpretation**:
  - recovery continuation avoids catastrophic failure but does not restore quality to the pre-collapse `step=550` level
  - alpha tuning remains low-leverage in this regime; optimization/training-state dynamics remain the primary suspect


### 5.13 Optimizer-reset recovery branch (`--resume-model-only`, step 700)
To isolate optimizer-state effects, we repeated the step-550 recovery continuation with model-only resume semantics (weights/step restored, optimizer/scheduler/rng reset from config).

- **Training artifacts**:
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly/latent_summary.json`
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly/checkpoints/checkpoint_step_00000700.pt`
- **Training telemetry**:
  - terminal log: `lr≈8e-5`, `grad_norm≈0.563`, `mean_drift_norm≈12.293`
  - `mean_step_time_s≈6.25`, `img/s≈10.24`, `peak_vram≈22.47 GB`
- **Alpha sweep at `step=700` (`2k` samples/alpha)**:
  - summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly_alpha_sweep_s2k/alpha_sweep_summary.json`
  - `alpha=1.0`: `FID≈332.18`, `IS≈1.02703`
  - `alpha=1.5`: `FID≈332.14`, `IS≈1.02678`
  - `alpha=2.0`: `FID≈332.13`, `IS≈1.02665`
  - `alpha=2.5`: `FID≈332.12`, `IS≈1.02644` (best FID in sweep)
  - `alpha=3.0`: `FID≈332.12`, `IS≈1.02625`
- **Nearest-neighbor audit (`alpha=1.5`)**:
  - `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly_alpha_sweep_s2k/alpha_1p5/nn_audit.json`
  - `mean_cosine≈0.1914`, `median≈0.1897`, `p95≈0.2052`, `label_match_rate≈0.0020`
- **Comparative interpretation (vs optimizer-restored recovery branch)**:
  - optimizer-reset branch improved FID materially (`~332` vs `~354` at similar horizon/sample protocol)
  - audit similarity statistics dropped substantially (`mean_cosine ~0.19` vs `~0.33`)
  - this strongly implicates optimization-state carryover as a major contributor to late-run degradation behavior

### 5.14 P3 faithfulness remediation status (2026-02-20)
This update closes the highest-severity implementation deltas flagged by critique rounds for feature-space faithfulness.

- **MAE architecture parity path added**:
  - `drifting_models/features/mae.py` now includes `paper_resnet34_unet` with ResNet-34 stage layout `[3,4,6,3]`, GN+ReLU blocks, and bilinear-upsample + concat-skip decoder stages.
- **Feature tap semantics aligned**:
  - MAE exposes `encode_feature_taps()` for paper-facing every-2-block + stage-final taps; latent/pixel MAE adapters route through this API.
- **Faithful template contract hardened**:
  - Table-8 faithful latent templates pin `mae-encoder-arch: paper_resnet34_unet`, `queue-strict-without-replacement: true`, and the full feature-loss inventory/composition flags (`include-patch4-stats: true`, `feature-include-raw-drift-loss: true`, `feature-raw-drift-loss-weight: 1.0`).
  - Width parity is now pinned per Table-8 latent columns via template contracts: ablation-default `feature-base-channels: 256` (`w256` MAE export), B/2 and L/2 `feature-base-channels: 640` (`w640` MAE export expectation).
  - CI contract coverage: `tests/unit/test_table8_faithful_config_contract.py`.
- **Queue strictness gate added**:
  - Optional no-replacement hard guard is now available via `--queue-strict-without-replacement`, with queue-capacity compatibility checks and underfilled-label backfill to required per-class count.
- **Evidence gate documentation added**:
  - `docs/faithfulness_evidence_requirements.md` defines required MAE + latent artifact bundles for paper-facing claims.

Current claim posture remains conservative: architecture/loss-faithfulness blockers are materially reduced, but final paper-faithful metric claims stay gated on full-scale rerun evidence and release-gate closure.
## 6. Deviations From Paper (Known)
- Evaluation scaffolding includes a deterministic "random Inception" mode (`--inception-weights none`) used for tests; published comparisons require `--inception-weights pretrained`.
- Several paper ambiguities are not yet closed via ablation-level evidence; see `docs/IMPLEMENTATION_PLAN.md` and `docs/decision_log.md`.

## 7. Next Work (High-Leverage)
- Keep `w64` as a closest-feasible engineering branch for constrained runs; reserve faithful-template claims for runs backed by width-matched MAE exports (`w256`/`w640`) and corresponding Table-8 templates.
- Extend ablation-horizon latent checkpoints beyond step 600 and repeat alpha/last-K trend analyses with larger sample counts and stronger quality-recovery controls.
- Run a true optimizer-reset continuation using `--resume-model-only` from `step=550` to isolate optimizer-state effects from architecture/data effects.
- Continue Stage-7 pixel protocol scaling and add paper-mapped encoder-path parity checks.

## 8. Claim-to-Evidence Map (Current)
- Consolidated matrix:
  - `docs/claim_to_evidence_matrix.md`
- Deviations and impact:
  - `docs/deviations_table.md`
- Drift field invariants and loss sanity:
  - `tests/unit/test_drift_field.py`
  - `tests/unit/test_drift_loss.py`
  - `tests/integration/test_toy_training.py`
- Generator determinism and shape plumbing:
  - `tests/unit/test_dit_like.py`
  - `tests/unit/test_dit_determinism.py`
- Checkpointing + resume + config-guard behavior:
  - `tests/integration/test_checkpoint_resume.py`
  - `tests/integration/test_resume_mismatch_guard.py`
  - `tests/integration/test_sample_pixel_config_guard.py`
  - `tests/unit/test_load_pixel_generator_from_checkpoint.py`
- Sampling artifacts:
  - Pixel ImageFolder sampler: `tests/integration/test_sample_pixel_smoke.py`
  - Alpha schedule determinism: `tests/unit/test_sample_pixel_schedules.py`
- Evaluation pipeline (FID/IS scaffold):
  - Smoke: `tests/integration/test_eval_fid_is_smoke.py`
  - Reference stat cache: `tests/integration/test_eval_fid_is_cache_roundtrip.py`
  - ImageFolder filtering: `tests/integration/test_eval_image_exts_filter.py`
- Orchestrators / runners:
  - Pixel end-to-end: `tests/integration/test_end_to_end_pixel_eval_smoke.py`
  - Latent end-to-end: `tests/integration/test_end_to_end_latent_eval_smoke.py`
  - Alpha sweep: `tests/integration/test_eval_alpha_sweep_smoke.py`
  - Last-K evaluation: `tests/integration/test_eval_last_k_checkpoints_smoke.py`
- MAE export + classifier fine-tune compatibility:
  - `tests/integration/test_mae_cls_ft_export.py`

## 9. Gaps and Open Questions (Not Yet Closed)
- Paper-level metric parity:
  - Comparable pretrained-Inception runs exist for CIFAR-10 pixel protocol, but they are short-horizon and not expected to approach paper tables.
  - SD-VAE latent protocol is now wired end-to-end (encode/decode/eval) on a small benchmark, but ImageNet-scale latent protocol (the paper's main regime) is not yet implemented/evaluated at scale.
- Dataset protocol parity:
  - Reference exports exist for CIFAR-10 and ImageNet latent eval; paper-scale ImageNet training horizons remain pending.
- Architecture/algorithm ambiguities:
  - See `docs/IMPLEMENTATION_PLAN.md` "Open Decisions" and `docs/decision_log.md`.
- Scale engineering:
  - Long-horizon stability, throughput, and memory behavior under realistic batch sizes is not yet characterized.
