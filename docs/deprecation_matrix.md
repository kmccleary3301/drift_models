# Deprecation Matrix

Date: 2026-03-04  
Scope: script/config compatibility paths retained during cleanup.

## Status definitions

- `active`: default, documented, and supported path.
- `maintenance-only`: retained for compatibility; no new features.
- `deprecated`: compatibility path scheduled for removal; runtime warning includes migration target.

## Script lifecycle matrix

### Active

- Stable claim-facing entrypoints:
  - `scripts/runtime_preflight.py`
  - `scripts/runtime_newcomer_smoke.py`
  - `scripts/runtime_stable_lane.py`
  - `scripts/train_toy.py`
  - `scripts/train_latent.py`
  - `scripts/sample_latent.py`
  - `scripts/eval_fid_is.py`
- Stable support entrypoints:
  - `scripts/cache_reference_stats.py`
  - `scripts/export_sd_vae_latents_tensor_file.py`
  - `scripts/prepare_imagenet1k_from_archives.py`
  - `scripts/wait_and_run_latent_claim_bundle.py`
- Canonical experimental entrypoints:
  - `scripts/experimental/ablations/*`
  - `scripts/experimental/benchmarks/*`
  - `scripts/experimental/checks/*`
  - `scripts/experimental/eval/*`
  - `scripts/experimental/pipelines/*`
  - `scripts/experimental/recovery/*`

### Maintenance-only (compatibility wrappers)

- `scripts/check_feature_loss_reduction_scaling.py`
- `scripts/check_latent_overfit.py`
- `scripts/eval_alpha_sweep.py`
- `scripts/eval_last_k_checkpoints.py`
- `scripts/run_end_to_end_latent_eval.py`
- `scripts/run_end_to_end_pixel_eval.py`
- `scripts/run_feature_drift_kernel_compile_benchmark.py`
- `scripts/run_feature_drift_temperature_reuse_benchmark.py`
- `scripts/run_feature_drift_vectorization_benchmark.py`
- `scripts/run_generator_arch_ablations.py`
- `scripts/run_imagenet_latent_pipeline.py`
- `scripts/run_latent_ablations.py`
- `scripts/run_latent_table8_benchmark.py`
- `scripts/run_mae_width_parity_exports.py`
- `scripts/run_pixel_feature_encoder_ablations.py`
- `scripts/run_pixel_paper_facing_package.py`
- `scripts/run_pixel_proxy_ablation_package.py`
- `scripts/run_pixel_queue_mae_export_compare.py`
- `scripts/run_queue_hotpath_benchmark.py`
- `scripts/summarize_recovery_matrix_2x2.py`

### Deprecated (warning emitted at runtime)

| Entrypoint | Migration target | Deprecated in | Remove no earlier than |
|---|---|---|---|
| `scripts/check_feature_normalization_ab.py` | `scripts/experimental/checks/feature_normalization_ab.py` | `v0.1.2` | `v0.3.0` |
| `scripts/check_feature_x2_toggle_effect.py` | `scripts/experimental/checks/feature_x2_toggle_effect.py` | `v0.1.2` | `v0.3.0` |
| `scripts/run_pixel_mae_export_pipeline.py` | `scripts/experimental/pipelines/pixel_mae_export.py` | `v0.1.2` | `v0.3.0` |

## Config lifecycle matrix

### Active

- Stable defaults:
  - `configs/toy/base.yaml`
  - `configs/toy/quick.yaml`
  - `configs/latent/smoke_feature.yaml`
  - `configs/latent/smoke_feature_queue.yaml`
  - `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`
  - `configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml`
  - `configs/mae/imagenet1k_sdvae_latents_shards_smoke.yaml`
- Stable alias folder:
  - `configs/stable/*`
- Canonical experimental folders:
  - `configs/experimental/latent/*`
  - `configs/experimental/pixel/*`

### Maintenance-only (compatibility wrappers)

- `configs/latent/ablation_feature_queue.yaml`
- `configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2.yaml`
- `configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step400_w64.yaml`
- `configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step600_w64.yaml`
- `configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_step400_w64.yaml`
- `configs/latent/imagenet1k_sdvae_latents_recovery_from550_cuda1_step700_w64.yaml`
- `configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml`
- `configs/latent/imagenet1k_sdvae_latents_table8_ablation_default_template.yaml`
- `configs/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml`
- `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`
- `configs/pixel/imagenet256_table8_b16_template.yaml`
- `configs/pixel/imagenet256_table8_l16_template.yaml`

### Deprecated

- none (as of 2026-03-04)

## Removal policy

1. Mark path as deprecated in this matrix.
2. Add runtime warning with explicit migration target for script entrypoints.
3. Keep deprecated path for at least one tagged release cycle.
4. At next eligible tagged release, either:
   - remove the path, or
   - explicitly move removal target forward with rationale.

Release validation must follow `docs/RELEASE_CHECKLIST.md`.
