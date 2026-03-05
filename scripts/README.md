# Script surface map

This file defines which scripts are stable entrypoints and which are exploratory tooling.

## Stable entrypoints

Use these for claim-facing workflows and public reproduction.

- `runtime_preflight.py`
- `runtime_newcomer_smoke.py`
- `runtime_stable_lane.py`
- `train_toy.py`
- `train_latent.py`
- `sample_latent.py`
- `eval_fid_is.py`

Naming policy for stable entrypoints:

- stable entrypoint files use one of prefixes: `train_*`, `sample_*`, `eval_*`, `runtime_*`.

## Stable support scripts

- `cache_reference_stats.py`
- `export_sd_vae_latents_tensor_file.py`
- `prepare_imagenet1k_from_archives.py`
- `verify_tensor_shards_manifest.py`
- `summarize_runtime_preflight.py`
- `audit_run_bundle.py`
- `check_config_tiers.py`
- `check_script_surface_registry.py`
- `make_run_root.py`
- `validate_run_artifacts.py`
- `wait_and_run_latent_claim_bundle.py`
- `wait_and_verify_train_latents.py`

## Experimental families

These scripts are useful for research but are not default public paths.

- canonical experimental paths are grouped under `scripts/experimental/`:
  - `scripts/experimental/ablations/`
  - `scripts/experimental/benchmarks/`
  - `scripts/experimental/checks/`
  - `scripts/experimental/eval/`
  - `scripts/experimental/pipelines/`
  - `scripts/experimental/recovery/`
- compatibility wrappers remain at the original `scripts/*.py` paths for moved entrypoints.
- detailed mapping and examples are in `scripts/experimental/README.md`.

## Deprecation lifecycle

Status policy is tracked in `docs/deprecation_matrix.md`.

- `maintenance-only`: compatibility wrappers remain executable but should not be used for new automation.
- `deprecated`: wrappers print migration targets at runtime and are scheduled for removal after at least one tagged release cycle.
- top-level experimental script names should use `exp_*` or `ablation_*` when new top-level experimental files are introduced.
- all top-level script files must be registered in `scripts/script_surface_registry.json` and pass `scripts/check_script_surface_registry.py`.

Current deprecated wrappers:

- `check_feature_normalization_ab.py` -> `scripts/experimental/checks/feature_normalization_ab.py`
- `check_feature_x2_toggle_effect.py` -> `scripts/experimental/checks/feature_x2_toggle_effect.py`
- `run_pixel_mae_export_pipeline.py` -> `scripts/experimental/pipelines/pixel_mae_export.py`

## Usage rule

If a result is intended for public claims:

1. use only stable entrypoints
2. follow `docs/minimal_repro_imagenet256.md`
3. log the run in `docs/reproducibility_scoreboard.md`
