# Stable vs experimental surface

This document is the support boundary for the repo.

If you are new, stay on the stable surface first. Experimental lanes are for research iteration and can change faster.

## Stable surface (recommended)

### Entry points

- `scripts/runtime_preflight.py`
- `scripts/runtime_newcomer_smoke.py`
- `scripts/runtime_stable_lane.py`
- `scripts/train_toy.py`
- `scripts/train_latent.py` (with stable configs below)
- `scripts/sample_latent.py`
- `scripts/eval_fid_is.py`

### Stable configs

- `configs/toy/quick.yaml`
- `configs/latent/smoke_feature.yaml`
- `configs/latent/smoke_feature_queue.yaml`
- `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`
- `configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml`
- `configs/mae/imagenet1k_sdvae_latents_shards_smoke.yaml`

### Stable support scripts

- `scripts/wait_and_run_latent_claim_bundle.py`
- `scripts/prepare_imagenet1k_from_archives.py`
- `scripts/export_sd_vae_latents_tensor_file.py`
- `scripts/cache_reference_stats.py`
- `scripts/validate_run_artifacts.py`

### Stable docs

- `docs/getting_started.md`
- `docs/minimal_repro_imagenet256.md`
- `docs/eval_contract.md`
- `docs/faithfulness_status.md`
- `docs/reproducibility_scoreboard.md`

## Experimental surface

These are useful for research, but are not the default public lane:

### Entry points and runners

- canonical experimental runners live under `scripts/experimental/*`.
- eval and orchestration families are under `scripts/experimental/eval/` and `scripts/experimental/pipelines/`.
- compatibility wrappers remain at legacy `scripts/*.py` paths for moved tools.
- path lifecycle status is tracked in `docs/deprecation_matrix.md`.
- `scripts/run_*ablation*.py`
- `scripts/run_*benchmark*.py`
- `scripts/run_pixel_*`
- `scripts/train_pixel.py`
- `scripts/train_mae.py` with non-smoke custom configs
- ad-hoc recovery helpers (`run_recovery_*`, `summarize_recovery_*`)

### Configs

- stable aliases are in `configs/stable/*` (recommended one-folder picks).
- canonical experimental config roots are `configs/experimental/latent/*` and `configs/experimental/pixel/*`.
- legacy template/ablation/recovery paths remain as compatibility wrappers.
- `configs/pixel/*` (pixel path remains experimental)
- `configs/latent/*ablation*`
- `configs/latent/*recovery*`
- `configs/latent/*template*` (paper templates are intentionally compute-heavy and not single-GPU defaults)

## Quick decision rule

If your goal is reproducible baseline evidence:

1. use `docs/minimal_repro_imagenet256.md`
2. stay on stable config(s)
3. log results in `docs/reproducibility_scoreboard.md`
4. use run root pattern `outputs/imagenet/stable_<timestamp>/`

If your goal is research iteration:

- use experimental scripts/configs
- keep outputs and claims explicitly labeled as non-baseline
- use run root pattern `outputs/imagenet/exp_<name>_<timestamp>/`

See `docs/output_artifact_contract.md` for full naming + artifact requirements.

## Public claim rule

Only make public performance claims from runs that satisfy all:

- stable surface entrypoints/configs
- eval contract in `docs/eval_contract.md`
- explicit evidence artifacts (`RUN.md`, fingerprints, eval summaries)

Everything else should be treated as exploratory.
