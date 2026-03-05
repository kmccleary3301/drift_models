# Config surface map

This file defines which config files are stable defaults and which are exploratory.

## Stable defaults

### Toy

- `toy/quick.yaml`
- `toy/base.yaml`

### Latent

- `latent/smoke_feature.yaml`
- `latent/smoke_feature_queue.yaml`
- `latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`
- `latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml`

### MAE

- `mae/imagenet1k_sdvae_latents_shards_smoke.yaml`

## Stable aliases

Use `configs/stable/*` for one-folder stable entrypoints:

- `stable/toy_quick.yaml`
- `stable/toy_base.yaml`
- `stable/latent_smoke_feature.yaml`
- `stable/latent_smoke_feature_queue.yaml`
- `stable/latent_imagenet_queue_smoke_mae.yaml`
- `stable/latent_imagenet_table8_b2_closest_feasible_single_gpu.yaml`
- `stable/mae_imagenet_shards_smoke.yaml`

## Experimental configs

Treat these as exploratory unless promoted:

- canonical experimental configs live under:
  - `experimental/latent/*`
  - `experimental/pixel/*`
- compatibility wrappers remain at legacy paths under:
  - `latent/*ablation*`
  - `latent/*recovery*`
  - `latent/*template*`
  - `pixel/*template*`
- lifecycle status (active / maintenance-only / deprecated) is tracked in `docs/deprecation_matrix.md`.

## Tier metadata

All config files declare `tier: stable|experimental`.

- CI guard: `scripts/check_config_tiers.py`
- The guard is enforced in `.github/workflows/ci.yml`

Canonical organization for exploratory work is documented in `docs/stable_vs_experimental.md`.

## Selection rule

For claim-facing runs:

1. choose from stable defaults only
2. run the full artifact chain in `docs/minimal_repro_imagenet256.md`
3. record outcomes in `docs/reproducibility_scoreboard.md`
