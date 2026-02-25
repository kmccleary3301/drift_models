# Provenance Contract (Dataset + Tokenizer + Evaluator)

This contract defines the canonical provenance artifacts required for paper-facing ImageNet latent claims.

## Canonical provenance artifacts

- Source archives (ImageNet train/val/devkit):
  - `outputs/datasets/imagenet1k_provenance.json`
- Split manifests (training/eval latent shards):
  - `outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json`
  - `outputs/datasets/imagenet1k_val_sdvae_latents_shards/manifest.json`
- SD-VAE tokenizer provenance:
  - `outputs/datasets/sdvae_provenance.json`
- Inception evaluator provenance:
  - `outputs/datasets/imagenet1k_val_reference_stats_pretrained_summary.json`

## Required contract fields

### `imagenet1k_provenance.json`
- `archives[*].role`, `archives[*].path`, `archives[*].size_bytes`
- `archives[*].hashes.md5`, `archives[*].hashes.sha256`

### Latent shard manifests
- train manifest: `exported_samples`, `shards[*].count`, `shards[*].sha256`
- val manifest: `shards[*].count`, `shards[*].sha256`

### `sdvae_provenance.json`
- `model_id`, `revision`
- `files[*].path`, `files[*].sha256`

### reference stats summary
- `inception_weights` (must be `pretrained` for claim-facing metrics)
- `count`, `feature_dim`

## Operational note

Use `outputs/ops/provenance_bundle_*.json` snapshots as point-in-time captures tying these artifacts together during execution.
