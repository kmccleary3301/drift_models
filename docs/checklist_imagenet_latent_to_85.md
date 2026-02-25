# ImageNet Latent Protocol Checklist (Target: >85% Overall)

This checklist is designed to push the weighted progress in `docs/reproduction_report.md` beyond ~85% by completing the core ImageNet latent protocol (Stage 5) and producing an ImageNet-trained MAE encoder artifact (Stage 6), plus minimal reporting hardening (Stage 8).

## 1) Extraction (Stage 5)
- [x] Train split extraction completes to ImageFolder:
  - `outputs/datasets/imagenet1k_raw/train/<wnid>/*.JPEG`
  - no remaining `outputs/datasets/imagenet1k_raw/train/*.tar`
- [x] Train ImageFolder sanity:
  - 1000 wnid dirs
  - spot-check a few wnids for non-zero image counts
- [x] Val split sanity (already expected true):
  - 1000 wnid dirs
  - exactly 50,000 images total

## 2) Latent exports (Stage 5)
- [x] Val SD-VAE latents sharded export:
  - `outputs/datasets/imagenet1k_val_sdvae_latents_shards/manifest.json`
- [x] Train SD-VAE latents sharded export:
  - `outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json`
  - `outputs/datasets/imagenet1k_train_sdvae_latents_shards/export_summary.json`
- [x] Ingestion sanity:
  - `RealBatchProviderConfig.source=tensor_shards` iterates batches without error
  - labels cover many classes (spot-check)

## 3) Feature encoder (latent-MAE) (Stage 6)
- [x] MAE smoke pretrain completes:
  - `outputs/imagenet/mae_smoke/mae_encoder.pt`
  - `outputs/imagenet/mae_smoke/mae_summary.json`
- [x] Config wiring:
  - `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml` points to `outputs/imagenet/mae_smoke/mae_encoder.pt`

## 4) Drifting smoke (Stage 5)
- [x] Latent drifting smoke completes and writes a checkpoint:
  - `outputs/imagenet/latent_smoke_mae/checkpoint.pt`
  - `outputs/imagenet/latent_smoke_mae/latent_summary.json`
- [x] Sample + decode completes:
  - `outputs/imagenet/latent_smoke_mae_samples_*/images/*`
- [x] Eval completes using cached reference stats:
  - `outputs/imagenet/latent_smoke_mae_eval_*/eval_pretrained.json`

## 5) Reporting + hygiene (Stage 8)
- [x] Append ImageNet runs to `docs/experiment_log.md`
- [x] Update `docs/reproduction_report.md` Stage 5/6 notes and percentages
- [x] Record any new decisions in `docs/decision_log.md`
- [ ] Disk review checkpoint:
  - record `df -h /mnt/drive_4`
  - decide whether to keep `outputs/datasets/imagenet1k_raw/train` (requires explicit approval before deletion)
