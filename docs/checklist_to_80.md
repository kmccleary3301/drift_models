# Checklist To >80% Overall Completion

This checklist is scoped to the staged rubric in `docs/reproduction_report.md` and is designed to move the overall weighted completion above **80%** by completing the ImageNet latent protocol (Stage 5) and hardening the MAE + reporting pieces (Stages 6 and 8).

## A) ImageNet archives and extraction (Stage 5)
- [x] Download `ILSVRC2012_img_train.tar` (train images)
- [x] Download `ILSVRC2012_img_val.tar` (val images)
- [x] Download `ILSVRC2012_devkit_t12.tar.gz` (devkit)
- [ ] Verify MD5 for each archive matches torchvision’s expected MD5
- [~] Place archives under `data/` with exact filenames (train is present as `ILSVRC2012_img_train.tar.1`; using override path)
- [ ] Extract to ImageFolder via `scripts/prepare_imagenet1k_from_archives.py`
- [ ] Confirm output dirs exist:
  - `outputs/datasets/imagenet1k_raw/train/`
  - `outputs/datasets/imagenet1k_raw/val/`
- [~] Sanity check counts: ~1.28M train, 50K val (val confirmed; train in-progress)
- [ ] Record dataset fingerprint (fast mode expected at ImageNet scale)

## B) SD-VAE latent export at ImageNet scale (Stage 5)
- [ ] Export **train** SD-VAE latents to shards:
  - `outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json`
- [x] Export **val** SD-VAE latents to shards (optional but recommended)
- [ ] Decide shard size (10K default) and dtype (`fp16` default)
- [ ] Validate one shard payload keys: `images`, `labels`, `meta`
- [ ] Validate `RealBatchProviderConfig.source=tensor_shards` can iterate batches
- [ ] Validate class coverage in queue warmup report (spot-check)

## C) Reference stats caching for ImageNet val (Stage 5)
- [x] Cache ImageNet val reference stats using pretrained Inception:
  - `outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`
- [ ] Lock eval protocol parameters (sample count, resizing, weights)
- [ ] Ensure subsequent evals use `--load-reference-stats` for comparability

## D) Latent-MAE: ImageNet-latent feature encoder (Stage 6)
- [ ] Pretrain latent-MAE on **ImageNet train SD-VAE latents** (smoke)
- [ ] Export encoder artifact:
  - `outputs/imagenet/mae_smoke/mae_encoder.pt`
- [ ] Run a longer MAE pretrain (timeboxed) and export a “v0” encoder
- [ ] (Optional) Implement paper’s classifier finetune stage (3k steps) and export “v1”
- [ ] Track MAE training summary curves and store under `outputs/`

## E) Latent drifting: ImageNet smoke and early trends (Stage 5)
- [ ] Run `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`
- [ ] Sample + decode with SD-VAE (`scripts/sample_latent.py --decode-mode sd_vae`)
- [ ] Evaluate FID/IS vs ImageNet val with cached reference stats
- [ ] Run alpha sweep on a fixed checkpoint (`scripts/eval_alpha_sweep.py`)
- [ ] Run last-K checkpoint trend evaluation (`scripts/eval_last_k_checkpoints.py`)
- [ ] Verify qualitative trends:
  - alpha affects IS/FID tradeoff
  - alpha=1 near best FID (per paper’s Fig. 5 claim for L/2)

## F) Table 8 parity configs and scale readiness (Stage 5)
- [ ] Create per-column configs matching Table 8 (B/2 latent, L/2 latent)
- [ ] Use correct alpha sampling:
  - B/2 latent: `p(alpha) ∝ alpha^-5` on `[1,4]`
  - L/2 latent: 50% `alpha=1`, 50% `p(alpha) ∝ alpha^-3`
- [ ] Use correct temperatures set and feature-loss temperatures
- [ ] Ensure queue sizes and prime samples are reasonable for scale

## G) Reporting and reproducibility hardening (Stage 8)
- [ ] Add ImageNet run entries to `docs/experiment_log.md`
- [ ] Update `docs/reproduction_report.md` Stage 5 and Stage 6 completion notes
- [ ] Record all “paper-backed vs inferred” decisions touched in `docs/decision_log.md`
- [ ] Ensure all ImageNet commands are captured in `docs/imagenet_runbook.md`

## H) Disk hygiene (must not regress)
- [ ] Track `/mnt/drive_4` free space after each major artifact step
- [ ] After train latents export, decide whether to keep `outputs/datasets/imagenet1k_raw/train` (requires explicit approval before deleting)
