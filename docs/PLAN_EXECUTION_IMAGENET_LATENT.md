# ImageNet Latent Protocol Execution Plan (Methodical, Artifact-First)

Date: 2026-02-10

This plan operationalizes Stage 5 (ImageNet latent protocol) and Stage 6 (latent-MAE) from `docs/IMPLEMENTATION_PLAN.md`,
with an emphasis on repeatability, disk hygiene, and clear “gates” between steps.

Key references:
- Paper scan: `Drift_Models/Drift_Models.md` (Table 8, multi-temperature semantics, alpha sampling)
- Runbook: `docs/imagenet_runbook.md`
- Checklist: `docs/checklist_to_80.md`
- Progress rubric: `docs/reproduction_report.md`

## Ground Rules

1. All large artifacts go under `outputs/`.
2. Avoid duplicate large copies:
   - Prefer symlinks to archives where possible.
   - Prefer sharded latent export to avoid monolithic >10GB `.pt` files.
3. Guard disk usage:
   - Keep an eye on `/mnt/drive_4` free space (`df -h /mnt/drive_4`).
   - Do not delete large datasets without explicit approval.
4. Prefer “smoke first, then scale”:
   - Validate wiring on ImageNet val / small steps before committing to long runs.

## Disk Budget (Reality Check)
Typical disk footprint components:
- `data/ILSVRC2012_img_train.tar.1`: ~138 GB (kept if you want re-extraction ability)
- `outputs/datasets/imagenet1k_raw/{train,val}`: ~146 GB total (deletable after latents + ref stats)
- `outputs/datasets/imagenet1k_train_sdvae_latents_shards`: ~10 GB
- `outputs/datasets/imagenet1k_val_sdvae_latents_shards`: ~0.4 GB
- decoded samples: ~0.2 GB per 5k JPGs (a few GB for 50k)

## Phase 0: Environment Sanity (Gate 0)

Goal: ensure our execution environment is consistent and tests are green.

Commands:
```bash
uv sync --extra dev --extra sdvae --extra imagenet
uv run python -m pytest -q
```

Acceptance:
- Tests pass.
- CUDA visible if intended (optional): `uv run python -c "import torch; print(torch.cuda.is_available())"`

Artifacts:
- none (other than local venv caches)

## Phase 1: ImageNet Archives + Extraction (Gate 1)

Goal: ImageFolder layout exists for ImageNet-1k ILSVRC2012.

Expected archives (from image-net.org, ILSVRC2012):
- Train: `ILSVRC2012_img_train.tar` (in our case currently downloaded as `data/ILSVRC2012_img_train.tar.1`)
- Val: `data/ILSVRC2012_img_val.tar`
- Devkit: `data/ILSVRC2012_devkit_t12.tar.gz`

Extraction command (supports non-standard filenames):
```bash
uv run python scripts/prepare_imagenet1k_from_archives.py \
  --output-root outputs/datasets/imagenet1k_raw \
  --train-archive-path data/ILSVRC2012_img_train.tar.1 \
  --val-archive-path data/ILSVRC2012_img_val.tar \
  --devkit-archive-path data/ILSVRC2012_devkit_t12.tar.gz \
  --allow-existing \
  --split all \
  --skip-md5
```

Acceptance:
- `outputs/datasets/imagenet1k_raw/val/<wnid>/*.JPEG` exists and totals 50,000.
- `outputs/datasets/imagenet1k_raw/train/<wnid>/*.JPEG` exists and totals ~1.28M.
- Train synset `.tar` intermediates (if any) are removed as extraction completes.

Artifacts:
- `outputs/datasets/imagenet1k_raw/{train,val}/...`
- `outputs/datasets/imagenet1k_raw/meta.bin` (devkit mapping)

## Phase 2: Cache ImageNet Val Reference Stats (Gate 2)

Goal: stable, reusable reference stats for pretrained Inception.

Command:
```bash
uv run python scripts/cache_reference_stats.py \
  --imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --output-path outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --inception-weights pretrained
```

Acceptance:
- `outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt` exists.
- A small JSON summary exists (protocol metadata).

Artifacts:
- `outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`
- `outputs/datasets/imagenet1k_val_reference_stats_pretrained_summary.json`

## Phase 3: SD-VAE Latent Export (Sharded) (Gate 3)

Goal: sharded SD-VAE latents on disk for fast training and queueing.

Val export (already completed in current workspace):
```bash
uv run python scripts/export_sd_vae_latents_tensor_file.py \
  --imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --output-shards-dir outputs/datasets/imagenet1k_val_sdvae_latents_shards \
  --shard-size 10000 \
  --batch-size 16 \
  --num-workers 8 \
  --device auto \
  --image-size 256 \
  --save-dtype fp16 \
  --latent-sampling mean
```

Train export (time- and disk-intensive; run once extraction finishes):
```bash
uv run python scripts/export_sd_vae_latents_tensor_file.py \
  --imagefolder-root outputs/datasets/imagenet1k_raw/train \
  --output-shards-dir outputs/datasets/imagenet1k_train_sdvae_latents_shards \
  --shard-size 10000 \
  --batch-size 16 \
  --num-workers 8 \
  --device auto \
  --image-size 256 \
  --save-dtype fp16 \
  --latent-sampling mean
```

Acceptance:
- `outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json` exists.
- We can iterate a few batches using `real-batch-source: tensor_shards` without errors.

Artifacts:
- `outputs/datasets/imagenet1k_{train,val}_sdvae_latents_shards/*`

Note: `tensor_shards` ingestion requires `real-num-workers: 0` (enforced) to avoid shard-cache thrash.

Disk checkpoint (explicit):
- After train latents exist and are verified, decide whether to keep or remove
  `outputs/datasets/imagenet1k_raw/train` to reclaim ~100GB+ (requires explicit approval).

## Phase 4: Latent-MAE Feature Encoder (Gate 4)

Goal: produce an MAE encoder checkpoint usable as the feature extractor in drifting loss.

Smoke pretrain (short):
```bash
uv run python scripts/train_mae.py \
  --config configs/mae/imagenet1k_sdvae_latents_shards_smoke.yaml \
  --output-dir outputs/imagenet/mae_smoke \
  --real-tensor-shards-manifest-path outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json
```

Acceptance:
- `outputs/imagenet/mae_smoke/mae_encoder.pt` exists.
- Loss decreases at least modestly over 200 steps (sanity only).

Artifacts:
- `outputs/imagenet/mae_smoke/mae_encoder.pt`
- `outputs/imagenet/mae_smoke/mae_summary.json`

## Phase 5: ImageNet Latent Drifting Smoke (Gate 5)

Goal: run a short drifting-model training job end-to-end on ImageNet latents.

Steps:
1. Update the config:
   - `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`
   - Set `mae-encoder-path` to the MAE export.
   - Ensure `real-tensor-shards-manifest-path` points to the train shards manifest.
2. Train (smoke):
```bash
uv run python scripts/train_latent.py \
  --config configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml \
  --output-dir outputs/imagenet/latent_smoke_mae \
  --checkpoint-path outputs/imagenet/latent_smoke_mae/checkpoint.pt
```

Acceptance:
- Training runs without NaNs / crashes.
- Summary JSON written with dataset fingerprint and queue warmup stats.

Artifacts:
- `outputs/imagenet/latent_smoke_mae/*`

## Phase 6: Sample, Decode, and Evaluate (Gate 6)

Goal: produce comparable FID/IS against ImageNet val using cached stats.

Sample + decode:
```bash
uv run python scripts/sample_latent.py \
  --checkpoint-path outputs/imagenet/latent_smoke_mae/checkpoint.pt \
  --output-root outputs/imagenet/latent_smoke_mae_samples \
  --n-samples 5000 \
  --batch-size 64 \
  --write-imagefolder \
  --decode-mode sd_vae \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --sd-vae-revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --decode-image-size 256 \
  --postprocess-mode clamp_0_1
```

Eval:
```bash
uv run python scripts/eval_fid_is.py \
  --reference-source imagefolder \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --generated-source imagefolder \
  --generated-imagefolder-root outputs/imagenet/latent_smoke_mae_samples/images \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --inception-weights pretrained \
  --output-path outputs/imagenet/latent_smoke_mae_eval/eval_pretrained.json
```

Acceptance:
- Eval completes; JSON includes protocol fields.
- Values are used only as a smoke baseline (not expected to match paper at this horizon).

Artifacts:
- `outputs/imagenet/latent_smoke_mae_samples/images/*`
- `outputs/imagenet/latent_smoke_mae_eval/eval_pretrained.json`

## Phase 7: Logging + Report Update (Gate 7)

Goal: keep the reproduction narrative consistent with executed artifacts.

Checklist:
- Append executed commands to `docs/experiment_log.md` (via `scripts/append_experiment_log.py` or manual).
- Update `docs/reproduction_report.md`:
  - Stage 5 status (ImageNet latent pipeline now real)
  - Stage 6 status (ImageNet MAE encoder available)
- Record any new paper-interpretation decisions in `docs/decision_log.md`.
