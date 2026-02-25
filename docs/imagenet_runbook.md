# ImageNet-1k (ILSVRC2012) Runbook (Latent Protocol)

This runbook covers the “standard latent-space generation protocol” on **ImageNet 256x256** described in the paper:
- SD-VAE tokenizer to **32x32x4** latents
- drifting loss in a **latent-MAE** feature space
- 1-NFE sampling (decode with SD-VAE for evaluation)

It is intentionally staged: you can stop after each stage and still have a coherent artifact set under `outputs/`.

## Disk budget (practical)
Approximate sizes observed in this repo (will vary by filesystem and image format):
- Archives in `data/`:
  - Train archive (`ILSVRC2012_img_train.tar.1`): ~138 GB
  - Val archive (`ILSVRC2012_img_val.tar`): ~6.3 GB
  - Devkit: negligible
- Extracted ImageFolder (`outputs/datasets/imagenet1k_raw/{train,val}`): ~146 GB total (safe to delete after latents + reference stats are verified)
- SD-VAE latents (sharded):
  - Train shards: ~10 GB
  - Val shards: ~0.4 GB
- Cached reference stats (`outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`): ~32 MB
- Smoke checkpoint (`outputs/imagenet/latent_smoke_mae/checkpoint.pt`): multiple GB (optimizer state dominates)
- Samples:
  - 5k decoded JPGs: ~200 MB
  - 50k decoded JPGs: plan for a few GB

## 0) Prereqs

### 0.1 Archives present (from image-net.org)
Place the following files in `data/` (or another `--archives-root`):
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_devkit_t12.tar.gz`

### 0.2 Environment
Use the repo venv:
```bash
uv sync --extra dev --extra sdvae --extra imagenet
```

### 0.3 Device pinning (recommended on multi-GPU hosts)
To force runs onto a specific GPU, pass `--device cuda:<index>` to train/sample/eval entrypoints.
Example: `--device cuda:0`.

If the host only exposes a single GPU, `cuda:1` will fail with `invalid device ordinal`.

### 0.4 Clean-room smoke drill (recommended)
Before long runs, execute a no-manual-step smoke drill:
```bash
uv run python scripts/train_latent.py \
  --config configs/latent/smoke_raw.yaml \
  --output-dir outputs/ops/cleanroom_drill_latent_smoke \
  --checkpoint-path outputs/ops/cleanroom_drill_latent_smoke/checkpoint.pt
```

Expected auto-written artifacts:
- `latent_summary.json`
- `RUN.md`
- `env_snapshot.json`
- `codebase_fingerprint.json`
- `env_fingerprint.json`
- `checkpoint.pt`

## 1) Extract ImageNet to ImageFolder

This uses torchvision’s official parsing logic (including devkit mapping) and produces:
- `outputs/datasets/imagenet1k_raw/train/<wnid>/*.JPEG`
- `outputs/datasets/imagenet1k_raw/val/<wnid>/*.JPEG`

```bash
uv run python scripts/prepare_imagenet1k_from_archives.py \
  --archives-root data \
  --output-root outputs/datasets/imagenet1k_raw
```

If you need to bypass MD5 verification (non-standard variants), add `--skip-md5`.

If your downloaded archives have non-standard names (e.g. `ILSVRC2012_img_train.tar.1`), pass explicit paths:
```bash
uv run python scripts/prepare_imagenet1k_from_archives.py \
  --output-root outputs/datasets/imagenet1k_raw \
  --train-archive-path data/ILSVRC2012_img_train.tar.1 \
  --val-archive-path data/ILSVRC2012_img_val.tar \
  --devkit-archive-path data/ILSVRC2012_devkit_t12.tar.gz
```

### Split integrity checks (recommended)
Torchvision’s ImageNet extractor writes:
- `outputs/datasets/imagenet1k_raw/meta.bin` (devkit parse output; needed for val mapping)
- `outputs/datasets/imagenet1k_raw/{train,val}/...` (ImageFolder layout)

Quick checks:
```bash
test -f outputs/datasets/imagenet1k_raw/meta.bin && echo "meta.bin present"
test -d outputs/datasets/imagenet1k_raw/val && echo "val dir present"
test -d outputs/datasets/imagenet1k_raw/train && echo "train dir present"
```

Note: if you delete `outputs/datasets/imagenet1k_raw/train` after exporting latents (to save disk), the more reliable integrity gate becomes the **latent shards manifests** below.

## 2) Export SD-VAE latents (sharded)

Train latents:
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

Val latents (optional, useful for feature encoder pretraining sanity checks):
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

Each shard dir includes a `manifest.json` consumed by `real-batch-source: tensor_shards`.

For provenance and recipe clarity, export metadata now records:
- `pre_encode_transforms` (ordered preprocessing ops applied to pixels)
- `transform_ordering: preprocess_before_sd_vae_encode`

This explicitly documents that resize/crop/RGB conversion happen **before** SD-VAE encoding.

Note: `tensor_shards` ingestion requires `real-num-workers: 0` (enforced) to avoid shard-cache thrash.

Quick verification (recommended after export):
```bash
uv run python scripts/verify_tensor_shards_manifest.py \
  --manifest-path outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json \
  --mode quick \
  --max-shards 2 \
  --check-sha256 \
  --output-path outputs/datasets/imagenet1k_train_sdvae_latents_shards/verify_quick.json
```

Expected values (ILSVRC2012 ImageNet-1k classification):
- `num_classes_manifest` should be `1000`
- `label_min` should be `0`
- `label_max` should be `999`
- `total_items_manifest` should match the split:
  - train: `1,281,167`
  - val: `50,000`

Val manifest verification:
```bash
uv run python scripts/verify_tensor_shards_manifest.py \
  --manifest-path outputs/datasets/imagenet1k_val_sdvae_latents_shards/manifest.json \
  --mode quick \
  --max-shards 2 \
  --check-sha256 \
  --output-path outputs/datasets/imagenet1k_val_sdvae_latents_shards/verify_quick.json
```

### Disk hygiene note (recommended)
Once you have:
- cached reference stats (`outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`)
- train/val SD-VAE latents exported and verified

you can consider deleting the extracted ImageFolder trees (`outputs/datasets/imagenet1k_raw/train` and `outputs/datasets/imagenet1k_raw/val`) to reclaim ~150GB. Keep `data/*.tar*` if you want the ability to re-extract later.

### Safe deletions / cleanup tooling
When disk is tight, prefer keeping:
- provenance (`outputs/datasets/*provenance*.json`)
- shard manifests (`outputs/datasets/*manifest.json`)
- cached reference stats (`outputs/datasets/*reference_stats*.pt`)
- step checkpoints referenced in docs/runbooks

For pruning duplicate `checkpoint.pt` files (replace with symlink to the latest step checkpoint) and deleting bulk decoded `samples/images` trees, use:
```bash
uv run python scripts/disk_cleanup_checkpoints.py --help
```

Curated “keep list”:
- `docs/golden_artifacts.md`

## 3) Cache ImageNet val reference stats (pretrained Inception)

This creates a reference stats tensor so that eval runs reuse the same reference distribution.

```bash
uv run python scripts/cache_reference_stats.py \
  --imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --output-path outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --inception-weights pretrained
```

## 4) Pretrain latent-MAE (feature encoder)

Start with a short smoke MAE run on **train SD-VAE latents**:
```bash
uv run python scripts/train_mae.py \
  --config configs/mae/imagenet1k_sdvae_latents_shards_smoke.yaml \
  --device cuda:0 \
  --output-dir outputs/imagenet/mae_variant_a_w64 \
  --checkpoint-path outputs/imagenet/mae_variant_a_w64/checkpoint.pt \
  --save-every 20 \
  --encoder-arch paper_resnet34_unet \
  --blocks-per-stage 2 \
  --norm-groups 8
```

This produces an encoder export at:
- `outputs/imagenet/mae_variant_a_w64/mae_encoder.pt`

Optional classifier fine-tune branch (`cls ft`) and dedicated export:
```bash
uv run python scripts/train_mae.py \
  --resume-from outputs/imagenet/mae_variant_a_w64/checkpoint.pt \
  --steps 200 \
  --output-dir outputs/imagenet/mae_variant_a_w64_clsft \
  --checkpoint-path outputs/imagenet/mae_variant_a_w64_clsft/checkpoint.pt \
  --real-batch-source tensor_shards \
  --real-tensor-shards-manifest-path outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json \
  --real-loader-batch-size 64 \
  --real-num-workers 0 \
  --cls-ft-steps 120 \
  --cls-ft-batch-size 64 \
  --cls-ft-learning-rate 1e-4 \
  --export-cls-ft-encoder-path outputs/imagenet/mae_variant_a_w64_clsft/mae_encoder_clsft.pt
```

### Table-8 width-parity MAE exports (`w256`, `w640`)

To materialize the faithful-template MAE width contracts (`w256` for ablation-default, `w640` for B/2 and L/2), use:

```bash
uv run python scripts/run_mae_width_parity_exports.py \
  --device cuda:0 \
  --skip-existing \
  --wait-state-path outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/claim_bundle/wait_state.json \
  --wait-for-gpu-idle \
  --manifest-path outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json
```

This runner:
- waits for the claim-bundle watcher state to reach `status=done` (if `--wait-state-path` is provided),
- waits for GPU-bound training/eval jobs to go idle (if `--wait-for-gpu-idle` is set),
- runs width-parity MAE export jobs sequentially:
  - `configs/mae/imagenet1k_sdvae_latents_shards_table8_w256_bootstrap.yaml` -> `outputs/imagenet/mae_variant_a_w256/mae_encoder.pt`
  - `configs/mae/imagenet1k_sdvae_latents_shards_table8_w640_bootstrap.yaml` -> `outputs/imagenet/mae_variant_a_w640/mae_encoder.pt`
- audits each bundle to ensure `base_channels` and `encoder_arch` metadata match expected width + `paper_resnet34_unet`.

Artifacts are written under `outputs/ops/mae_width_parity_exports_<timestamp>/`.

## 5) Train drifting model (latent protocol)

Smoke config:
- `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml`

By default, ImageNet latent configs now point at `outputs/imagenet/mae_variant_a_w64/mae_encoder.pt`.
Update `real-tensor-shards-manifest-path` as needed, then run:
```bash
uv run python scripts/train_latent.py \
  --config configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml \
  --output-dir outputs/imagenet/latent_smoke_mae \
  --checkpoint-path outputs/imagenet/latent_smoke_mae/checkpoint.pt
```

## 6) Sample + decode with SD-VAE

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

## 7) Evaluate FID/IS

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

## 8) Best checkpoint recommendation (current)

Recommendation memo:
- `docs/imagenet_step400_best_checkpoint_recommendation.md`

Current default operating point from completed continuation evals:
- checkpoint: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt`
- alpha: `1.5`
- device lane: host-dependent (prefer explicitly setting `--device cuda:0` on single-GPU hosts)

Sample (default recommendation):
```bash
uv run python scripts/sample_latent.py \
  --device cuda:0 \
  --seed 1337 \
  --checkpoint-path outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt \
  --config configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_step400_w64.yaml \
  --output-root outputs/imagenet/step400_best_alpha1p5_samples \
  --n-samples 5000 \
  --batch-size 32 \
  --alpha 1.5 \
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
  --device cuda:0 \
  --batch-size 32 \
  --inception-weights pretrained \
  --reference-source imagefolder \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --generated-source imagefolder \
  --generated-imagefolder-root outputs/imagenet/step400_best_alpha1p5_samples/images \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --output-path outputs/imagenet/step400_best_alpha1p5_eval/eval_pretrained.json
```

If IS is prioritized over FID, use the same checkpoint with `--alpha 3.0`.

## 9) Paper-scale templates

See:
- `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`

This template matches Table 8’s large-batch L/2 latent regime, but requires substantial compute.

## 10) Post-run claim bundle automation

To avoid manual gaps after a long run, start a watcher that waits for the final step checkpoint and then runs:
- last-K checkpoint eval package,
- fixed-checkpoint alpha sweep package,
- nearest-neighbor audits,
- claim-scope sample-set generation + pretrained Inception eval.

Example for the active long-horizon run:
```bash
uv run python scripts/wait_and_run_latent_claim_bundle.py \
  --run-dir outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045 \
  --device cuda:0
```

Outputs are written under:
- `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/claim_bundle`

Progress state file:
- `.../claim_bundle/wait_state.json`
