# Disk / Storage Requirements (Practical)

This repo intentionally generates large intermediate artifacts (datasets, checkpoints, decoded samples). This document summarizes **what you need** and **what you can safely delete**.

All numbers below are approximate and will vary by filesystem/block size and which artifacts you choose to keep.

## Quick budgets

### Minimal (tests + toy + small-scale sanity)
- **Recommended free space:** 10–20 GB
- Includes: `.venv`, test artifacts under `outputs/ci/`, small toy outputs.

### CIFAR-10 pixel/latent protocol iteration
- **Recommended free space:** 30–60 GB
- Includes: CIFAR exports, SD-VAE latent exports, short-run checkpoints, 5k–10k decoded samples.

### ImageNet-1k latent pipeline (real data)
- **Recommended free space (comfortable):** 300–500 GB
- You can operate with less if you delete extracted ImageFolder trees after exporting latents.

## ImageNet-1k (ILSVRC2012) latent protocol: where the space goes

### Archives (kept in `data/`)
These are the official downloads you obtained from image-net.org:
- `data/ILSVRC2012_img_train.tar.1`: ~138 GB
- `data/ILSVRC2012_img_val.tar`: ~6.3 GB
- `data/ILSVRC2012_devkit_t12.tar.gz`: ~2.5 MB

Checksums + sizes should be recorded in:
- `outputs/datasets/imagenet1k_provenance.json`

### Extracted ImageFolder (temporary, safe to delete after export)
The extraction step produces:
- `outputs/datasets/imagenet1k_raw/train/<wnid>/*.JPEG`
- `outputs/datasets/imagenet1k_raw/val/<wnid>/*.JPEG`

These can be large (on the order of the archives). After:
- exporting SD-VAE latents, and
- caching reference stats for ImageNet val,

it is safe to delete `outputs/datasets/imagenet1k_raw/train` (and optionally `val`) to reclaim space, as long as you keep the original archives.

### SD-VAE latents (kept for training)
Typical observed sizes in this repo:
- `outputs/datasets/imagenet1k_train_sdvae_latents_shards`: ~10 GB
- `outputs/datasets/imagenet1k_val_sdvae_latents_shards`: ~0.4 GB

These are the “real data” for the latent training loop.

### Checkpoints (can get huge)
Optimizer state dominates checkpoint size.
- Smoke and ablation checkpoints can be multiple GB each.
- Keep only the “best” / “recommended” checkpoint(s) once you have evaluation summaries.

### Decoded samples (for eval)
- 5k decoded JPGs: a few hundred MB
- 50k decoded JPGs: a few GB

If disk is tight, prefer:
- keeping only the eval JSON + sample manifest + a small qualitative subset,
- deleting the bulk decoded sample ImageFolder once FID/IS is recorded and audited.

## “Safe deletions” principles
- Prefer deleting **derived artifacts** (decoded samples, extracted ImageFolder trees, old checkpoints) over **sources** (archives, manifests).
- Keep:
  - `data/*.tar*` (archives)
  - `outputs/datasets/*manifest.json` (latents shard manifests)
  - `outputs/datasets/*reference_stats*.pt` (cached eval reference)
  - the single checkpoint(s) referenced in report/runbooks
  - `RUN.md` + `*_summary.json` + eval JSON for each run you care about

### Cleanup helper (this repo)
To prune duplicate `checkpoint.pt` files and delete bulk decoded sample ImageFolders while keeping latents + summaries, use:
- `scripts/disk_cleanup_checkpoints.py`

## Related runbook
- `docs/imagenet_runbook.md` includes a disk budget section and specific cleanup advice.
