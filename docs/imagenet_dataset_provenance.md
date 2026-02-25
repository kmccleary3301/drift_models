# ImageNet-1k (ILSVRC2012) Dataset Provenance (Latent Protocol)

This repo’s “ImageNet latent protocol” uses **ILSVRC2012 (ImageNet-1k) classification** train/val images, resized to `256x256`, encoded to SD‑VAE latents (`32x32x4`), then decoded for evaluation (FID/IS).

This document is intentionally operational: it says **exactly what to download** and how to verify what’s on disk.

## What to download from image-net.org (the three required files)

From the “Download ImageNet Data” page:
- Go to **ImageNet Large-scale Visual Recognition Challenge (ILSVRC)** → **2012**
- Download:
  1) **ILSVRC2012 training images** (`ILSVRC2012_img_train.tar`)
  2) **ILSVRC2012 validation images** (`ILSVRC2012_img_val.tar`)
  3) **ILSVRC2012 devkit** (`ILSVRC2012_devkit_t12.tar.gz`)

These are the only downloads required for the standard ImageNet‑1k train/val pipeline in this repo.

### What *not* to download (unless you have a specific reason)
- **Winter 2021 ImageNet21K / resized ImageNet21K / ImageNet10K**: not used for ImageNet‑1k reproduction.
- **Face‑blurred ILSVRC2012–2017** archives: privacy-aware variant; not the default ILSVRC2012 dataset. Use only if you explicitly choose to switch variants and record the deviation.
- **ILSVRC test sets / evaluation server**: not needed for val-based FID/IS reproduction.

## Local on-disk layout used by this repo

Place the official archives under `data/`:
- `data/ILSVRC2012_img_train.tar.1` (this workspace’s filename; functionally equivalent to `ILSVRC2012_img_train.tar`)
- `data/ILSVRC2012_img_val.tar`
- `data/ILSVRC2012_devkit_t12.tar.gz`

Extraction target (temporary / can be deleted after exports):
- `outputs/datasets/imagenet1k_raw/{train,val}/...`

Latent shards (kept for training):
- `outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json`
- `outputs/datasets/imagenet1k_val_sdvae_latents_shards/manifest.json`

## Checksums (this workspace)

This workspace records **md5 + sha256 + size** for each archive in:
- `outputs/datasets/imagenet1k_provenance.json`

If you need to re-verify manually:
```bash
md5sum data/ILSVRC2012_img_train.tar.1 data/ILSVRC2012_img_val.tar data/ILSVRC2012_devkit_t12.tar.gz
sha256sum data/ILSVRC2012_img_train.tar.1 data/ILSVRC2012_img_val.tar data/ILSVRC2012_devkit_t12.tar.gz
```

## License / access constraints (important)

ImageNet access is granted under image-net.org’s terms (typically **non-commercial research/educational** use). Do not upload the raw images (or archives) to public storage/HF.

## Related runbook
- `docs/imagenet_runbook.md` (end-to-end extract → latents → train → sample → eval)

