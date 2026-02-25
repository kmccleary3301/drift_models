#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="outputs/imagenet/latent_recovery_from550_step700_cuda1_w64/checkpoint.pt"
ALPHA_OUT="outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_alpha_sweep_s2k"

until [ -f "$CKPT" ]; do
  sleep 30
done

uv run python scripts/eval_alpha_sweep.py \
  --output-root "$ALPHA_OUT" \
  --mode latent \
  --checkpoint-path "$CKPT" \
  --config configs/latent/imagenet1k_sdvae_latents_recovery_from550_cuda1_step700_w64.yaml \
  --device cuda:1 \
  --alphas 1.0 1.5 2.0 2.5 3.0 \
  --n-samples 2000 \
  --batch-size 32 \
  --decode-mode sd_vae \
  --decode-image-size 256 \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --inception-weights pretrained \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --overwrite

uv run python scripts/audit_nearest_neighbors.py \
  --generated-root "$ALPHA_OUT/alpha_1p5/samples/images" \
  --reference-root outputs/datasets/imagenet1k_raw/val \
  --device cuda:1 \
  --max-generated 512 \
  --max-reference 10000 \
  --output-path "$ALPHA_OUT/alpha_1p5/nn_audit.json"
