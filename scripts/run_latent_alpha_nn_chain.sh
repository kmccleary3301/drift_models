#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <checkpoint_path> <config_path> <alpha_output_root> <device>" >&2
  exit 2
fi

CKPT_PATH="$1"
CONFIG_PATH="$2"
ALPHA_OUT="$3"
DEVICE="$4"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

until [ -f "$CKPT_PATH" ]; do
  sleep 30
done

uv run python scripts/eval_alpha_sweep.py \
  --output-root "$ALPHA_OUT" \
  --mode latent \
  --checkpoint-path "$CKPT_PATH" \
  --config "$CONFIG_PATH" \
  --device "$DEVICE" \
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
  --device "$DEVICE" \
  --max-generated 512 \
  --max-reference 10000 \
  --output-path "$ALPHA_OUT/alpha_1p5/nn_audit.json"
