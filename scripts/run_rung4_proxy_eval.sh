#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <rung4_run_dir> [device]" >&2
  exit 2
fi

RUN_DIR="$1"
DEVICE="${2:-cuda:0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_DIR="$RUN_DIR/checkpoints"
EVAL_ROOT="$RUN_DIR/proxy_eval"
REF_TENSOR="outputs/stage8_eval_smoke/ref.pt"
STEPS=(2000 5000 10000)

mkdir -p "$EVAL_ROOT"

for STEP in "${STEPS[@]}"; do
  STEP_PADDED="$(printf '%08d' "$STEP")"
  CKPT_PATH="$CHECKPOINT_DIR/checkpoint_step_${STEP_PADDED}.pt"
  STEP_ROOT="$EVAL_ROOT/step_${STEP_PADDED}"
  SAMPLE_ROOT="$STEP_ROOT/samples"
  EVAL_SUMMARY="$STEP_ROOT/eval_summary.json"

  if [ ! -f "$CKPT_PATH" ]; then
    echo "[rung4-proxy-eval] missing checkpoint: $CKPT_PATH" >&2
    exit 1
  fi

  uv run python scripts/sample_latent.py \
    --device "$DEVICE" \
    --checkpoint-path "$CKPT_PATH" \
    --output-root "$SAMPLE_ROOT" \
    --n-samples 2000 \
    --batch-size 64 \
    --alpha 1.0 \
    --write-imagefolder \
    --decode-mode conv \
    --decode-image-size 32

  uv run python scripts/eval_fid_is.py \
    --device "$DEVICE" \
    --batch-size 64 \
    --inception-weights none \
    --reference-source tensor_file \
    --reference-tensor-file-path "$REF_TENSOR" \
    --generated-source imagefolder \
    --generated-imagefolder-root "$SAMPLE_ROOT/images" \
    --output-path "$EVAL_SUMMARY"
done

echo "[rung4-proxy-eval] completed: $EVAL_ROOT"
