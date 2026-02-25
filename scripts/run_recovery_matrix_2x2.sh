#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="cuda:0"
SRC_CKPT="outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/checkpoint_step_00000550.pt"
BASE_OUT="outputs/imagenet/recovery_matrix_2x2_20260213_restart_r3"
CONFIG="configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml"

# All runs: same horizon, conservative checkpointing.
STEPS=700
LOG_EVERY=10
SAVE_EVERY=50
KEEP_LAST_K=4

run_variant() {
  local name="$1"; shift
  local lr="$1"; shift
  local resume_mode="$1"; shift

  local out_dir="$BASE_OUT/$name"
  local ckpt_dir="$out_dir/checkpoints"
  local ckpt_path="$out_dir/checkpoint.pt"
  local alpha_out="$BASE_OUT/${name}_alpha_sweep_s2k"

  mkdir -p "$out_dir"

  echo "[run] $name lr=$lr resume_mode=$resume_mode"

  local train_args=(
    uv run python scripts/train_latent.py
      --config "$CONFIG"
      --device "$DEVICE"
      --steps "$STEPS"
      --log-every "$LOG_EVERY"
      --learning-rate "$lr"
      --scheduler constant
      --warmup-steps 0
      --save-every "$SAVE_EVERY"
      --keep-last-k-checkpoints "$KEEP_LAST_K"
      --checkpoint-dir "$ckpt_dir"
      --checkpoint-path "$ckpt_path"
      --output-dir "$out_dir"
      --resume-from "$SRC_CKPT"
      --allow-resume-config-mismatch
  )

  if [ "$resume_mode" = "model_only" ]; then
    train_args+=(--resume-model-only)
  elif [ "$resume_mode" = "restore_opt" ]; then
    train_args+=(--resume-reset-scheduler --resume-reset-optimizer-lr)
  else
    echo "unknown resume_mode: $resume_mode" >&2
    exit 2
  fi

  "${train_args[@]}" 2>&1 | tee "$out_dir/train.log"

  ./scripts/run_latent_alpha_nn_chain.sh \
    "$ckpt_dir/checkpoint_step_00000700.pt" \
    "$CONFIG" \
    "$alpha_out" \
    "$DEVICE" 2>&1 | tee "$out_dir/postrun_chain.log"
}

run_variant A_restoreopt_lr8e5 8e-05 restore_opt
run_variant B_restoreopt_lr2e4 2e-04 restore_opt
run_variant C_modelonly_lr8e5 8e-05 model_only
run_variant D_modelonly_lr2e4 2e-04 model_only
