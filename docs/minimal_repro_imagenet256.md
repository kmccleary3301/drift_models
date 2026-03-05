# Minimal ImageNet-256 reproduction lane (single GPU)

This document defines one narrow, repeatable lane for ImageNet-256 latent training and evaluation.

Use this if you want:

- one stable config
- one training command
- one post-run evidence command
- cost/time numbers that can be compared between runs

This lane is intentionally strict. If you need ablations, use `docs/commands.md` and treat those runs as non-minimal.

## Scope

- Dataset protocol: ImageNet-1k (ILSVRC2012) 256x256
- Representation: SD-VAE latents (32x32x4)
- Training config: `configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml`
- Eval metric contract: pretrained Inception FID/IS via `scripts/eval_fid_is.py`
- Claim boundary: mechanical/implementation evidence only (not paper-parity claims)

## Required assets

Before starting, verify these paths exist:

```bash
test -f outputs/datasets/imagenet1k_train_sdvae_latents_shards/manifest.json
test -f outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt
test -f outputs/imagenet/mae_variant_a_w64/mae_encoder.pt
test -d outputs/datasets/imagenet1k_raw/val
```

If missing, use `docs/imagenet_runbook.md` to build them.

## 0) Environment and preflight

```bash
uv sync --extra dev --extra eval --extra sdvae --extra imagenet
uv run python scripts/runtime_preflight.py --device auto --check-torchvision --strict
```

## 1) Launch the training run

One-command wrapper (includes preflight + train + artifact validation):

```bash
uv run python scripts/runtime_stable_lane.py \
  --config configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml \
  --device cuda:0
```

Manual equivalent:

```bash
OUT=$(uv run python scripts/make_run_root.py --lane stable --base-dir outputs/imagenet | python -c "import json,sys; print(json.load(sys.stdin)['run_root'])")
mkdir -p "${OUT}"

uv run python scripts/train_latent.py \
  --config configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml \
  --device cuda:0 \
  --output-dir "${OUT}" \
  --checkpoint-dir "${OUT}/checkpoints" \
  --checkpoint-path "${OUT}/checkpoint.pt" \
  --keep-last-k-checkpoints 2
```

Notes:

- Replace `cuda:0` if needed.
- This config is a closest-feasible single-GPU approximation, not full Table-8 paper compute.

## 2) Run post-training evidence bundle

When training reaches configured `steps`, run:

```bash
uv run python scripts/wait_and_run_latent_claim_bundle.py \
  --run-dir "${OUT}" \
  --device cuda:0 \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --target-step 200000
```

This writes a single evidence bundle under:

- `${OUT}/claim_bundle/bundle_summary.json`
- `${OUT}/claim_bundle/claim_eval/eval_pretrained.json`
- `${OUT}/claim_bundle/alpha_sweep/alpha_sweep_summary.json`
- `${OUT}/claim_bundle/last_k_eval/last_k_summary.json`

## 3) Minimal run acceptance checklist

A run is considered complete for this lane only if all are true:

- `checkpoint_step_00200000.pt` exists under `${OUT}/checkpoints/`
- `${OUT}/claim_bundle/bundle_summary.json` exists and has `"status": "done"`
- `${OUT}/claim_bundle/claim_eval/eval_pretrained.json` exists
- Run metadata exists: `${OUT}/RUN.md`, `${OUT}/env_snapshot.json`, `${OUT}/codebase_fingerprint.json`

Validate in one command:

```bash
uv run python scripts/validate_run_artifacts.py --run-root "${OUT}" --lane stable
```

## 4) Time and cost recording

Use this helper to compute estimated GPU-hours from run summary:

```bash
python - <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/imagenet/<RUN_ID>")
summary = json.loads((run_dir / "latent_summary.json").read_text())
steps = summary["train_config"]["steps"]
mean_step = summary["perf"]["mean_step_time_s"]
gpu_hours = steps * mean_step / 3600.0
print({"steps": steps, "mean_step_time_s": mean_step, "estimated_gpu_hours": gpu_hours})
PY "${OUT}"
```

Then record:

- `estimated_gpu_hours`
- your effective `cost_per_gpu_hour`
- `estimated_cost = estimated_gpu_hours * cost_per_gpu_hour`

Use `docs/reproducibility_scoreboard.md` for canonical logging.

## 5) What this lane is for

This lane is for:

- repeatable quality/cost tracking
- comparing config changes apples-to-apples
- publishing one clean baseline path publicly

This lane is not for:

- broad architecture ablation sweeps
- pixel-path benchmarking
- claiming paper-level parity
