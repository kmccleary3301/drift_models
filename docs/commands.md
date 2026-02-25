# Command Catalog

This page keeps the detailed command inventory previously embedded in the README.

## Environment and tests

```bash
./scripts/setup_env.sh
uv run python scripts/runtime_preflight.py --device auto --check-torchvision --strict --output-path outputs/runtime_preflight/local.json
uv run python scripts/summarize_runtime_preflight.py --input-glob "outputs/runtime_preflight/*.json" --output-md outputs/runtime_preflight/summary.md --output-json outputs/runtime_preflight/summary.json
uv run pytest -q
```

## Toy + latent training commands

```bash
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --ablation all --device cpu

uv run python scripts/train_latent.py --steps 3 --log-every 1 --groups 2 --negatives-per-group 2 --positives-per-group 2 --image-size 16 --patch-size 4 --hidden-dim 64 --depth 2 --num-heads 4 --device cpu

uv run python scripts/train_latent.py --steps 2 --log-every 1 --groups 2 --negatives-per-group 2 --positives-per-group 2 --image-size 16 --patch-size 4 --hidden-dim 64 --depth 2 --num-heads 4 --device cpu --use-feature-loss --feature-base-channels 8 --feature-stages 2

uv run python scripts/train_latent.py --steps 2 --log-every 1 --groups 2 --negatives-per-group 2 --positives-per-group 2 --unconditional-per-group 2 --image-size 16 --patch-size 4 --hidden-dim 64 --depth 2 --num-heads 4 --device cpu --use-queue

uv run python scripts/train_latent.py --config configs/latent/smoke_raw.yaml
uv run python scripts/train_latent.py --config configs/latent/smoke_feature.yaml
uv run python scripts/train_latent.py --config configs/latent/smoke_feature_queue.yaml
```

## MAE and pixel pipeline commands

```bash
uv run python scripts/train_mae.py --config configs/mae/smoke_latent.yaml

uv run python scripts/train_pixel.py --config configs/pixel/smoke_feature.yaml
uv run python scripts/train_pixel.py --config configs/pixel/smoke_feature_queue_mae.yaml

uv run python scripts/train_mae.py --device cpu --steps 3 --in-channels 3 --base-channels 8 --stages 2 --export-encoder-path outputs/stage7_mae_encoder_for_pixel/mae_encoder.pt --output-dir outputs/stage7_mae_encoder_for_pixel && \
uv run python scripts/train_pixel.py --device cpu --steps 2 --use-feature-loss --feature-encoder mae --feature-base-channels 8 --feature-stages 2 --feature-selected-stages 0 1 --mae-encoder-path outputs/stage7_mae_encoder_for_pixel/mae_encoder.pt --output-dir outputs/stage7_pixel_mae_encoder_load_smoke
```

## Eval + sampling commands

```bash
uv run python scripts/eval_fid_is.py --device cpu --inception-weights none --reference-source tensor_file --reference-tensor-file-path outputs/stage8_eval_smoke/ref.pt --generated-source tensor_file --generated-tensor-file-path outputs/stage8_eval_smoke/gen.pt --output-path outputs/stage8_eval_smoke/eval_summary.json

uv run python scripts/eval_fid_is.py --device cpu --inception-weights pretrained --reference-source imagefolder --reference-imagefolder-root outputs/stage8_cifar10/reference --generated-source imagefolder --generated-imagefolder-root outputs/stage8_cifar10/generated_noisy --output-path outputs/stage8_cifar10/eval_pretrained_noisy.json

uv run python scripts/run_end_to_end_pixel_eval.py --output-root outputs/e2e_pixel_smoke --device cpu --train-steps 2 --sample-count 32 --reference-imagefolder-root outputs/stage8_cifar10/reference --inception-weights none

uv run python scripts/run_end_to_end_latent_eval.py --output-root outputs/e2e_latent_smoke --device cpu --train-steps 2 --sample-count 32 --reference-imagefolder-root outputs/stage8_cifar10/reference --inception-weights none

uv run python scripts/eval_alpha_sweep.py --mode pixel --checkpoint-path outputs/<run>/checkpoint.pt --output-root outputs/<run>_alpha_sweep --alphas 1 2 3 --n-samples 512 --reference-imagefolder-root outputs/stage8_cifar10/reference --inception-weights pretrained --reference-cache

uv run python scripts/eval_last_k_checkpoints.py --mode pixel --checkpoint-dir outputs/<run>/checkpoints --k 5 --output-root outputs/<run>_last_k --n-samples 512 --reference-imagefolder-root outputs/stage8_cifar10/reference --inception-weights pretrained --reference-cache

uv run python scripts/sample_pixel.py --device cpu --checkpoint-path outputs/<run>/checkpoint.pt --output-root outputs/<run>_samples --n-samples 128 --batch-size 16
```

## Operational flags and patterns

```bash
# Save reproducible run artifacts
--output-dir outputs/<run_name>

# Save and resume checkpoints
--checkpoint-path outputs/<run>/checkpoint.pt --save-every 1
--resume-from outputs/<run>/checkpoint.pt

# Queue mode real-batch provider
--real-batch-source synthetic_dataset --real-dataset-size 4096 --real-loader-batch-size 128

# Legacy MAE export compatibility
--mae-encoder-arch legacy_conv

# Compile fallback policy controls
--compile-fail-action warn
--compile-fail-action raise
--compile-fail-action disable
```

## Device note

- All CLIs share one runtime resolver.
- `--device auto` follows `cuda` → `xpu` → `mps` → `cpu`.
- `--device gpu` requests any available accelerator backend.
- Prefer explicit pinning (`--device cuda:0`, `--device cuda:1`, or `--device cpu`) for reproducible experiments.
