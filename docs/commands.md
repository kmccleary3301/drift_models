# Command Catalog

> Complete reference for all training, evaluation, and utility commands.

---

## Command Categories

| Category | Description | Jump |
|-------------|--------------|---------|
| Environment | Setup, preflight, and testing | [Environment & Tests](#environment--tests) |
| Toy Training | Quick sanity checks | [Toy Training](#toy-training) |
| Latent Pipeline | Primary training pipeline | [Latent Pipeline](#latent-pipeline) |
| Pixel Pipeline | Experimental pixel-space | [Pixel Pipeline](#pixel-pipeline) |
| MAE Training | Feature encoder training | [MAE Training](#mae-training) |
| Evaluation | FID/IS metrics | [Evaluation & Sampling](#evaluation--sampling) |
| Sampling | Image generation | [Evaluation & Sampling](#evaluation--sampling) |
| Utilities | Helper scripts | [Operational Patterns](#operational-patterns) |

---

## Environment & Tests

### Setup Environment
```bash
./scripts/setup_env.sh
```

### Runtime Preflight
```bash
# Basic preflight check
uv run python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/local.json

# Aggregate multiple preflight reports
uv run python scripts/summarize_runtime_preflight.py \
  --input-glob "outputs/runtime_preflight/*.json" \
  --output-md outputs/runtime_preflight/summary.md \
  --output-json outputs/runtime_preflight/summary.json
```

### Run Tests
```bash
# Full test suite
uv run pytest -q

# Integration tests only
uv run pytest -q tests/integration/test_stage2_smoke.py
```

---

## Toy Training

> Quick sanity checks — complete in ~2 minutes on CPU.

| Config | Speed | Description |
|-----------|---------|---------------|
| `quick.yaml` | Fastest | 2D toy distribution, minimal compute |
| `default.yaml` | Standard | Default toy settings |

```bash
# Quick toy run (recommended for first test)
uv run python scripts/train_toy.py \
  --config configs/toy/quick.yaml \
  --output-dir outputs/toy_quick \
  --ablation all \
  --device cpu
```

---

## Latent Pipeline

> Primary pipeline — trains in latent space (recommended).

### Quick Start
```bash
# Using config file (recommended)
uv run python scripts/train_latent.py \
  --config configs/latent/smoke_feature.yaml \
  --output-dir outputs/latent_smoke
```

### Available Configs

| Config | Features | Description |
|-----------|-------------|---------------|
| `smoke_raw.yaml` | Raw pixels | No feature loss, minimal compute |
| `smoke_feature.yaml` | Feature loss | SD-VAE encoder, quality boost |
| `smoke_feature_queue.yaml` | Feature + Queue | Full pipeline with queue |

### Manual Configuration
```bash
# Minimal manual config
uv run python scripts/train_latent.py \
  --steps 3 \
  --log-every 1 \
  --groups 2 \
  --negatives-per-group 2 \
  --positives-per-group 2 \
  --image-size 16 \
  --patch-size 4 \
  --hidden-dim 64 \
  --depth 2 \
  --num-heads 4 \
  --device cpu

# With feature loss
uv run python scripts/train_latent.py \
  --steps 2 \
  --log-every 1 \
  --groups 2 \
  --negatives-per-group 2 \
  --positives-per-group 2 \
  --image-size 16 \
  --patch-size 4 \
  --hidden-dim 64 \
  --depth 2 \
  --num-heads 4 \
  --device cpu \
  --use-feature-loss \
  --feature-base-channels 8 \
  --feature-stages 2

# With queue mode
uv run python scripts/train_latent.py \
  --steps 2 \
  --log-every 1 \
  --groups 2 \
  --negatives-per-group 2 \
  --positives-per-group 2 \
  --unconditional-per-group 2 \
  --image-size 16 \
  --patch-size 4 \
  --hidden-dim 64 \
  --depth 2 \
  --num-heads 4 \
  --device cpu \
  --use-queue
```

---

## Pixel Pipeline

> **Experimental** — Pixel-space training (not for paper comparison).

```bash
# Using config
uv run python scripts/train_pixel.py \
  --config configs/pixel/smoke_feature.yaml \
  --output-dir outputs/pixel_smoke

# With MAE encoder
uv run python scripts/train_pixel.py \
  --config configs/pixel/smoke_feature_queue_mae.yaml \
  --output-dir outputs/pixel_mae
```

### MAE + Pixel Combined
```bash
# Train MAE encoder first
uv run python scripts/train_mae.py \
  --device cpu \
  --steps 3 \
  --in-channels 3 \
  --base-channels 8 \
  --stages 2 \
  --export-encoder-path outputs/stage7_mae_encoder_for_pixel/mae_encoder.pt \
  --output-dir outputs/stage7_mae_encoder_for_pixel

# Then train pixel with loaded encoder
uv run python scripts/train_pixel.py \
  --device cpu \
  --steps 2 \
  --use-feature-loss \
  --feature-encoder mae \
  --feature-base-channels 8 \
  --feature-stages 2 \
  --feature-selected-stages 0 1 \
  --mae-encoder-path outputs/stage7_mae_encoder_for_pixel/mae_encoder.pt \
  --output-dir outputs/stage7_pixel_mae_encoder_load_smoke
```

---

## MAE Training

> Train Masked Autoencoder feature encoders.

```bash
# Latent space MAE
uv run python scripts/train_mae.py \
  --config configs/mae/smoke_latent.yaml \
  --output-dir outputs/mae_smoke
```

---

## Evaluation & Sampling

### FID/IS Evaluation

```bash
# Tensor-to-tensor evaluation (smoke test)
uv run python scripts/eval_fid_is.py \
  --device cpu \
  --inception-weights none \
  --reference-source tensor_file \
  --reference-tensor-file-path outputs/stage8_eval_smoke/ref.pt \
  --generated-source tensor_file \
  --generated-tensor-file-path outputs/stage8_eval_smoke/gen.pt \
  --output-path outputs/stage8_eval_smoke/eval_summary.json

# ImageFolder evaluation (real data)
uv run python scripts/eval_fid_is.py \
  --device cpu \
  --inception-weights pretrained \
  --reference-source imagefolder \
  --reference-imagefolder-root outputs/stage8_cifar10/reference \
  --generated-source imagefolder \
  --generated-imagefolder-root outputs/stage8_cifar10/generated_noisy \
  --output-path outputs/stage8_cifar10/eval_pretrained_noisy.json
```

### Generate Samples

```bash
# Sample from trained checkpoint
uv run python scripts/sample_pixel.py \
  --device cpu \
  --checkpoint-path outputs/<run>/checkpoint.pt \
  --output-root outputs/<run>_samples \
  --n-samples 128 \
  --batch-size 16
```

### End-to-End Workflows

```bash
# Pixel pipeline E2E
uv run python scripts/run_end_to_end_pixel_eval.py \
  --output-root outputs/e2e_pixel_smoke \
  --device cpu \
  --train-steps 2 \
  --sample-count 32 \
  --reference-imagefolder-root outputs/stage8_cifar10/reference \
  --inception-weights none

# Latent pipeline E2E
uv run python scripts/run_end_to_end_latent_eval.py \
  --output-root outputs/e2e_latent_smoke \
  --device cpu \
  --train-steps 2 \
  --sample-count 32 \
  --reference-imagefolder-root outputs/stage8_cifar10/reference \
  --inception-weights none
```

### Advanced Evaluation

```bash
# Alpha sweep (guidance scale search)
uv run python scripts/eval_alpha_sweep.py \
  --mode pixel \
  --checkpoint-path outputs/<run>/checkpoint.pt \
  --output-root outputs/<run>_alpha_sweep \
  --alphas 1 2 3 \
  --n-samples 512 \
  --reference-imagefolder-root outputs/stage8_cifar10/reference \
  --inception-weights pretrained \
  --reference-cache

# Evaluate last K checkpoints
uv run python scripts/eval_last_k_checkpoints.py \
  --mode pixel \
  --checkpoint-dir outputs/<run>/checkpoints \
  --k 5 \
  --output-root outputs/<run>_last_k \
  --n-samples 512 \
  --reference-imagefolder-root outputs/stage8_cifar10/reference \
  --inception-weights pretrained \
  --reference-cache
```

---

## Operational Patterns

### Checkpoint Management

```bash
# Save checkpoint every N steps
--save-every 1

# Resume from checkpoint
--checkpoint-path outputs/<run>/checkpoint.pt --save-every 1
--resume-from outputs/<run>/checkpoint.pt

# Load checkpoint for sampling
--checkpoint-path outputs/<run>/checkpoint.pt
```

### Queue Configuration

```bash
# Synthetic dataset queue
--real-batch-source synthetic_dataset \
  --real-dataset-size 4096 \
  --real-loader-batch-size 128

# Enable queue mode
--use-queue
```

### Compilation Options

```bash
# Warn on compile failure (default)
--compile-fail-action warn

# Raise error on compile failure
--compile-fail-action raise

# Disable compilation on failure
--compile-fail-action disable
```

### Legacy Compatibility

```bash
# Use legacy MAE encoder architecture
--mae-encoder-arch legacy_conv
```

---

## Device Selection

| Flag | Behavior | Use When |
|---------|-------------|-------------|
| `--device auto` | Auto: `cuda` → `xpu` → `mps` → `cpu` | Let system decide |
| `--device cuda` | NVIDIA GPU, fails if unavailable | Require NVIDIA |
| `--device cuda:0` | Specific GPU index | Multi-GPU systems |
| `--device mps` | Apple Silicon | macOS on M-series |
| `--device cpu` | CPU only | Testing or no GPU |
| `--device gpu` | Any accelerator, fails fast if none | Require any GPU |

---

## Output Structure

All training commands produce:

```
outputs/<run_name>/
├── checkpoint.pt          # Model checkpoint
├── config.yaml            # Saved configuration
├── metrics.json           # Training metrics
├── samples/               # Generated samples
└── logs/                  # Training logs
```

---

<div align="center">

**Full command reference complete!** See [Getting Started](getting_started.md) for first steps.

</div>
