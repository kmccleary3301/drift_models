# Getting Started

Setup and first-run instructions for the Drifting Models repository.

---

## Known-Good Path

```bash
# 1. Sync dependencies
uv sync --extra dev --extra eval

# 2. One-command onboarding smoke (recommended)
uv run python scripts/runtime_newcomer_smoke.py --device cpu

# 3. Run your first toy training directly (optional)
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu

# 4. Run integration smoke tests
uv run pytest -q tests/integration/test_stage2_smoke.py
```

Expected time: ~5 minutes on CPU. Output: onboarding summary under `outputs/onboarding/newcomer_smoke/` plus stable latent smoke artifacts.

---

## Setup Checklist

| Step | Action | Time | Details |
|------|----------|---------|------------|
| 1 | Environment bootstrap | 30s | [Environment Bootstrap](#environment-bootstrap) |
| 2 | Newcomer smoke (one command) | 5min | [Known-Good Path](#known-good-path) |
| 3 | Runtime preflight | 15s | [Runtime Preflight](#runtime-preflight) |
| 4 | Toy training run | 2min | [Toy Run (CPU)](#toy-run-cpu) |
| 5 | Full training | Varies | [Commands](commands.md) |

---

## Environment Bootstrap

### Option A: `uv` (Recommended)

```bash
uv sync --extra dev --extra eval --extra sdvae
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,eval,sdvae]"
```

### Option C: PyPI (Users Only)

```bash
pip install drift-models
```

---

## Runtime Preflight

Verify your environment before starting long training runs:

```bash
uv run python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/getting_started.json
```

| Exit Code | Meaning | Action |
|:------------:|:-----------|:-----------|
| `0` | All checks passed | Proceed to training |
| `1` | Warning | Check logs, may proceed with caution |
| `2` | Critical failure | Fix issues before training |

**Device selection:**

| Flag | Behavior |
|---------|-------------|
| `--device auto` | Auto-detect: `cuda` → `xpu` → `mps` → `cpu` |
| `--device cuda` | NVIDIA GPU (fails if unavailable) |
| `--device cuda:1` | Specific GPU index |
| `--device mps` | Apple Silicon |
| `--device cpu` | CPU only |

---

## Toy Run (CPU)

Minimal training run to verify the installation:

```bash
uv run python scripts/train_toy.py \
  --config configs/toy/quick.yaml \
  --output-dir outputs/toy_quick \
  --device cpu
```

| Output | Description |
|-----------|--------------|
| `outputs/toy_quick/final_model.pt` | Trained checkpoint |
| `outputs/toy_quick/samples/` | Generated samples |
| `outputs/toy_quick/metrics.json` | Training metrics |

---

## Pipeline Smoke Tests

### Latent Pipeline (Recommended)

```bash
uv run python scripts/train_latent.py \
  --config configs/latent/smoke_feature.yaml \
  --output-dir outputs/latent_smoke
```

### Pixel Pipeline (Experimental)

```bash
uv run python scripts/train_pixel.py \
  --config configs/pixel/smoke_feature.yaml \
  --output-dir outputs/pixel_smoke
```

The pixel pipeline is experimental and should not be used for paper comparison.

---

## Evaluation Smoke Test

```bash
uv run python scripts/eval_fid_is.py \
  --device cpu \
  --inception-weights none \
  --reference-source tensor_file \
  --reference-tensor-file-path outputs/stage8_eval_smoke/ref.pt \
  --generated-source tensor_file \
  --generated-tensor-file-path outputs/stage8_eval_smoke/gen.pt \
  --output-path outputs/stage8_eval_smoke/eval_summary.json
```

| Metric | Expected | Description |
|-----------|:-----------:|----------------|
| FID | Varies | Fréchet Inception Distance (lower = better) |
| IS | Varies | Inception Score (higher = better) |

---

## Generating Samples

After running a latent smoke train, generate images from that checkpoint:

```bash
uv run python scripts/train_latent.py \
  --config configs/latent/smoke_feature_queue.yaml \
  --output-dir outputs/latent_smoke

uv run python scripts/sample_latent.py \
  --checkpoint-path outputs/latent_smoke/checkpoint.pt \
  --output-root outputs/latent_smoke_samples \
  --n-samples 512 \
  --batch-size 32 \
  --decode-mode sd_vae \
  --decode-image-size 256 \
  --write-imagefolder
```

---

## Next Steps

| Topic | Link | When to Read |
|----------|---------|-----------------|
| Linux + CUDA | [Linux + CUDA](install_linux_cuda.md) | Primary development platform |
| macOS | [macOS](install_macos.md) | Apple Silicon users |
| Windows + WSL2 | [Windows + WSL2](install_windows_wsl2.md) | Windows users |
| CPU Only | [CPU Only](install_cpu_only.md) | No GPU available |
| Full Commands | [Commands](commands.md) | Ready for full training |
| Faithfulness Status | [Faithfulness](faithfulness_status.md) | Understanding claims |
| Troubleshooting | [Troubleshooting](troubleshooting.md) | Something went wrong |

---

## Common Issues

| Problem | Solution |
|------------|-------------|
| CUDA out of memory | Use `--device cpu` or reduce batch size |
| Import errors | Run `uv sync` or `pip install -e ".[dev,eval]"` |
| Preflight fails | Check [Runtime Health](runtime_health.md) |
| Slow training | Expected on CPU — use GPU for real training |

---

For the full command catalog, see [Commands](commands.md).
