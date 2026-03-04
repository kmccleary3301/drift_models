# Getting Started

> Your first steps with Drifting Models — from zero to running in under 5 minutes.

---

## First 15 Minutes (Known-Good Path)

```bash
# 1. Sync dependencies
uv sync --extra dev --extra eval

# 2. Verify your environment
uv run python scripts/runtime_preflight.py --device cpu --check-torchvision --strict

# 3. Run your first toy training
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu

# 4. Run integration smoke tests
uv run pytest -q tests/integration/test_stage2_smoke.py
```

**Expected time:** ~2 minutes on CPU  
**Expected output:** Training loss decreasing, final model saved to `outputs/toy_quick/`

---

## Setup Checklist

| Step | Action | Time | Details |
|------|----------|---------|------------|
| 1 | Environment bootstrap | 30s | [Environment Bootstrap](#environment-bootstrap) |
| 2 | Runtime preflight | 15s | [Runtime Preflight](#runtime-preflight) |
| 3 | Toy training run | 2min | [Toy Run (CPU)](#toy-run-cpu) |
| 4 | Smoke tests | 30s | [Smoke Tests](#smoke-tests) |
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
| `0` | All checks passed | Ready to train! |
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

The fastest way to verify everything works:

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

**Note:** Pixel pipeline is experimental and should not be used for paper comparison.

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

After training, generate images from your checkpoint:

```bash
uv run python scripts/sample.py \
  --checkpoint outputs/toy_quick/final_model.pt \
  --output-dir samples/ \
  --num-samples 16
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

<div align="center">

**Ready to dive in?** Check out [Commands](commands.md) for full training workflows!

</div>
