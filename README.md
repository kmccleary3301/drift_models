# Drifting Models

[![CI](https://img.shields.io/github/actions/workflow/status/kmccleary3301/drift_models/ci.yml?branch=main&label=CI&logo=github&color=2088FF)](https://github.com/kmccleary3301/drift_models/actions/workflows/ci.yml)
[![Nightly](https://img.shields.io/github/actions/workflow/status/kmccleary3301/drift_models/nightly.yml?label=Nightly&logo=github&color=6f42c1)](https://github.com/kmccleary3301/drift_models/actions/workflows/nightly.yml)
[![PyPI](https://img.shields.io/pypi/v/drift-models?logo=pypi&color=yellow)](https://pypi.org/project/drift-models/)
[![Python](https://img.shields.io/pypi/pyversions/drift-models?logo=python&color=3776AB)](https://pypi.org/project/drift-models/)
[![License](https://img.shields.io/github/license/kmccleary3301/drift_models?color=2ea44f)](./LICENSE)

Community PyTorch reproduction of [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770) (Deng et al., 2026). This repository provides a concrete, installable implementation of the drifting objective for inspection and experimentation.

---

## Project Overview

```
drift_models/
├── 📦 drifting_models/       Core package (pip install drift-models)
│   ├── models/               DiT-like generative architectures
│   ├── train/                Training loops, drift loss, queue pipeline
│   ├── eval/                 FID / Inception Score evaluation
│   ├── sampling/             One-step image generation
│   └── utils/                Device helpers, preflight, I/O
├── ⚙️ configs/               YAML training configurations
│   ├── latent/               Latent-space pipeline (primary)
│   ├── pixel/                Pixel-space pipeline (experimental)
│   └── toy/                  2D sanity-check distributions
├── 📄 docs/                  Documentation & claim boundaries
├── 🔧 scripts/               CLI entry points for train / eval / sample
└── 📤 dist/                  PyPI release artifacts
```

---

## Key Results

| Metric | Drifting | DiT-XL/2 | Improvement |
|--------|:--------:|:--------:|:-------------:|
| ImageNet 256×256 FID | **1.54** | 2.27 | **1 step vs 250 steps** |
| Inference Steps | **1** | 250 | 250× fewer steps |
| Parameters | 463M | 675M | 31% smaller |

---

## Quickstart

### Option A: `uv` (recommended)

```bash
uv sync --extra dev --extra eval --extra sdvae
uv run python scripts/runtime_preflight.py --device auto --strict
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval,sdvae]"
python scripts/runtime_preflight.py --device auto --strict
python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu
```

### Option C: PyPI

```bash
pip install drift-models
```

---

## Documentation Navigator

### If You Only Read 3 Docs

1. [Minimal Repro Lane](docs/minimal_repro_imagenet256.md)
2. [Stable vs Experimental](docs/stable_vs_experimental.md)
3. [Reproducibility Scoreboard](docs/reproducibility_scoreboard.md)

| Topic | Link | Description |
|----------|---------|----------------|
| Getting Started | [Getting Started](docs/getting_started.md) | Toy runs, smoke tests, first training |
| Installation | [Linux + CUDA](docs/install_linux_cuda.md) | Platform-specific setup guides |
| Commands | [Commands](docs/commands.md) | Full command catalog |
| Faithfulness | [Faithfulness](docs/faithfulness_status.md) | What we claim vs. what's proven |
| Evaluation | [Eval Contract](docs/eval_contract.md) | How we measure quality |
| Compatibility | [Compatibility](docs/compatibility_matrix.md) | Supported platforms & backends |
| Runtime Health | [Runtime Health](docs/runtime_health.md) | Preflight diagnostics |
| Lifecycle Status | [Deprecation Matrix](docs/deprecation_matrix.md) | Active vs maintenance vs deprecated paths |
| Reproduction | [Reproduction Report](docs/reproduction_report.md) | Current results vs. paper |

---

## How Drifting Differs from Diffusion

| Property | Traditional Diffusion | Drifting |
|-----------|--------------------------|----------|
| Inference steps | 20–100 iterative passes | Single forward pass |
| Per-step cost | Full model evaluation each step | One evaluation total |
| Inference mechanism | ODE/SDE solvers at generation time | Drift field absorbed during training |

The core idea is to push distribution evolution into the training phase so that inference reduces to a single network evaluation.

---

## Project Status

| Scope | Status |
|-----------|-----------|
| Community reproduction of the drifting objective | 🟢 Active |
| Mechanical faithfulness to the paper | 🟢 Implemented |
| Latent pipeline | 🟡 Stable; parity hardening in progress |
| Pixel pipeline | 🟡 Experimental |
| Full metric parity with paper | 🔴 Pending long-horizon runs |

This is not official author code. See [Faithfulness Status](docs/faithfulness_status.md) for the full claim-to-evidence mapping.

---

## Platform Support

| Platform | Tier | Accelerator | Status |
|-------------|---------|---------------|-----------|
| Linux | Primary | NVIDIA CUDA | 🟢 Full support |
| Linux | Primary | CPU | 🟢 CI tested |
| macOS | Secondary | Apple Silicon (MPS) | 🟡 CI tested |
| Windows | Secondary | WSL2 + CUDA | 🟡 CI tested |
| Windows | Secondary | Native CPU | 🟡 CI tested |

---

## Common Workflows

```bash
# Newcomer end-to-end smoke (preflight + toy + stable latent smoke)
uv run python scripts/runtime_newcomer_smoke.py --device cpu

# One-command stable lane (preflight + train + artifact validation)
uv run python scripts/runtime_stable_lane.py --device cpu

# Runtime preflight
uv run python scripts/runtime_preflight.py --device auto --check-torchvision --strict

# Toy sanity check (CPU)
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu

# Stable-lane latent run root
OUT=$(uv run python scripts/make_run_root.py --lane stable --base-dir outputs/imagenet | python -c "import json,sys; print(json.load(sys.stdin)['run_root'])")
mkdir -p "${OUT}"

# Stable-lane latent train
uv run python scripts/train_latent.py \
  --config configs/stable/latent_smoke_feature_queue.yaml \
  --device cuda:0 \
  --output-dir "${OUT}" \
  --checkpoint-dir "${OUT}/checkpoints" \
  --checkpoint-path "${OUT}/checkpoint.pt"

# Validate artifact bundle
uv run python scripts/validate_run_artifacts.py --run-root "${OUT}" --lane stable --allow-missing-eval-summaries
```

Use deep docs for dense catalogs and legacy/experimental flows:

- [Commands](docs/commands.md)
- [ImageNet Runbook](docs/imagenet_runbook.md)
- [Experiment Log](docs/experiment_log.md)

---

## Runtime Health & CI

| Check | Frequency | Output |
|----------|-------------|-----------|
| Preflight diagnostics | Every run | JSON capability report |
| CI matrix | Push/PR | Linux, macOS, Windows |
| Nightly runs | Daily | Full integration tests |
| Coverage | Weekly | Test coverage reports |

```bash
python scripts/runtime_preflight.py --device auto --check-torchvision --strict
```

---

## Contributing

| Resource | Link |
|-------------|---------|
| Contributing Guide | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Code of Conduct | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Security Policy | [SECURITY.md](SECURITY.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

---

## Citation

If you use this repository, please cite both the original paper and this implementation:

### Original Paper

```bibtex
@article{deng2026drifting,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```

### This Implementation

```bibtex
@software{drift_models2026,
  title={Drift Models: Community PyTorch Reproduction},
  author={McCleary, Kyle},
  url={https://github.com/kmccleary3301/drift_models},
  year={2026}
}
```

---

## Quick Links

| Resource | Location |
|-------------|-------------|
| Paper | [arXiv:2602.04770](https://arxiv.org/abs/2602.04770) |
| Repository | [github.com/kmccleary3301/drift_models](https://github.com/kmccleary3301/drift_models) |
| PyPI Package | [pypi.org/project/drift-models](https://pypi.org/project/drift-models/) |
| Twitter/X | [@kyle_mccleary](https://x.com/kyle_mccleary) |

---

<div align="center">

*Licensed under MIT. Contributions welcome.*

</div>
