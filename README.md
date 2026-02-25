# Drifting Models Reproduction (PyTorch)

[![CI](https://img.shields.io/github/actions/workflow/status/kmccleary3301/drift_models/ci.yml?branch=main&label=CI)](https://github.com/kmccleary3301/drift_models/actions/workflows/ci.yml)
[![Nightly](https://img.shields.io/github/actions/workflow/status/kmccleary3301/drift_models/nightly.yml?label=Nightly)](https://github.com/kmccleary3301/drift_models/actions/workflows/nightly.yml)
[![Runtime Health](https://img.shields.io/badge/runtime%20health-preflight%20enabled-2ea44f)](docs/runtime_health.md)
[![PyPI](https://img.shields.io/pypi/v/drift-models)](https://pypi.org/project/drift-models/)
[![Python](https://img.shields.io/pypi/pyversions/drift-models)](https://pypi.org/project/drift-models/)
[![License](https://img.shields.io/github/license/kmccleary3301/drift_models)](./LICENSE)

Community reproduction of *Generative Modeling via Drifting* in PyTorch.

## Project status and claim boundaries

- This repository is **not an official release** from the paper authors.
- We are actively hardening paper-faithful semantics and evidence artifacts.
- We do **not** currently claim full paper-level metric reproduction.
- Pixel pipeline remains **experimental** and should not be treated as parity-closed.

See:
- `docs/faithfulness_status.md`
- `docs/reproduction_report.md`
- `docs/experiment_log.md`
- `docs/eval_contract.md`

## Quickstart (60 seconds)

### Option A: `uv` (recommended)

```bash
uv sync --extra dev --extra eval --extra sdvae
uv run python scripts/runtime_preflight.py --device auto --check-torchvision --strict
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval,sdvae]"
python scripts/runtime_preflight.py --device auto --check-torchvision --strict
python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu
```

## Installation guides

- Linux + NVIDIA CUDA: `docs/install_linux_cuda.md`
- CPU-only: `docs/install_cpu_only.md`
- macOS (Apple Silicon / MPS): `docs/install_macos.md`
- Windows + WSL2: `docs/install_windows_wsl2.md`

## Common workflows

- Toy trajectory training: `docs/getting_started.md`
- Latent smoke training: `docs/getting_started.md`
- Sampling/eval smoke: `docs/getting_started.md`
- Full command catalog: `docs/commands.md`

## Compatibility tiers

Compatibility and support policy is documented in:
- `docs/compatibility_matrix.md`

## Runtime health

- Runtime preflight is enforced in CI on Linux/macOS/Windows and nightly on Linux.
- Preflight JSON reports are uploaded as workflow artifacts for each run.
- CI also generates an aggregated runtime summary + failure triage and posts it as a sticky PR comment.
- Runtime diagnostics guide: `docs/runtime_health.md`
- Local preflight entrypoint: `scripts/runtime_preflight.py`

## Reproducibility and evidence

- Run metadata contracts: `docs/provenance_contract.md`
- Claim/evidence mapping: `docs/claim_to_evidence_matrix.md`
- Release parity gate: `docs/release_gate_checklist.md`
- Public release gate: `docs/RELEASE_CHECKLIST.md`
- Branch protection policy: `docs/branch_protection.md`
- PyPI/TestPyPI publish setup: `docs/pypi_trusted_publishing.md`

## Contributing and governance

- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Changelog: `CHANGELOG.md`

## Citation

If you use this repository, cite the original paper and this implementation repo.

Paper: *Generative Modeling via Drifting* (arXiv:2602.04770).
