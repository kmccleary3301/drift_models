# Compatibility Matrix

This document defines **support tiers** and the public compatibility claims for this repository.

## Support tiers

- **First-class**: actively exercised and expected to work for core workflows.
- **Best-effort**: should work for subsets of workflows; breakage may occur.
- **Experimental**: known instability or incomplete operator/runtime coverage.
- **Unsupported**: not targeted in current roadmap.

## Platform/runtime matrix

| Tier | Platform | Scope | Notes |
| --- | --- | --- | --- |
| First-class | Linux x86_64 + CPU | unit/integration tests, toy/latent/pixel smokes, eval tooling | default CI target |
| First-class | Linux x86_64 + NVIDIA CUDA | primary research training workflows | prefer explicit `--device cuda:0` |
| Best-effort | Windows native + CPU | install + smoke/eval subsets | use WSL2 for heavy workflows |
| Best-effort | Windows + WSL2 | Linux-like CPU/CUDA workflows when properly configured | environment variability expected |
| Experimental | macOS Apple Silicon (MPS) | toy/smoke/sampling subsets | compile paths may be unstable |
| Experimental | Linux + AMD ROCm | toy/smoke subsets; partial training viability | requires host-specific validation |
| Unsupported | Multi-node distributed / TPU-XLA | none | single-process/single-device is current scope |

## Python / package policy

| Component | Current baseline | Notes |
| --- | --- | --- |
| Python | 3.10+ | 3.12 remains primary maintainer environment |
| PyTorch | 2.3+ | exact backend wheel selection is environment-specific |
| torchvision | optional (`eval`/`imagenet` extra) | not required for minimal base install |
| diffusers stack | optional (`sdvae` extra) | only required for SD-VAE related workflows |

## Compile policy

| Backend | Compile policy | Recommended fail action |
| --- | --- | --- |
| CPU | supported | `warn` (or `raise` in strict benchmarking lanes) |
| CUDA | supported | `warn` (or `raise` for controlled performance runs) |
| MPS | best-effort fallback only | `disable` |
| XPU | best-effort fallback only | `disable` |

Repository-wide compile fallback actions:
- `warn`: keep eager path and emit runtime warning
- `raise`: fail fast on unsupported/failed compile path
- `disable`: skip compile attempt entirely

## Known caveats

- `torch.compile` behavior is backend-dependent; use `--compile-fail-action warn` for safe fallback.
- For MPS/XPU lanes use `--compile-fail-action disable` unless explicitly validating backend progress.
- MPS may require fallback behavior (`PYTORCH_ENABLE_MPS_FALLBACK=1`) on unsupported ops.
- Queue/data pipeline configurations can fail if dataset class coverage assumptions are unmet.
- Pixel pipeline remains experimental and should not be interpreted as parity-closed.

## Claim boundaries

Compatibility support does **not** imply paper-level metric parity. Claim scope remains governed by:
- `docs/faithfulness_status.md`
- `docs/reproduction_report.md`
- `docs/eval_contract.md`
