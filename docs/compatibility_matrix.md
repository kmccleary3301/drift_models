# Compatibility Matrix

> Support tiers and platform compatibility for Drifting Models.

---

## Support Tiers

| Tier | Definition | Expectation |
|---------|-------------|---------------|
| First-class | Actively exercised | Core workflows guaranteed to work |
| Best-effort | Should work | Subsets only; breakage may occur |
| Experimental | Known instability | Incomplete coverage, use with caution |
| Unsupported | Not targeted | Out of current scope |

---

## Platform Support

| Tier | Platform | Accelerator | Scope | Notes |
|:-------:|-------------|----------------|----------|----------|
| ⭐ | Linux x86_64 | CPU | Tests, toy/latent/pixel smokes, eval | Default CI target |
| ⭐ | Linux x86_64 | NVIDIA CUDA | Research training workflows | Use `--device cuda:0` |
| 🔶 | Windows Native | CPU | Install + smoke/eval subsets | Use WSL2 for heavy work |
| 🔶 | Windows + WSL2 | CUDA/CPU | Linux-like workflows | Environment may vary |
| 🧪 | macOS | Apple Silicon | Toy/smoke/sampling | Compile paths unstable |
| 🧪 | Linux | AMD ROCm | Toy/smoke only | Requires host validation |
| ❌ | Multi-node | TPU-XLA | None | Single-device only |

---

## Python & Package Policy

| Component | Baseline | Notes |
|--------------|:-----------:|----------|
| Python | 3.10+ | 3.12 recommended for dev |
| PyTorch | 2.3+ | Backend wheel environment-specific |
| torchvision | Optional | `eval`/`imagenet` extra |
| diffusers | Optional | `sdvae` extra for SD-VAE workflows |

---

## torch.compile Policy

| Backend | Compile Support | Recommended Action |
|------------|:----------------:|---------------------|
| CPU | Supported | `--compile-fail-action warn` |
| NVIDIA CUDA | Supported | `--compile-fail-action warn` |
| Apple MPS | Fallback only | `--compile-fail-action disable` |
| Intel XPU | Fallback only | `--compile-fail-action disable` |

### Fail Action Options

| Flag | Behavior |
|---------|-------------|
| `warn` | Keep eager path, emit warning |
| `raise` | Fail fast on compile failure |
| `disable` | Skip compile attempt entirely |

---

## Known Caveats

| Issue | Mitigation |
|----------|-------------|
| `torch.compile` backend-dependent | Use `--compile-fail-action warn` for safe fallback |
| MPS/XPU compile issues | Use `--compile-fail-action disable` |
| MPS unsupported ops | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Queue dataset assumptions | Validate dataset class coverage |
| Pixel pipeline | Experimental — not parity-closed |

---

## Claim Boundaries

> **Compatibility support ≠ paper-level metric parity**

See also:
- [faithfulness_status.md](faithfulness_status.md)
- [reproduction_report.md](reproduction_report.md)
- [eval_contract.md](eval_contract.md)

---

<div align="center">

**Platform support matrix complete** — choose your tier wisely!

</div>
