# Getting Started

## First 15 minutes (known-good path)

```bash
uv sync --extra dev --extra eval
uv run python scripts/runtime_preflight.py --device cpu --check-torchvision --strict
uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick --device cpu
uv run pytest -q tests/integration/test_stage2_smoke.py
```

## 1) Environment bootstrap

```bash
uv sync --extra dev --extra eval --extra sdvae
```

## 1.5) Runtime preflight

```bash
uv run python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/getting_started.json
```

## 2) Toy run (CPU)

```bash
uv run python scripts/train_toy.py \
  --config configs/toy/quick.yaml \
  --output-dir outputs/toy_quick \
  --device cpu
```

## Device selection

- Every CLI command now shares the same `--device` behavior.
- `--device auto` uses runtime priority: `cuda` → `xpu` → `mps` → `cpu`.
- You can target specific accelerators directly (for example: `--device cuda:1`).
- Use `--device gpu` to request any available accelerator backend (fails fast if none exists).

## 3) Latent smoke run (CPU)

```bash
uv run python scripts/train_latent.py --config configs/latent/smoke_feature.yaml
```

## 4) Pixel smoke run (CPU, experimental)

```bash
uv run python scripts/train_pixel.py --config configs/pixel/smoke_feature.yaml
```

## 5) Eval smoke

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

## Next docs

- Install guides: `docs/install_linux_cuda.md`, `docs/install_cpu_only.md`, `docs/install_macos.md`, `docs/install_windows_wsl2.md`
- Compatibility policy: `docs/compatibility_matrix.md`
- Runtime diagnostics: `docs/runtime_health.md`
- Troubleshooting: `docs/troubleshooting.md`
- Full command catalog: `docs/commands.md`
