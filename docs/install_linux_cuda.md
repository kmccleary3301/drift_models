# Install: Linux + NVIDIA CUDA

## Recommended path (`uv`)

```bash
uv sync --extra dev --extra eval --extra sdvae
```

## `pip` path

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval,sdvae]"
```

## Validate runtime

```bash
uv run python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/linux_cuda.json
```

## Notes

- Prefer explicit device flags (`--device cuda:0`) in long runs.
- If using compile paths, keep `--compile-fail-action warn` for safer fallback behavior.
