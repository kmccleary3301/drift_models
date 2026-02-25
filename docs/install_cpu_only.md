# Install: CPU-only

## `uv`

```bash
uv sync --extra dev --extra eval
```

## `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval]"
```

## Smoke test

```bash
uv run python scripts/runtime_preflight.py \
  --device cpu \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/cpu_only.json

uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick_cpu --device cpu
```
