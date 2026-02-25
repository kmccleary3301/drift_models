# Install: macOS (Apple Silicon / MPS)

Support tier: **experimental**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval]"
```

## Validate runtime

```bash
python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/macos.json
```

## Recommended runtime flags

- Prefer `--device mps` or `--device cpu` explicitly.
- Prefer compile disabled on MPS unless explicitly testing:
  - do not pass `--compile-generator`
  - keep fallback path behavior explicit

## Fallback behavior

If you hit MPS op gaps, run CPU or use:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Then rerun your command.
