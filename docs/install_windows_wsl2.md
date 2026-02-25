# Install: Windows + WSL2

Support tier:
- Windows native: **best-effort** (CPU-oriented)
- WSL2: recommended for research workflows

## WSL2 install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,eval,sdvae]"
```

## Validate runtime

```bash
python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/wsl2.json
```

## Notes

- For reproducibility runs and large data workflows, use WSL2/Linux paths consistently.
- Prefer explicit `--device cpu` unless CUDA is configured and verified inside WSL2.
