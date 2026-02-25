# CI/Release Local Validation Snapshot (2026-02-25)

## Scope

- Validate workflow YAML syntax.
- Validate runtime preflight + fallback semantics.
- Validate packaging build/check.
- Validate wheel install smoke from built artifact.

## Commands and outcomes

- Workflow YAML parse:
  - `python` + `yaml.safe_load` on:
    - `.github/workflows/ci.yml`
    - `.github/workflows/nightly.yml`
    - `.github/workflows/release.yml`
  - Outcome: pass

- Targeted runtime/compile tests:
  - `uv run pytest -q tests/unit/test_runtime_utils.py tests/unit/test_compile_toggle_helpers.py tests/integration/test_runtime_preflight_smoke.py`
  - Outcome: pass (`23 passed`)

- Packaging build/check:
  - `uv run python -m build`
  - `uv run twine check dist/*`
  - Outcome: pass

- Wheel smoke install:
  - Created smoke venv: `outputs/packaging_smoke_venv`
  - Install command used explicit temp/cache on `drive_4` to avoid root `/tmp` exhaustion:
    - `TMPDIR=outputs/tmp_pip PIP_CACHE_DIR=outputs/pip_cache .../pip install dist/drifting_models_repro-0.1.0-py3-none-any.whl`
  - Runtime smoke:
    - import package, print `drifting_models.__version__`
    - `resolve_device("cpu")`
  - Outcome: pass

- Editable install path smoke:
  - `uv sync --extra dev --extra eval --extra sdvae`
  - `uv run python scripts/train_toy.py --config configs/toy/quick.yaml --output-dir outputs/toy_quick_smoke --device cpu`
  - Outcome: pass

## Runtime evidence bundle

- Consolidated runtime evidence:
  - `outputs/runtime_preflight/closure/preflight_auto_compile_warn.json`
  - `outputs/runtime_preflight/closure/preflight_cpu_compile_disable.json`
  - `outputs/runtime_preflight/closure/preflight_summary.md`
  - `outputs/runtime_preflight/closure/preflight_summary.json`
