# Runtime Health

This repository continuously validates runtime/device readiness via `scripts/runtime_preflight.py`.

## CI preflight (push + PR)

- Workflow: `.github/workflows/ci.yml`
- Matrix:
  - `ubuntu-latest` / Python `3.12`
  - `macos-latest` / Python `3.12`
  - `windows-latest` / Python `3.12`
- Checks:
  - `--device auto --check-torchvision --strict`
  - `--device cpu --check-torchvision --strict`
- Artifacts uploaded per matrix job:
  - `runtime-preflight-<os>-py3.12`
  - Includes JSON summaries from `outputs/runtime_preflight_ci/*.json`
- Aggregated summary job:
  - Downloads matrix artifacts and builds:
    - `outputs/runtime_preflight_ci/preflight_summary.md`
    - `outputs/runtime_preflight_ci/preflight_summary.json`
  - Uploads artifact: `runtime-preflight-summary`
  - Publishes summary into the GitHub job summary panel
  - On pull requests, posts/updates a sticky PR comment with:
    - report status table
    - severity-ranked failure triage section (top failed checks + backend grouping)
- Experimental non-blocking jobs:
  - `runtime-preflight-experimental (mps)`
  - `runtime-preflight-experimental (rocm)`
  - These jobs are informational and do not gate merges.

## Nightly preflight

- Workflow: `.github/workflows/nightly.yml`
- Runner: `ubuntu-latest` / Python `3.12`
- Checks:
  - `--device auto --check-torchvision --check-compile --strict`
- Artifact:
  - `runtime-preflight-nightly-ubuntu-py3.12`
  - Paths:
    - `outputs/runtime_preflight_nightly/preflight.json`
    - `outputs/runtime_preflight_nightly/preflight_summary.md`
    - `outputs/runtime_preflight_nightly/preflight_summary.json`

## Local run

```bash
uv run python scripts/runtime_preflight.py \
  --device auto \
  --check-torchvision \
  --strict \
  --output-path outputs/runtime_preflight/local.json
```

## Local aggregation

```bash
uv run python scripts/summarize_runtime_preflight.py \
  --input-glob "outputs/runtime_preflight/*.json" \
  --output-md outputs/runtime_preflight/summary.md \
  --output-json outputs/runtime_preflight/summary.json
```

## Reading the report

- `status`: overall pass/fail
- `pass_count` / `fail_count` / `skip_count`: check totals
- `checks[]`: per-check status, duration, and details/error
- `env_fingerprint`: environment snapshot hash payload
- `triage.top_failed_checks[]`: severity-ranked failures (`critical`, `high`, `medium`, `low`)

## Severity triage policy

- `critical` (`90-100`): `device.resolve`, `capabilities.detect`, `tensor.smoke` failures
  - Action: block release candidates and fix before merge
- `high` (`75-89`): compile smoke hard failures on first-class backends
  - Action: either fix or set explicit compile fallback policy and document rationale
- `medium` (`60-74`): optional dependency import failures (`torchvision`, etc.)
  - Action: validate extras/install docs and keep claim boundaries explicit
- `low` (`<60`): non-critical optional stack warnings
  - Action: track in backlog and keep troubleshooting notes current
