# Output & Artifact Contract

This document defines canonical output root naming and required artifact files.

## Canonical run-root patterns

- **Stable lane**: `outputs/imagenet/stable_<timestamp>/`
- **Experimental lane**: `outputs/imagenet/exp_<name>_<timestamp>/`
- Timestamp format: `YYYYMMDD_HHMMSS` (UTC)

Examples:

- `outputs/imagenet/stable_20260304_231501`
- `outputs/imagenet/exp_table8_proxy_20260304_231742`

## Create canonical roots

Use `scripts/make_run_root.py`:

```bash
uv run python scripts/make_run_root.py --lane stable --base-dir outputs/imagenet
uv run python scripts/make_run_root.py --lane experimental --name table8_proxy --base-dir outputs/imagenet
```

Add `--mkdir` to create the folder.

## Required core artifacts

Every claim-facing run root must include:

- `RUN.md`
- `env_snapshot.json`
- `codebase_fingerprint.json`

Eval summaries must also be present (for example `eval/eval_summary.json` or claim-bundle eval summaries).

## Validate artifact bundles

Use `scripts/validate_run_artifacts.py`:

```bash
uv run python scripts/validate_run_artifacts.py \
  --run-root outputs/imagenet/stable_20260304_231501 \
  --lane stable
```

For experimental runs:

```bash
uv run python scripts/validate_run_artifacts.py \
  --run-root outputs/imagenet/exp_table8_proxy_20260304_231742 \
  --lane experimental
```

Optional:

- `--output-json <path>` writes validation report JSON.
- `--allow-missing-eval-summaries` skips eval-summary presence checks.
