# Generator Stability Report (Engineering)

This report is about **training stability and reproducibility mechanics**, not paper-metric parity.

## What “stable” means here
- Runs complete without NaNs/Infs.
- Checkpointing and resume work (including resume-mode ablations).
- Overfit-mode can drive loss below a fixed threshold (regression gate).
- Artifacts are written deterministically (`env_snapshot.json`, `codebase_fingerprint.json`, `RUN.md`, summaries).

## Evidence (current repo)

### 1) Full test suite passes
- `uv run pytest -q` (latest run in this workspace, 2026-02-14): `127 passed`

### 2) Tiny-overfit regression gate exists and is test-covered
- Gate script: `scripts/check_latent_overfit.py`
- Integration test: `tests/integration/test_latent_overfit_gate.py`
- Purpose: verifies `--overfit-fixed-batch` can converge to `loss <= 1e-3` on CPU within `60` steps.

### 3) Generator stability run artifacts (example)
- `outputs/stage2_generator_stability_120/latent_summary.json`
- `outputs/stage2_generator_stability_120/checkpoint.pt`
- `outputs/stage2_generator_stability_120/checkpoints/`

This run demonstrates:
- end-to-end train loop executes under bf16 settings,
- regular step checkpointing,
- stable perf telemetry (`perf.*` fields in the summary JSON).

## Known gaps / follow-ups
- `outputs/stage2_generator_stability_long/` does not currently include a `latent_summary.json` (likely created before summary-writing was standardized). If we want “one place” for long-run stability evidence, re-run that profile with `--output-dir` enabled and archive `RUN.md` + summary.
