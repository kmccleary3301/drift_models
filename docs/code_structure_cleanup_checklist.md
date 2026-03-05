# Code structure cleanup checklist

Date: 2026-03-04  
Scope: repository code and command surface only.

## Goal

Make the repo easy to evaluate in one pass without weakening research depth.

Success means a new engineer can answer these in under 10 minutes:

1. What is the one stable ImageNet lane?
2. Which scripts are claim-facing vs exploratory?
3. Which configs are safe defaults?
4. Where are the quality/time/cost results?

## Hard rules

- One stable lane stays front-and-center.
- Experimental work stays available but clearly marked.
- Public claims come only from stable lane artifacts.
- No breaking of existing run scripts during cleanup.

## Phase A: stable public surface

- [x] Add a single stable ImageNet lane doc.
- [x] Add explicit stable vs experimental boundary doc.
- [x] Add a reproducibility scoreboard with cost/time.
- [x] Add a one-command wrapper for the stable lane.
- [x] Add a newcomer smoke script that verifies prerequisites and exits with clear errors.

## Phase B: scripts taxonomy

- [x] Add `scripts/README.md` with stable and experimental groups.
- [x] Move ablation/benchmark/proxy scripts under `scripts/experimental/` with compatibility wrappers.
- [x] Move eval/pipeline orchestration scripts under `scripts/experimental/` with compatibility wrappers.
- [x] Enforce naming prefixes:
  - stable: `train_*`, `sample_*`, `eval_*`, `runtime_*`
  - experimental: `exp_*` or `ablation_*`
- [x] Add CI guard that fails on new top-level script files without taxonomy registration.

## Phase C: configs taxonomy

- [x] Add `configs/README.md` with stable config list.
- [x] Add `configs/stable/` aliases that point to canonical stable configs.
- [x] Move ablation/recovery templates into `configs/experimental/`.
- [x] Add CI guard that requires new configs to declare `tier: stable|experimental`.

## Phase D: outputs and artifacts

- [x] Define one canonical stable output root pattern:
  - `outputs/imagenet/stable_<timestamp>/...`
- [x] Define one canonical experimental output root pattern:
  - `outputs/imagenet/exp_<name>_<timestamp>/...`
- [x] Add artifact validator script for required files:
  - `RUN.md`, `env_snapshot.json`, `codebase_fingerprint.json`, eval summaries.

## Phase E: documentation tightening

- [x] Keep README command section to stable commands only.
- [x] Move dense command catalogs and legacy flows to deep docs.
- [x] Add “If you only read 3 docs” block:
  - `docs/minimal_repro_imagenet256.md`
  - `docs/stable_vs_experimental.md`
  - `docs/reproducibility_scoreboard.md`

## Phase F: deprecation policy

- [x] Add deprecation matrix for legacy scripts/configs:
  - active, maintenance-only, deprecated.
- [x] For deprecated entrypoints, print migration target on execution.
- [ ] Remove dead paths after one tagged release cycle (release-gated; earliest target from matrix is `v0.3.0`).

## Validation gate

A cleanup cycle is complete only if all pass:

- [x] Fresh clone newcomer test succeeds in < 10 minutes to first stable smoke run.
- [x] Stable lane commands are copy/paste runnable from docs without edits.
- [x] No public docs use deprecated scripts/configs.
- [x] Scoreboard row for latest stable run is complete with artifact links.

Latest fresh-clone evidence (2026-03-05):

- Command: `uv sync --extra dev --extra eval && uv run python scripts/runtime_newcomer_smoke.py --device cpu --stable-steps 1 --timestamp 20260305_060606 --output-root outputs/onboarding/fresh_clone_smoke`
- Wall time (`/usr/bin/time -p`): `real 31.11` seconds
