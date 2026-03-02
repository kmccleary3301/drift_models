# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and this project uses semantic versioning for public releases.

## Release policy

- `MAJOR`: incompatible public API/CLI contract changes.
- `MINOR`: backward-compatible feature additions.
- `PATCH`: backward-compatible fixes and documentation corrections.
- Before `1.0.0`, `0.MINOR` may still include substantial workflow changes; all claim-boundary changes must be documented explicitly in release notes.

## [Unreleased]

## [0.1.1] - 2026-03-02

### Fixed
- Stabilized latent queue resume reporting in `scripts/train_latent.py` so checkpoint resume emits a schema-consistent
  `queue_warmup_report` (`mode=resume_restore`) and nightly checkpoint-resume integration tests pass again.

### Changed
- Slimmed git-tracked repository footprint by untracking large local-only paper convenience artifacts:
  - `Drift_Models.pdf`
  - `Drift_Models/Drift_Models.json`
  and documenting this policy in `README.md`.
- Hardened MAE width-parity export reliability on the current single-GPU host by adding MAE optimizer selection
  support and running the `w640` bootstrap config with SGD to avoid AdamW optimizer-step OOM.

## [0.1.0] - 2026-02-24

### Initial
- Research reproduction baseline for Drifting Models with latent/pixel pipelines, tests, and artifact-centric evaluation workflow.
