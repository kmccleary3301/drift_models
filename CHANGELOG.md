# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and this project uses semantic versioning for public releases.

## Release policy

- `MAJOR`: incompatible public API/CLI contract changes.
- `MINOR`: backward-compatible feature additions.
- `PATCH`: backward-compatible fixes and documentation corrections.
- Before `1.0.0`, `0.MINOR` may still include substantial workflow changes; all claim-boundary changes must be documented explicitly in release notes.

## [Unreleased]

### Added
- Public release governance baseline:
  - `LICENSE`
  - `CONTRIBUTING.md`
  - `CODE_OF_CONDUCT.md`
  - `SECURITY.md`
  - `CHANGELOG.md`
- Publicity/release execution planning docs:
  - `docs_tmp/socials/PUBLICITY_STRATEGY_V1_IMPLEMENTATION_PLAN.md`

### Changed
- Packaging metadata in `pyproject.toml`:
  - broadened Python compatibility target,
  - shifted heavy runtime deps into extras,
  - added project metadata/classifiers/urls.

## [0.1.0] - 2026-02-24

### Initial
- Research reproduction baseline for Drifting Models with latent/pixel pipelines, tests, and artifact-centric evaluation workflow.
