# Changelog

All notable changes to this project are documented here.

Format inspired by [Keep a Changelog](https://keepachangelog.com/). This project uses [Semantic Versioning](https://semver.org/).

---

## Release Policy

| Version | Change Type | Scope |
|:-----------:|----------------|----------|
| `MAJOR` | Breaking | Incompatible API/CLI changes |
| `MINOR` | Feature | Backward-compatible additions |
| `PATCH` | Fix | Backward-compatible fixes & docs |

> **Pre-1.0.0:** `0.MINOR` may include substantial workflow changes. All claim-boundary changes are documented explicitly.

---

## [Unreleased]

### Coming Soon
- Full ImageNet latent pipeline runs
- Paper-level metric reproduction
- Improved pixel pipeline stability

---

## [0.1.1] - 2026-03-02

### Fixed
| Issue | Fix |
|----------|--------|
| Queue resume reporting | Stabilized checkpoint resume emits `queue_warmup_report` with `mode=resume_restore`; nightly tests pass |

### Changed
| Change | Description |
|-----------|---------------|
| Repository footprint | Untracked large artifacts: `Drift_Models.pdf`, `Drift_Models/Drift_Models.json` |
| MAE export reliability | Added MAE optimizer selection; `w640` config uses SGD to avoid AdamW OOM |

---

## [0.1.0] - 2026-02-24 - Initial Release

### Features
| Component | Description |
|--------------|---------------|
| Latent Pipeline | Primary training pipeline with SD-VAE |
| Pixel Pipeline | Experimental pixel-space training |
| Toy Pipeline | Quick sanity checks |
| MAE Encoder | Feature encoder training |
| Evaluation | FID/IS metrics with Inception |
| Tests | Unit + integration test suite |
| PyPI | `pip install drift-models` |

---

<div align="center">

**Follow releases:** [Releases](https://github.com/kmccleary3301/drift_models/releases)

</div>
