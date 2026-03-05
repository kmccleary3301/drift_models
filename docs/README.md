# Documentation

Reference documentation for the Drifting Models repository.

---

## Documentation Map

### Getting Started
| Document | Description | Read Time |
|-------------|---------------|:------------:|
| [Getting Started](getting_started.md) | First steps & quickstart | 5 min |
| [Commands](commands.md) | Complete command reference | 10 min |
| [Troubleshooting](troubleshooting.md) | Common issues & fixes | 5 min |

### Installation
| Document | Description | Platform |
|-------------|---------------|:-----------:|
| [Linux + CUDA](install_linux_cuda.md) | NVIDIA GPU setup | Linux |
| [Windows + WSL2](install_windows_wsl2.md) | Windows with WSL2 | Windows |
| [macOS](install_macos.md) | Apple Silicon setup | macOS |
| [CPU Only](install_cpu_only.md) | No GPU required | All |
| [Compatibility](compatibility_matrix.md) | Platform support tiers | All |

### Research & Reproduction
| Document | Description | Detail |
|-------------|---------------|:---------:|
| [Code Structure Cleanup](code_structure_cleanup_checklist.md) | Execution checklist for code surface cleanup | High |
| [Minimal Repro Lane](minimal_repro_imagenet256.md) | Single-GPU ImageNet-256 baseline lane | High |
| [Stable vs Experimental](stable_vs_experimental.md) | Support boundary for scripts/configs | High |
| [Reproducibility Scoreboard](reproducibility_scoreboard.md) | Quality/time/cost run table | High |
| [Output & Artifact Contract](output_artifact_contract.md) | Canonical run roots + required artifact files | High |
| [Deprecation Matrix](deprecation_matrix.md) | Active vs maintenance vs deprecated paths | High |
| [Faithfulness](faithfulness_status.md) | What we claim vs. reality | High |
| [Reproduction Report](reproduction_report.md) | Results vs. paper | High |
| [Eval Contract](eval_contract.md) | How we measure quality | Medium |
| [Claims & Evidence](claim_to_evidence_matrix.md) | Specific claim mapping | High |
| [Experiment Log](experiment_log.md) | Training run history | High |

### Technical Reference
| Document | Description | Audience |
|-------------|---------------|:-----------:|
| [Scripts Surface Map](../scripts/README.md) | Stable vs exploratory script entrypoints | Developers |
| [Experimental Scripts](../scripts/experimental/README.md) | Canonical layout for exploratory runners | Developers |
| [Configs Surface Map](../configs/README.md) | Stable vs exploratory config selection | Developers |
| [Runtime Health](runtime_health.md) | Preflight diagnostics | Users |
| [Output & Artifact Contract](output_artifact_contract.md) | Run-root naming and artifact validation | Users |
| [Deprecation Matrix](deprecation_matrix.md) | Lifecycle status for scripts and configs | Users |
| [Decision Log](decision_log.md) | Why we chose X over Y | Developers |
| [Decisions Status](decision_closure_status.md) | Open/closed decisions | Developers |
| [Provenance](provenance_contract.md) | Run metadata contracts | Researchers |

---

## Learning Paths

### New User Path
```
1. getting_started.md        → 5 min
2. install_*.md (your OS)    → 10 min
3. Run toy training           → 2 min
4. commands.md               → 10 min
5. Latent smoke test         → 5 min
```

### Researcher Path
```
1. minimal_repro_imagenet256.md  → 10 min
2. stable_vs_experimental.md     → 5 min
3. faithfulness_status.md        → 5 min
4. reproduction_report.md        → 10 min
5. reproducibility_scoreboard.md → 5 min
```

### Contributor Path
```
1. decision_log.md           → 10 min
2. decision_closure_status.md → 5 min
3. runtime_health.md         → 5 min
4. CONTRIBUTING.md           → 5 min
5. Make changes + test
```

---

## Quick Search

| Looking For | Found In |
|----------------|-------------|
| Current cleanup execution checklist | [code_structure_cleanup_checklist.md](code_structure_cleanup_checklist.md) |
| Stable script entrypoints | [../scripts/README.md](../scripts/README.md) |
| Stable config defaults | [../configs/README.md](../configs/README.md) |
| One clean ImageNet lane | [minimal_repro_imagenet256.md](minimal_repro_imagenet256.md) |
| Stable vs experimental boundary | [stable_vs_experimental.md](stable_vs_experimental.md) |
| Run quality/time/cost table | [reproducibility_scoreboard.md](reproducibility_scoreboard.md) |
| Output naming + artifact checks | [output_artifact_contract.md](output_artifact_contract.md) |
| Path lifecycle status | [deprecation_matrix.md](deprecation_matrix.md) |
| Installation help | [Installation](#installation) section |
| Command syntax | [commands.md](commands.md) |
| Platform support | [compatibility_matrix.md](compatibility_matrix.md) |
| What we claim | [faithfulness_status.md](faithfulness_status.md) |
| Current results | [reproduction_report.md](reproduction_report.md) |
| Common errors | [troubleshooting.md](troubleshooting.md) |
| Why decisions made | [decision_log.md](decision_log.md) |

---

## Quick Reference

| Question | Resource |
|-----------|-------------|
| Does my setup work? | Run `scripts/runtime_preflight.py` |
| Command syntax? | [commands.md](commands.md) |
| Something broken? | [troubleshooting.md](troubleshooting.md) |
| What can we claim? | [faithfulness_status.md](faithfulness_status.md) |
| Want to contribute? | [CONTRIBUTING.md](../CONTRIBUTING.md) |
