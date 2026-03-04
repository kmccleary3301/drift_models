# Drifting Models Documentation

> Your guide to understanding, installing, and using Drifting Models.

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
| [Faithfulness](faithfulness_status.md) | What we claim vs. reality | High |
| [Reproduction Report](reproduction_report.md) | Results vs. paper | High |
| [Eval Contract](eval_contract.md) | How we measure quality | Medium |
| [Claims & Evidence](claim_to_evidence_matrix.md) | Specific claim mapping | High |
| [Experiment Log](experiment_log.md) | Training run history | High |

### Technical Reference
| Document | Description | Audience |
|-------------|---------------|:-----------:|
| [Runtime Health](runtime_health.md) | Preflight diagnostics | Users |
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
1. faithfulness_status.md     → 5 min
2. reproduction_report.md    → 10 min
3. eval_contract.md          → 5 min
4. Full training runs
5. Result analysis
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
| Installation help | [Installation](#installation) section |
| Command syntax | [commands.md](commands.md) |
| Platform support | [compatibility_matrix.md](compatibility_matrix.md) |
| What we claim | [faithfulness_status.md](faithfulness_status.md) |
| Current results | [reproduction_report.md](reproduction_report.md) |
| Common errors | [troubleshooting.md](troubleshooting.md) |
| Why decisions made | [decision_log.md](decision_log.md) |

---

## Need Help?

| Question | Resource |
|-----------|-------------|
| "Does my setup work?" | Run `scripts/runtime_preflight.py` |
| "How do I...?" | Check [commands.md](commands.md) |
| "Is this a bug?" | See [troubleshooting.md](troubleshooting.md) |
| "Can I claim X?" | Read [faithfulness_status.md](faithfulness_status.md) |
| "Want to contribute?" | See [CONTRIBUTING.md](../CONTRIBUTING.md) |

---

<div align="center">

**Start with [Getting Started](getting_started.md) or jump to [Commands](commands.md)**

</div>
