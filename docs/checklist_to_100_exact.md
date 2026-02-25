# Checklist To Exactly 100% Overall Completion (Executable)

This is the **full closure** checklist for reaching **exactly 100%** on the staged rubric in `docs/reproduction_report.md`, with every item backed by an artifact, a command, or a decision record.

Conventions:
- Each item has an **ID** so it can be tracked in logs and the plan tool.
- “Artifact” means a path under `outputs/` (or `docs/`) that demonstrates the claim.
- “Decision” means a new entry in `docs/decision_log.md` with evidence or an explicit deferral + impact.

---

## S0 — Foundation + Environment (5%)
- [ ] **S0.1** Post-restart sanity: confirm `uv` works, `.venv` present, `torch.cuda.device_count()` recorded.
- [ ] **S0.2** Run full test suite and archive results: `uv run pytest -q` (capture to `outputs/ci/pytest_<date>.txt`).
- [ ] **S0.3** Verify extras install paths: `uv sync --extra dev --extra sdvae --extra imagenet` is reproducible (document command and duration).
- [ ] **S0.4** Record disk requirements and recommendations (latent-only / pixel / 50k-eval) in `docs/imagenet_runbook.md` or a dedicated doc.
- [ ] **S0.5** Ensure every major entrypoint emits `RUN.md` + `env_snapshot.json` + `codebase_fingerprint.json` (audit list and patch gaps).

## S1 — Drift Core + Toy Validation (10%)
- [ ] **S1.1** Add “equilibrium sanity” test: matched positives/negatives ⇒ drift near 0 (unit test).
- [ ] **S1.2** Add stress test sweep for extreme kernel temperatures (finite + bounded outputs; no NaN/Inf).
- [ ] **S1.3** Add precision stability coverage for drift kernel across `fp32/bf16/fp16` (autocast-aware).
- [ ] **S1.4** Create a longer-horizon toy config and store qualitative artifacts (trajectories + plots) under `outputs/toy/`.
- [ ] **S1.5** Deterministic toy replay gate: fixed seed ⇒ identical artifact hashes (document in `tests/integration/`).

## S2 — Generator Architecture Parity (10%)
- [ ] **S2.1** Expand architecture parity table: for each component, tag PAPER-BACKED / INFERRED / OPEN in `docs/generator_arch_parity.md`.
- [ ] **S2.2** Verify generator parameter counts and I/O shapes against paper appendix and record in `docs/generator_arch_parity.md`.
- [ ] **S2.3** Add/confirm deterministic generator overfit gate (tiny subset, stable convergence) with artifact bundle.
- [ ] **S2.4** Run and document a generator ablation matrix (RoPE / QK-norm / RMSNorm / register tokens / style path) with stability + throughput notes.
- [ ] **S2.5** Enforce checkpoint→sampling config parity (no hidden overrides); add an integration test for “sample from checkpoint uses embedded config”.

## S3 — Feature-Space Drifting Stack Parity (10%)
- [ ] **S3.1** Enumerate paper-required feature inventory (locations/scales/normalization) in `docs/feature_loss_parity.md` with explicit code pointers.
- [ ] **S3.2** Close parity delta #1: implement “drift loss without φ” combination exactly as paper describes (or decide to deviate + record impact).
- [ ] **S3.3** Close parity delta #2: implement multi-temperature aggregation semantics as paper describes (or decide to deviate + record impact).
- [ ] **S3.4** Add unit tests pinning the two parity semantics above (fails on old behavior).
- [ ] **S3.5** Add feature extraction shape tests for each feature family and scale (unit tests).
- [ ] **S3.6** Add feature normalization invariance tests (affine scaling behavior; logs `S_j` and `lambda_j` non-degenerate).
- [ ] **S3.7** Add a feature ablation runner producing comparable summaries (JSON + markdown in `outputs/feature_ablations/`).

## S4 — CFG + Grouping + Queueing (15%)
- [ ] **S4.1** Map CFG math to code symbols and logged quantities in `docs/paper_to_code_map.md` (include alpha sampling and negative mixing).
- [ ] **S4.2** Add numeric test: alpha → unconditional weighting consistent for multiple alpha values (unit).
- [ ] **S4.3** Add grouped-batch semantics tests (class-consistent positives; unconditional negatives handled correctly).
- [ ] **S4.4** Add queue determinism test (single-process): fixed seed ⇒ identical warmup report + identical underflow counters.
- [ ] **S4.5** Define DDP policy: either implement DDP-safe queueing or explicitly guard + document single-process limitation.
- [ ] **S4.6** Add a throughput benchmark script for queue hot path (step time, VRAM, CPU RAM) under representative configs.

## S5 — Latent ImageNet Pipeline (20%)
### Provenance and dataset lifecycle (must be explicit)
- [ ] **S5.1** Record checksums (md5 + sha256) for ImageNet archives in `data/` and write `docs/imagenet_dataset_provenance.md`.
- [ ] **S5.2** Record exact extraction command + devkit handling + split integrity checks (train count, val count) in `docs/imagenet_runbook.md`.
- [ ] **S5.3** Record SD-VAE provenance (source repo + commit hash + weights hash) in a dedicated doc and in run artifacts.
- [ ] **S5.4** Record Inception weight provenance for eval (`torchvision` version + weight enum + hash where available) in eval JSON.

### Table-8 readiness (paper mapping)
- [ ] **S5.5** Create exact Table 8 configs (one per column/row entry) with explicit mapping to table labels.
- [ ] **S5.6** Add a “Table 8 readiness” benchmark runner: step time, VRAM, underflow stats, disk churn; output to `outputs/table8_bench/`.
- [ ] **S5.7** Decide scaling strategy (single GPU vs DDP) and record in `docs/decision_log.md` with constraints.

### Paper-scale latent run (or closest feasible, explicitly stated)
- [ ] **S5.8** Launch paper-scale (or closest feasible) latent training with the finalized parity semantics (S3/S4), fixed runbook, and checkpoint cadence.
- [ ] **S5.9** Generate **50k** decoded samples (256x256) with fixed seed policy; archive sample manifest.
- [ ] **S5.10** Evaluate FID/IS with pretrained Inception + cached reference stats; archive reference stats and eval JSON.
- [ ] **S5.11** Run nearest-neighbor audit on generated samples and archive report JSON/markdown.
- [ ] **S5.12** Update `docs/reproduction_report.md` with artifact-backed summary and explicit parity deltas (if any remain).

## S6 — MAE Encoder Reproduction (10%)
- [ ] **S6.1** Pin MAE pretraining hyperparameters to paper appendix (batch, lr, wd, aug, EMA, schedule) in a config + doc.
- [ ] **S6.2** Verify augmentation placement (pre/post VAE encode) matches paper intent; document and test.
- [ ] **S6.3** Train MAE variants at meaningful horizons (widths/epochs) and export encoders with embedded config metadata.
- [ ] **S6.4** Implement and validate classifier fine-tuning stage (“cls ft”) per paper; export fine-tuned encoder.
- [ ] **S6.5** Quantify downstream impact of MAE variants under a controlled latent drifting protocol; produce ranked summary.
- [ ] **S6.6** Decide default encoder variant for scale runs (decision log + evidence).

## S7 — Pixel-Space Pipeline (10%)
- [ ] **S7.1** Choose and document pixel data format for ImageNet scale (ImageFolder vs WebDataset) with disk/IO tradeoffs.
- [ ] **S7.2** Implement pixel dataset lifecycle tooling (extract/shard/cache/cleanup) with reproducible commands.
- [ ] **S7.3** Implement paper-described pixel feature encoder path(s) and parity docs (MAE/ConvNeXt-V2/etc if required).
- [ ] **S7.4** Validate pixel encoder grad flow + multi-temperature semantics (unit + integration tests).
- [ ] **S7.5** Run pixel ablation-scale training with checkpoints, sampling, alpha sweep, and last-K eval.
- [ ] **S7.6** Run pixel paper-scale (or closest feasible) training + **50k** sample eval; archive all artifacts and update report.

## S8 — Scale, Harden, and Final Report (10%)
### Close open ambiguities (required for “100%”)
- [ ] **S8.1** Enumerate remaining OPEN decisions (if any) and close or explicitly defer with impact (decision log entries).
- [ ] **S8.2** Add regression tests for any scaling-time footguns discovered (resume semantics, config override traps, eval determinism).

### Reproducibility package (clean-room)
- [ ] **S8.3** Write a clean-room reproduction doc: env → data → train → sample → eval (single canonical path, minimal branching).
- [ ] **S8.4** Build a claim-to-evidence table mapping paper figures/tables to exact artifacts and commands.
- [ ] **S8.5** Build a deviations table with impact assessment (paper vs repo).
- [ ] **S8.6** Add orchestrators for paper-scale latent and pixel runs (resume-friendly, idempotent, writes `RUN.md`).

### Disk hygiene / archival
- [ ] **S8.7** Disk cleanup pass: remove redundant caches/samples/checkpoints; keep a curated “golden artifacts” index.
- [ ] **S8.8** Add a “what can be safely deleted” section in runbooks (with size estimates).

### Final 100% gate
- [ ] **S8.9** Mark all stages 100% in `docs/reproduction_report.md` with explicit artifact references.
- [ ] **S8.10** Run a clean-room reproduction drill on a fresh machine/user account and record outcome + timings.

