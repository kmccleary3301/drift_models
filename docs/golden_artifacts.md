# Golden Artifacts Index (What To Keep)

This is a curated index of the **highest-value** artifacts to retain when cleaning disk. Everything else is re-creatable.

## Dataset provenance
- ImageNet archives checksums: `outputs/datasets/imagenet1k_provenance.json`
- ImageNet reference-stats contract: `outputs/datasets/imagenet1k_reference_stats_contract.json`
- SD‑VAE weights provenance: `outputs/datasets/sdvae_provenance.json`
- ImageNet val reference stats (pretrained Inception): `outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`

## ImageNet latent (recommended checkpoints)
- Step-400 best checkpoint (per memo): `docs/imagenet_step400_best_checkpoint_recommendation.md`
- Current default step-400 snapshot: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt`
- Step-600 continuation snapshots (for last‑K): `outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/`

## Decision + parity documents
- Decisions: `docs/decision_log.md`
- Decision closure status: `docs/decision_closure_status.md`
- Pixel scope status: `docs/pixel_scope_status.md`
- Paper→code mapping: `docs/paper_to_code_map.md`
- Claim→evidence matrix: `docs/claim_to_evidence_matrix.md`
- Deviations table: `docs/deviations_table.md`
- Feature parity deltas: `docs/feature_loss_parity.md`
- Generator parity audit: `docs/generator_parity_audit.md`

## Cleanup tooling + reports
- Cleanup script: `scripts/disk_cleanup_checkpoints.py`
- Cleanup reports (example): `outputs/imagenet/cleanup_2026-02-14_recovery_matrix_r3_applied.md`

## Gate artifacts (2026-02-20 tranche)
- Provenance bundle snapshot: `outputs/ops/provenance_bundle_20260220_001234.json`
- Claim eval contract audit: `outputs/ops/claim_eval_contract_audit_20260220/claim_eval_contract_audit.json`
- Run bundle audit: `outputs/ops/run_bundle_audit_20260220/run_bundle_audit.json`
- Faithful toggle audit: `outputs/ops/faithful_toggle_audit_20260220/faithful_toggle_audit.json`
- Parity validation archive: `outputs/ops/parity_validation_archive_20260220.md`
- Clean-room drill traceability:
  - `outputs/ops/cleanroom_drill_20260220_latent_smoke_manifest.json`
  - `outputs/ops/cleanroom_drill_20260220_notes.md`

## Retention policy
- Keep all files listed above until final Gate-9 closure.
- During disk pressure, prune first:
  - large decoded sample image folders not referenced by claim docs,
  - redundant mutable checkpoints when immutable step checkpoints exist,
  - temporary debug/probe outputs not referenced by `docs/experiment_log.md`.
