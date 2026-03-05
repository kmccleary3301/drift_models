# Claim → Evidence Matrix

This matrix maps current repo claims to concrete artifacts and test evidence.

## Legend
- `closed`: implementation + tests + artifacts present.
- `gated`: partially complete; final claim blocked on pending artifacts/runs.

| Claim | Status | Evidence |
| --- | --- | --- |
| Drift-field core semantics (attraction/repulsion + stop-grad objective) are implemented and tested | closed | `drifting_models/drift_field.py`, `drifting_models/drift_loss.py`, `tests/unit/test_drift_field.py`, `tests/unit/test_drift_loss.py` |
| Feature-space reduction/normalization semantics are explicit and regression-protected | closed | `tests/unit/test_feature_drift_loss.py`, `docs/decision_log.md` (DEC-0036) |
| Generator micro-variant parity controls are implemented and regression-tested | closed | `drifting_models/models/dit_like.py`, `scripts/train_latent.py`, `scripts/train_pixel.py`, `scripts/sample_latent.py`, `scripts/sample_pixel.py`, `tests/unit/test_dit_like.py` |
| Faithful Table-8 latent template contracts are pinned and test-enforced | closed | `configs/experimental/latent/imagenet1k_sdvae_latents_table8_*_template.yaml` (legacy wrappers remain under `configs/latent/*`), `tests/unit/test_table8_faithful_config_contract.py`, `docs/decision_log.md` (DEC-0040 width parity + MAE export-path expectations), `docs/generator_microvariant_selection_20260303.md` (micro-variant tuple pinning) |
| Paper MAE architecture/tap semantics are implemented | closed | `drifting_models/features/mae.py`, `tests/unit/test_paper_mae_arch.py` |
| Queue no-replacement semantics can be enforced for paper-facing runs | closed | `drifting_models/data/queue.py`, `tests/unit/test_data_queue.py`, `tests/integration/test_queue_strict_mode.py` |
| Dataset/tokenizer/evaluator provenance is explicitly captured | closed | `docs/provenance_contract.md`, `tests/unit/test_provenance_contract.py`, `outputs/ops/provenance_bundle_20260220_001234.json` |
| Claim-facing metrics use pretrained-Inception comparability contract | closed | `scripts/audit_claim_eval_contract.py`, `outputs/ops/claim_eval_contract_audit_20260220/claim_eval_contract_audit.md`, `docs/eval_contract.md` |
| Pixel encoder ablation wiring covers ConvNeXt and ConvNeXt-V2 families | closed | `scripts/train_pixel.py`, `scripts/experimental/ablations/pixel_feature_encoder.py` (wrapper: `scripts/run_pixel_feature_encoder_ablations.py`), `scripts/experimental/ablations/pixel_proxy_package.py` (wrapper: `scripts/run_pixel_proxy_ablation_package.py`), `tests/unit/test_pixel_orchestration_scripts.py` |
| Pixel Table-8 mechanical config/feature semantics are pinned | closed | `configs/experimental/pixel/imagenet256_table8_*_template.yaml` (legacy wrappers remain under `configs/pixel/*`), `drifting_models/features/mae.py` (pixel patchify/unpatchify semantics), `scripts/train_pixel.py` (`mae_convnextv2`), `tests/unit/test_table8_pixel_faithful_config_contract.py`, `tests/unit/test_paper_mae_arch.py` |
| Long-horizon latent faithful evidence package is complete | gated | Pending `G3.1..G3.6` outputs; active lane: `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045` |
| Pixel paper-faithful implementation/evidence package is complete | gated | Mechanical parity semantics are now pinned, and pre-production + scale2/scale3 paper-facing packages are complete (`outputs/feature_ablations/pixel_paper_facing_package_preprod_convnext_20260220/paper_facing_package_summary.json`, `outputs/feature_ablations/pixel_paper_facing_package_scale2_tiny_20260221/paper_facing_package_summary.json`, `outputs/feature_ablations/pixel_paper_facing_package_scale3_tiny_20260221/paper_facing_package_summary.json`) with eval-contract audits, but paper-scale claim package + final evidence gates remain pending (`G4.*`). |
| End-to-end faithful clean-room reproduction is complete | gated | Smoke drill complete (`outputs/ops/cleanroom_drill_20260220_latent_smoke`), faithful E2E pending (`G8.2`) |
