# Pixel Scope Status

Current pixel-path status for claim governance.

## Current scope
- Pixel pipeline is **experimental / closest-feasible**.
- Pixel paper-faithful claims are **deferred** pending closure of explicit implementation/evidence gates.

## Deferred closure gates
- Produce paper-scale (not ablation-scale) pixel claim package under pretrained-cached eval contract.
- Promote pixel artifacts from closest-feasible to paper-faithful only after claim-grade evidence and gate audit.

## Recently closed sub-gaps
- Pixel trainer now exposes scheduler controls required for paper-facing sweeps (`scheduler`, `warmup-steps`).
- Pixel checkpoints now persist scheduler state; smoke/integration coverage added for scheduler path.
- Pixel trainer now supports a ConvNeXt-family feature extractor path (`feature-encoder=convnext_tiny`) for multi-scale feature taps.
- Pixel trainer now also supports a ConvNeXt-V2 family feature extractor path (`feature-encoder=convnextv2_tiny`) for paper-closer encoder ablations.
- Pixel trainer now supports a dual paper-closer feature path (`feature-encoder=mae_convnextv2`) combining ResNet-style MAE taps with ConvNeXt-V2 taps.
- Pixel MAE now supports paper-specific pixel patchify semantics (`input 256x256x3 -> patchify 8x8 -> 32x32x192`) with decode-time unpatchify and regression tests.
- Pixel Table-8 faithful templates are now pinned:
  - `configs/experimental/pixel/imagenet256_table8_b16_template.yaml`
  - `configs/experimental/pixel/imagenet256_table8_l16_template.yaml`
  - compatibility wrappers remain at `configs/pixel/imagenet256_table8_*_template.yaml`.
  - guarded by `tests/unit/test_table8_pixel_faithful_config_contract.py`.
- Pixel feature-encoder ablation runner is available for quick evidence packaging (`scripts/experimental/ablations/pixel_feature_encoder.py`; compatibility wrapper: `scripts/run_pixel_feature_encoder_ablations.py`).
- Pixel pretrained-cached small-sample ablation package has been executed for feature-encoder variants (`outputs/feature_ablations/pixel_pretrained_cached_ablation_20260220_v2`).
- Pixel paper-facing package orchestrator is available (`scripts/experimental/pipelines/pixel_paper_facing_package.py`; compatibility wrapper: `scripts/run_pixel_paper_facing_package.py`) with smoke artifact at `outputs/feature_ablations/pixel_paper_facing_package_smoke_20260220`.
- Pixel paper-facing package orchestrator has been validated in both profiles:
  - proxy smoke: `outputs/feature_ablations/pixel_paper_facing_package_smoke_20260220`
  - pretrained-cached smoke: `outputs/feature_ablations/pixel_paper_facing_package_pretrained_smoke_20260220`
- Pixel pretrained-cached pre-production package run is complete:
  - `outputs/feature_ablations/pixel_paper_facing_package_preprod_convnext_20260220/paper_facing_package_summary.json`
  - `outputs/feature_ablations/pixel_paper_facing_package_preprod_convnext_20260220/claim/nn_audit.json`
- Ablation-scale pretrained-cached package run is complete:
  - `outputs/feature_ablations/pixel_pretrained_cached_ablation_20260220_scale1/pixel_proxy_ablation_summary.json`
  - best FID in this package: `tiny` (`~370.49`), then `mae` (`~395.99`), then `convnext_tiny` (`~418.05`).
  - eval-contract audits pass for all three variant evals:
    - `outputs/ops/claim_eval_contract_audit_20260221/pixel_scale1_tiny_eval_contract_audit.json`
    - `outputs/ops/claim_eval_contract_audit_20260221/pixel_scale1_mae_eval_contract_audit.json`
    - `outputs/ops/claim_eval_contract_audit_20260221/pixel_scale1_convnext_tiny_eval_contract_audit.json`
- Larger paper-facing pretrained-cached package run is complete (`scale2`, `tiny` checkpoint):
  - `outputs/feature_ablations/pixel_paper_facing_package_scale2_tiny_20260221/paper_facing_package_summary.json`
  - claim package (`n=4096`, `alpha=1.5`): `fid≈360.61`, `IS≈1.2789`
  - alpha sweep (`n=512` per alpha): `fid≈370.49..370.52` for `alpha={1.0,1.5,2.0}`
  - NN audit (`512x10000`): mean cosine `≈0.1804`, label-match rate `≈0.0059`
  - claim eval contract audit:
    - `outputs/ops/claim_eval_contract_audit_20260221/pixel_paper_pkg_scale2_tiny_claim_eval_contract_audit.json`
- Larger paper-facing pretrained-cached package run is complete (`scale3`, `tiny` checkpoint):
  - `outputs/feature_ablations/pixel_paper_facing_package_scale3_tiny_20260221/paper_facing_package_summary.json`
  - claim package (`n=8192`, `alpha=1.5`): `fid≈360.30`, `IS≈1.2742`
  - alpha sweep (`n=1024` per alpha): `fid≈365.35..365.38` for `alpha={1.0,1.5,2.0}`
  - NN audit (`1024x15000`): mean cosine `≈0.2282`, label-match rate `≈0.0000`
  - claim eval contract audit:
    - `outputs/ops/claim_eval_contract_audit_20260221/pixel_paper_pkg_scale3_tiny_claim_eval_contract_audit.json`
- Pixel orchestration scripts now have explicit regression coverage:
  - `tests/unit/test_pixel_orchestration_scripts.py` validates package wiring/helpers for
    `run_pixel_paper_facing_package.py`, `run_pixel_proxy_ablation_package.py`, and
    `run_pixel_feature_encoder_ablations.py` (canonical ablation paths under `scripts/experimental/ablations/`).
  - coverage now also pins eval-profile command contracts for paper-facing packaging
    (`proxy` vs `pretrained_cached` reference/eval flags).
  - variant-list coverage now pins both ConvNeXt variants (`convnext_tiny`, `convnextv2_tiny`) in ablation package wiring.

## Claim policy
- Do not promote pixel artifacts to paper-faithful claim set until all deferred gates are closed.
- Keep pixel claims labeled as experimental in report and release docs.
