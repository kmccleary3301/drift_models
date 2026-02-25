# Evaluation Contract (FID/IS)

This repo treats FID/IS comparability as a strict contract, not a loose convention.

## Contract fields

The canonical contract payload is:

- `inception_model`: `torchvision.inception_v3`
- `inception_weights`: `pretrained` or `none`
- `preprocess.resize_hw`: `[299, 299]`
- `preprocess.normalize.mean`: `[0.485, 0.456, 0.406]`
- `preprocess.normalize.std`: `[0.229, 0.224, 0.225]`
- `input_normalization_policy`: `auto_range_to_0_1_then_imagenet_normalize`

## Input normalization policy

Before Inception preprocessing:

- If input range is already `[0,1]`: keep as-is.
- If input range is `[-1,1]`: map via `(x + 1) / 2`.
- If input range appears `[0,255]`: divide by `255` and clamp.
- Otherwise: clamp to `[0,1]`.

## Provenance and hashing

- `scripts/cache_reference_stats.py` stores:
  - `mean`, `cov`, `count`
  - `contract` and `contract_sha256`
  - dataset provenance (`source`, root/path metadata, sample cap, extension filter)
- `scripts/eval_fid_is.py` includes in summary:
  - active `protocol.contract_sha256`
  - loaded stats provenance (`loaded_contract_sha256`, `loaded_contract`, source metadata)
  - loaded stats file SHA256 when stats are loaded from disk

## Enforcement behavior

- By default, `scripts/eval_fid_is.py --load-reference-stats ...` enforces contract match.
- Override only intentionally with:
  - `--allow-reference-contract-mismatch`

This should be used only for controlled back-compat checks, never for paper-facing claims.

## Claim-facing audit helper

For claim-facing report gates, run:
- `scripts/audit_claim_eval_contract.py`

and require all audited entries to pass the pretrained/standard/reference-stats checks.
