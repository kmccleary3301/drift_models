# Paper Table 8 → Repo Config Map

Paper scan reference: `Drift_Models/Drift_Models.md` around “Table 8. Configurations for ImageNet 256×256”.

This document maps each **paper Table 8 column** to a concrete repo config file and the script that consumes it.

## Canonical entrypoint

Latent training uses:

```bash
uv run python scripts/train_latent.py --config <CONFIG_PATH> --output-dir <OUT_DIR> --checkpoint-dir <CKPT_DIR>
```

Notes:
- `scripts/train_latent.py --config` expects a simple `key: value` file (YAML-like, no nested objects).
- Generator optimizer parity is controlled via `learning-rate`, `adam-beta1`, `adam-beta2`, and `weight-decay`.
- Generator FFN sizing for SwiGLU is controlled via `ffn-inner-dim` (see `docs/generator_arch_parity.md` and DEC-0034).

## Map (latent columns)

| Paper Table 8 column | Repo config | Tier | Status |
| --- | --- | --- | --- |
| ablation default | `configs/latent/imagenet1k_sdvae_latents_table8_ablation_default_template.yaml` | faithful-template | Template pinned to paper-facing loss/queue fields; requires `w256` MAE export; scale run pending |
| B/2, latent (Table 5) | `configs/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml` | faithful-template | Template pinned to paper-facing loss/queue fields; requires `w640` MAE export; scale run pending |
| L/2, latent (Table 5) | `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml` | faithful-template | Template pinned to paper-facing loss/queue fields; requires `w640` MAE export; scale run pending |

### Key parity notes for latent templates
- **AdamW betas**: paper uses `(β1=0.9, β2=0.95)`; configs set `adam-beta2: 0.95`.
- **Weight decay**: Table 8 varies by column; configs set `weight-decay` explicitly.
- **CFG α distribution**:
  - ablation default: `alpha-dist: powerlaw`, `alpha-power: 3.0`
  - B/2 latent: `alpha-dist: powerlaw`, `alpha-power: 5.0`
  - L/2 latent: `alpha-dist: table8_l2_latent` (repo’s explicit mixture implementation)
- **Model size / params**: paper reports ~`133M` (B/2) and ~`463M` (L/2) generator params (excluding SD‑VAE); configs set `ffn-inner-dim` to match those counts tightly.
- **Feature multi-temperature aggregation**: Table-8 latent templates pin `feature-temperature-aggregation: sum_drifts_then_mse` to match Appendix A.6 (sum normalized drifts over temperatures, then apply one stopgrad target loss).
- **Feature loss term reduction**: Table-8 latent templates pin `feature-loss-term-reduction: sum` so feature terms are summed rather than averaged across keys/slots.
- **Feature inventory completeness**: Table-8 latent templates pin `include-input-x2-mean: true` and `include-patch4-stats: true`.
- **Raw + feature loss composition**: Table-8 latent templates pin `feature-include-raw-drift-loss: true` and `feature-raw-drift-loss-weight: 1.0`.
- **Feature encoder architecture parity**: Table-8 latent templates pin `mae-encoder-arch: paper_resnet34_unet` for paper-faithful feature extractor semantics.
- **Feature encoder width parity**: Table-8 latent templates pin `feature-base-channels` to Table-8 values (`256` for ablation default, `640` for B/2 and L/2) and expect matching MAE export paths (`w256` / `w640`).
- **Width-parity export orchestration**: bootstrap width-matched MAE export configs live in `configs/mae/imagenet1k_sdvae_latents_shards_table8_w256_bootstrap.yaml` and `configs/mae/imagenet1k_sdvae_latents_shards_table8_w640_bootstrap.yaml`; queued execution/provenance runner is `scripts/run_mae_width_parity_exports.py`.
- **Queue defaults for paper-facing templates**: table-facing latent templates pin `queue-global-capacity: 1000` and `queue-push-batch: 64` to match Appendix queue semantics.
- **Queue no-replacement guard**: Table-8 latent templates pin `queue-strict-without-replacement: true` so per-class positives and unconditional negatives fail fast if queues cannot satisfy no-replacement sampling.
- **Contract guardrail**: `tests/unit/test_table8_faithful_config_contract.py` enforces these paper-facing latent template fields in CI.

### Closest-feasible template caveat
- `configs/latent/imagenet1k_sdvae_latents_table8_b2_closest_feasible_single_gpu.yaml` is intentionally not paper-identical.
- It preserves structural ratios and faithful semantics where possible, but uses reduced grouped-batch sizes and larger queue capacities for single-GPU practicality.
- It keeps two explicit temporary exceptions with rationale in-file: `mae-encoder-arch: resnet_unet` (legacy MAE export compatibility) and `queue-strict-without-replacement: false` (single-GPU churn stability).
- Active long-horizon restart configs generated from this template (for example `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/config.yaml`) inherit the same `closest-feasible` tier and must not be promoted as faithful-template evidence.
- Tier label for this config is `closest-feasible` (not `faithful-template`).

## Pixel columns (deferred to Stage 7)

Paper Table 8 also specifies pixel-space columns (B/16, L/16) including encoder details (ResNet + ConvNeXt‑V2, pixel‑MAE) and a pixel generator.

**Status**: pixel Table‑8 configs are not declared “exact” yet in this repo because:
- the pixel feature-encoder inventory in Table 8 is not fully implemented/pinned here (a `convnext_tiny` feature path exists, but the paper-specific ConvNeXt‑V2/pixel-MAE stack is not fully reproduced),
- scheduler controls (`scheduler`, `warmup-steps`) are now exposed in `scripts/train_pixel.py`, but paper-specific pixel schedule values and scale behavior are not yet validated with claim-facing evidence.

These are tracked under Stage 7 and will be filled in once the pixel pipeline is promoted to paper-facing parity.
See also: `docs/pixel_scope_status.md` and `DEC-0039` in `docs/decision_log.md`.
