# Reproducibility scoreboard

This table is the canonical place to track quality, time, and estimated cost for claim-facing runs.

## Contract

- Use pretrained Inception (`--inception-weights pretrained`) for claim-facing FID/IS.
- Record exact artifact paths for every row.
- Record estimated GPU-hours from `latent_summary.json` (`steps * mean_step_time_s / 3600`).
- Record cost using your effective rate:
  - `estimated_cost = estimated_gpu_hours * ($ / GPU-hour)`

## Current rows

| Run | Lane | Config | Steps | Eval protocol | FID | IS | Estimated GPU-hours | Estimated cost @ USD3/h | Evidence |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| `stable_20260305_020202` | Stable smoke wrapper | `configs/stable/latent_smoke_feature_queue.yaml` | 2 | artifact validation only (`validate_run_artifacts`), no FID/IS | N/A | N/A | 0.00010 | USD0.00031 | `outputs/imagenet/stable_20260305_020202/stable_lane_summary.json`, `outputs/imagenet/stable_20260305_020202/artifact_validation.json`, `outputs/imagenet/stable_20260305_020202/latent_summary.json` |
| `latent_smoke_mae` | Stable smoke | `configs/latent/imagenet1k_sdvae_latents_queue_smoke_mae.yaml` | 60 | pretrained Inception, 5k generated vs 50k val | 429.31 | 1.4032 | 0.11 | USD0.32 | `outputs/imagenet/latent_smoke_mae/latent_summary.json`, `outputs/imagenet/latent_smoke_mae_eval_2026-02-10_183641/eval_pretrained.json` |
| `paperscale_b2_corrected_restart_nokernelcompile_20260219_152045` | Closest-feasible B/2 long run | `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/config.yaml` | 200000 | pretrained Inception, 50k generated vs 50k val | 396.63 | 1.8252 | 110.24 | USD330.73 | `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/latent_summary.json`, `outputs/imagenet/paperscale_b2_corrected_restart_nokernelcompile_20260219_152045/claim_bundle/claim_eval/eval_pretrained.json` |

## Interpretation

- This repo currently provides mechanical and pipeline faithfulness evidence.
- It does not currently provide paper-level FID parity evidence.
- If you add a new run, append a new row instead of editing historical rows.

## Row template

Copy this template when logging a new run:

```md
| `<run_id>` | `<lane>` | `<config_path>` | `<steps>` | `<eval protocol>` | `<fid>` | `<is>` | `<gpu_hours>` | `<cost>` | `<artifact paths>` |
```
