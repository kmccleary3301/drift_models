# ImageNet Step-550 Recovery Comparison (2026-02-13)

## Scope
Both branches continue from the same source checkpoint:
- `outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/checkpoint_step_00000550.pt`

Common protocol:
- terminal step: `700`
- eval: alpha sweep (`2k` samples/alpha, pretrained Inception, cached ImageNet val stats)
- NN audit: `alpha=1.5`, `max_generated=512`, `max_reference=10000`

## Branch A: Optimizer-restored resume
- train summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64/latent_summary.json`
- sweep summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_alpha_sweep_s2k_rerun/alpha_sweep_summary.json`
- nn audit: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_alpha_sweep_s2k_rerun/alpha_1p5/nn_audit.json`
- terminal lr: `2.0e-4` (inherited)
- best FID in sweep: `~354.08` (`alpha=1.0`)
- IS near best FID: `~1.00838`
- NN mean cosine: `~0.3274`

## Branch B: Optimizer-reset (`--resume-model-only`)
- train summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly/latent_summary.json`
- sweep summary: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly_alpha_sweep_s2k/alpha_sweep_summary.json`
- nn audit: `outputs/imagenet/latent_recovery_from550_step700_cuda1_w64_modelonly_alpha_sweep_s2k/alpha_1p5/nn_audit.json`
- terminal lr: `8.0e-5` (from config)
- best FID in sweep: `~332.12` (`alpha=2.5`)
- IS near best FID: `~1.02644`
- NN mean cosine: `~0.1914`

## Delta (B vs A)
- FID improvement: `~21.96` lower
- IS improvement: `~+0.018`
- NN mean cosine reduction: `~0.136`

## Interpretation
At this horizon and protocol, optimizer-state carryover is a high-impact degradation factor. The optimizer-reset branch is materially better and should be the default diagnostic baseline for subsequent recovery experiments.
