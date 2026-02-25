# ImageNet Step-400 Post-Run Summary

Date: 2026-02-11  
Primary run root: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64`

## Completion state
- Final checkpoint: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt`
- Final train summary: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/latent_summary.json`
- Device lane: `cuda:1`

## Evaluation bundle
- Last-K summary (`k=3`, `n=2000`): `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_lastk_eval_k3_s2k/last_k_summary.json`
  - step 300: FID `318.1318`, IS `1.1796`
  - step 350: FID `342.0843`, IS `1.2085`
  - step 400: FID `293.5726`, IS `1.4601`
- Alpha sweep summary (`alphas=1.0,1.5,2.0,2.5,3.0`, `n=2000`): `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_alpha_sweep_s2k/alpha_sweep_summary.json`
  - alpha 1.0: FID `293.5726`, IS `1.4601`
  - alpha 1.5: FID `293.5298`, IS `1.4607`
  - alpha 2.0: FID `293.5328`, IS `1.4616`
  - alpha 2.5: FID `293.5845`, IS `1.4625`
  - alpha 3.0: FID `293.6162`, IS `1.4631`

## Qualitative + audit artifacts
- Fixed-seed trajectory grids (checkpoints 300/350/400):  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_fixed_seed_grids_postcleanup/fixed_seed_grid_summary.json`
- Nearest-neighbor audit (alpha 1.5 branch):  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_alpha_sweep_s2k/alpha_1p5/nn_audit.json`
  - mean cosine: `0.3413`
  - median cosine: `0.3399`
  - p95 cosine: `0.3919`
  - label-match rate: `0.001953125`

## Cleanup action (conservative)
- Cleanup report: `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_cleanup_20260211.md`
- Removed: redundant checkpoints `150/200/250` and stale relaunch artifacts.
- Run directory size: `~17G -> ~9.4G`.
