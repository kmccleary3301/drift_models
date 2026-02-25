# Stage2/Stage4 Execution Snapshot (2026-02-12)

## Runtime / disk status
- `/mnt/drive_4` free space: `368G` (`90%` used at capture time)
- workspace heavy roots:
  - `data`: `145G`
  - `outputs`: `77G`
- largest `outputs/` subtrees:
  - `outputs/imagenet`: `46G`
  - `outputs/datasets`: `24G`
  - `outputs/milestone80`: `3.3G`
  - `outputs/milestone70`: `1.7G`
  - `outputs/stage2_generator_stability_120`: `1.2G`

## Newly completed artifacts
- Stage-2 stability run:
  - `outputs/stage2_generator_stability_120/latent_summary.json`
  - `outputs/stage2_generator_stability_120/checkpoints/checkpoint_step_00000120.pt`
- Stage-2 architecture ablations:
  - `outputs/stage2_generator_arch_ablations_cuda1/generator_arch_ablation_summary.json`
  - `outputs/stage2_generator_arch_ablations_cuda1/generator_arch_ablation_summary.md`
- Stage-2 tiny overfit sanity:
  - `outputs/stage2_tiny_overfit_cuda1/latent_summary.json`
  - `outputs/stage2_tiny_overfit_cuda1/overfit_check.json`
- Stage-3 feature ablation runner check:
  - `outputs/stage3_feature_ablation_check_cuda1/latent_ablation_summary.json`
  - `outputs/stage3_feature_ablation_check_cuda1/latent_ablation_summary.md`
- Stage-4 queue determinism check:
  - `outputs/stage4_queue_determinism/queue_determinism.json`

## Notes
- No active `train_latent.py` or `run_generator_arch_ablations.py` processes remained at snapshot time.
- Queue determinism check passed with identical warmup reports and identical dataset manifest fingerprints.
