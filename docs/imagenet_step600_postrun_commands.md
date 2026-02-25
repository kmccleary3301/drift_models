# ImageNet Step-600 Post-Run Commands (cuda:1)

Use this bundle after `outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoint.pt` is written.

## 1) Last-K checkpoint evaluation (K=4, 2k samples/checkpoint)

```bash
uv run python scripts/eval_last_k_checkpoints.py \
  --output-root outputs/imagenet/latent_ablation_b2_600_cuda1_w64_lastk_eval_k4_s2k \
  --mode latent \
  --checkpoint-dir outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints \
  --k 4 \
  --config configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step600_w64.yaml \
  --device cuda:1 \
  --n-samples 2000 \
  --batch-size 32 \
  --decode-mode sd_vae \
  --decode-image-size 256 \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --sd-vae-revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --inception-weights pretrained \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --overwrite
```

## 2) Alpha sweep at step-600 checkpoint (2k samples/alpha)

```bash
uv run python scripts/eval_alpha_sweep.py \
  --output-root outputs/imagenet/latent_ablation_b2_600_cuda1_w64_alpha_sweep_s2k \
  --mode latent \
  --checkpoint-path outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoint.pt \
  --config configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step600_w64.yaml \
  --device cuda:1 \
  --alphas 1.0 1.5 2.0 2.5 3.0 \
  --n-samples 2000 \
  --batch-size 32 \
  --decode-mode sd_vae \
  --decode-image-size 256 \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --sd-vae-revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --inception-weights pretrained \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --overwrite
```

## 3) Fixed-seed grids across final checkpoints

```bash
uv run python scripts/sample_fixed_seed_grids.py \
  --checkpoint-dir outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints \
  --output-root outputs/imagenet/latent_ablation_b2_600_cuda1_w64_fixed_seed_grids \
  --config configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step600_w64.yaml \
  --device cuda:1 \
  --alpha 1.5 \
  --grid-samples 16 \
  --batch-size 16 \
  --decode-mode sd_vae \
  --decode-image-size 256 \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --sd-vae-revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --max-checkpoints 4
```

## 4) Nearest-neighbor audit on best-alpha branch

```bash
uv run python scripts/audit_nearest_neighbors.py \
  --generated-root outputs/imagenet/latent_ablation_b2_600_cuda1_w64_alpha_sweep_s2k/alpha_1p5/samples/images \
  --reference-root outputs/datasets/imagenet1k_raw/val \
  --device cuda:1 \
  --max-generated 512 \
  --max-reference 10000 \
  --output-path outputs/imagenet/latent_ablation_b2_600_cuda1_w64_alpha_sweep_s2k/alpha_1p5/nn_audit.json
```
