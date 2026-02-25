# ImageNet Step-400 Best Checkpoint Recommendation

Date: 2026-02-11  
Scope: select the best available checkpoint/alpha pair from the completed continuation run.

## Evidence scope
- Checkpoints compared (same protocol):  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000300.pt`  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000350.pt`  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt`
- Last-K summary source:  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_lastk_eval_k3_s2k/last_k_summary.json`
- Alpha sweep source (step 400):  
  `outputs/imagenet/latent_ablation_b2_400_cuda1_w64_alpha_sweep_s2k/alpha_sweep_summary.json`

## Selection objective
- Primary objective: minimize FID under fixed protocol.
- Secondary objective: maintain/improve IS and stability around the chosen alpha.

## Results snapshot
- Last-K (`n=2000`, alpha=1.0):
  - step 300: FID `318.1318`, IS `1.1796`
  - step 350: FID `342.0843`, IS `1.2085`
  - step 400: FID `293.5726`, IS `1.4601` (best checkpoint by FID and IS)
- Alpha sweep at step 400 (`n=2000`):
  - alpha 1.0: FID `293.5726`, IS `1.4601`
  - alpha 1.5: FID `293.5298`, IS `1.4607` (best FID)
  - alpha 2.0: FID `293.5328`, IS `1.4616`
  - alpha 2.5: FID `293.5845`, IS `1.4625`
  - alpha 3.0: FID `293.6162`, IS `1.4631` (best IS, slightly worse FID)

## Recommendation
- **Default checkpoint**: `checkpoint_step_00000400.pt`
- **Default sampling alpha**: `1.5`
- Rationale: this pair gives the lowest observed FID in the step-400 sweep while staying on the same IS plateau as neighboring alphas.

## Alternate operating point
- If prioritizing IS over FID, use `alpha=3.0` on the same checkpoint.
- Tradeoff: marginally higher IS with slightly degraded FID.

## Canonical command pair (default)
```bash
uv run python scripts/sample_latent.py \
  --device cuda:1 \
  --seed 1337 \
  --checkpoint-path outputs/imagenet/latent_ablation_b2_400_cuda1_w64/checkpoints/checkpoint_step_00000400.pt \
  --config configs/latent/imagenet1k_sdvae_latents_ablation_horizon_b2_cuda1_step400_w64.yaml \
  --output-root outputs/imagenet/step400_best_alpha1p5_samples \
  --n-samples 5000 \
  --batch-size 32 \
  --alpha 1.5 \
  --write-imagefolder \
  --decode-mode sd_vae \
  --sd-vae-model-id stabilityai/sd-vae-ft-mse \
  --sd-vae-revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --decode-image-size 256 \
  --postprocess-mode clamp_0_1
```

```bash
uv run python scripts/eval_fid_is.py \
  --device cuda:1 \
  --batch-size 32 \
  --inception-weights pretrained \
  --reference-source imagefolder \
  --reference-imagefolder-root outputs/datasets/imagenet1k_raw/val \
  --generated-source imagefolder \
  --generated-imagefolder-root outputs/imagenet/step400_best_alpha1p5_samples/images \
  --load-reference-stats outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt \
  --output-path outputs/imagenet/step400_best_alpha1p5_eval/eval_pretrained.json
```

## Command-pair execution (validated)
- Executed on 2026-02-11 with the exact canonical pair above (device `cuda:1`).
- Output artifacts:
  - `outputs/imagenet/step400_best_alpha1p5_samples/sample_summary.json`
  - `outputs/imagenet/step400_best_alpha1p5_eval/eval_pretrained.json`
- Observed metrics (`generated_samples=5000`):
  - FID: `289.5544`
  - IS: `1.4762`

## Alternate operating-point execution (validated)
- Executed on 2026-02-11 with the same checkpoint and protocol, but `alpha=3.0`.
- Output artifacts:
  - `outputs/imagenet/step400_alt_alpha3p0_samples/sample_summary.json`
  - `outputs/imagenet/step400_alt_alpha3p0_eval/eval_pretrained.json`
- Observed metrics (`generated_samples=5000`):
  - FID: `289.6435`
  - IS: `1.4787`
- Direct delta (`alpha=3.0 - alpha=1.5`):
  - `ΔFID = +0.0891` (worse for FID)
  - `ΔIS = +0.0025` (better for IS)
- Decision: keep `alpha=1.5` as default when FID is primary; use `alpha=3.0` for IS-prioritized reporting.
