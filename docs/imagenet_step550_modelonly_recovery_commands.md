# ImageNet Step-550 Recovery Continuation (Model-Only Resume)

This variant is the **optimizer-reset diagnostic ablation**: resume model weights/step from `step=550` but do not restore optimizer/scheduler/rng state.

It is not the current default recovery policy; see:
- `docs/imagenet_recovery_matrix_2x2_20260213_restart_r3_results.md`

## 1) Train continuation (step 550 -> 700, model-only resume)

```bash
uv run python scripts/train_latent.py \
  --config configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml \
  --device cuda:0 \
  --steps 700 \
  --log-every 10 \
  --save-every 50 \
  --keep-last-k-checkpoints 4 \
  --learning-rate 8e-5 \
  --scheduler constant \
  --warmup-steps 0 \
  --resume-from outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/checkpoint_step_00000550.pt \
  --allow-resume-config-mismatch \
  --resume-model-only \
  --checkpoint-dir outputs/imagenet/latent_recovery_from550_step700_modelonly/checkpoints \
  --checkpoint-path outputs/imagenet/latent_recovery_from550_step700_modelonly/checkpoint.pt \
  --output-dir outputs/imagenet/latent_recovery_from550_step700_modelonly
```

## 2) Post-run chain (alpha sweep + NN audit)

```bash
./scripts/run_latent_alpha_nn_chain.sh \
  outputs/imagenet/latent_recovery_from550_step700_modelonly/checkpoints/checkpoint_step_00000700.pt \
  configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml \
  outputs/imagenet/latent_recovery_from550_step700_modelonly_alpha_sweep_s2k \
  cuda:0
```
