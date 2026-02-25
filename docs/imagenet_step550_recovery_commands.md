# ImageNet Step-550 Recovery Continuation

This runbook is the **current default** for continuing from the `step=550` checkpoint toward `step=700`.

Current best evidence (2x2 recovery matrix, post-restart):
- `docs/imagenet_recovery_matrix_2x2_20260213_restart_r3_results.md`

## 1) Train continuation (step 550 -> 700)

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
  --resume-reset-scheduler \
  --resume-reset-optimizer-lr \
  --checkpoint-dir outputs/imagenet/latent_recovery_from550_step700_default/checkpoints \
  --checkpoint-path outputs/imagenet/latent_recovery_from550_step700_default/checkpoint.pt \
  --output-dir outputs/imagenet/latent_recovery_from550_step700_default
```

## 2) Post-run alpha sweep + NN audit (step-700 checkpoint, 2k samples/alpha)

```bash
./scripts/run_latent_alpha_nn_chain.sh \
  outputs/imagenet/latent_recovery_from550_step700_default/checkpoints/checkpoint_step_00000700.pt \
  configs/latent/imagenet1k_sdvae_latents_recovery_from550_step700_w64.yaml \
  outputs/imagenet/latent_recovery_from550_step700_default_alpha_sweep_s2k \
  cuda:0
```

This chain gates on the immutable step checkpoint (avoids races with mutable `checkpoint.pt`).
