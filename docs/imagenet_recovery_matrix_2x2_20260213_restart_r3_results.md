# ImageNet Recovery Matrix 2x2 Summary

- Base output: `outputs/imagenet/recovery_matrix_2x2_20260213_restart_r3`

## Protocol (constant across variants)
- Resume source checkpoint: `outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/checkpoint_step_00000550.pt`
- Terminal step: `700`
- Alpha sweep: `alphas = [1.0, 1.5, 2.0, 2.5, 3.0]`, `n_samples = 2000` per alpha
- Decode + eval: SD-VAE decode to `256x256`, pretrained Inception, cached ImageNet val stats at `outputs/datasets/imagenet1k_val_reference_stats_pretrained.pt`
- NN audit: `alpha=1.5`, `max_generated=512`, `max_reference=10000`
- Hardware lane: `cuda:0` (single GPU currently visible on this host)

## Table (best-by-FID per sweep)
| Variant | Resume mode | LR | Best alpha | Best FID | IS@best | NN mean cosine |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| A_restoreopt_lr8e5 | restore_opt+resets | 0.000080 | 1.00 | 276.14 | 1.63121 | 0.2511 |
| B_restoreopt_lr2e4 | restore_opt+resets | 0.000200 | 3.00 | 374.78 | 1.83074 | 0.2558 |
| C_modelonly_lr8e5 | model_only | 0.000080 | 2.50 | 332.12 | 1.02644 | 0.1914 |
| D_modelonly_lr2e4 | model_only | 0.000200 | 3.00 | 356.78 | 1.04602 | 0.2129 |

## Takeaways (at this horizon + protocol)
- **Best FID** is `A_restoreopt_lr8e5` (optimizer restored, but with **scheduler reset + optimizer LR overridden**).
- **High LR (`2e-4`) is consistently worse** than low LR (`8e-5`) in this recovery window (compare A vs B, C vs D).
- **Optimizer restore is not inherently bad**: with resets + low LR (A), it materially outperforms model-only reset at the same LR (C).

## Recommended default for step-550→700 recovery (current)
- Prefer: restore optimizer state **with** `--resume-reset-scheduler --resume-reset-optimizer-lr` and a conservative resume LR (`~8e-5`), unless later evidence contradicts.
