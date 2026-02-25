# ImageNet Step-550 Recovery Matrix 2x2 (2026-02-13, post-restart)

## Scope
This run is intended to isolate whether the step-600 quality collapse is primarily driven by:
- optimizer-state carryover vs optimizer reset, and
- learning-rate choice at resume time (low vs high).

Common source checkpoint:
- `outputs/imagenet/latent_ablation_b2_600_cuda1_w64/checkpoints/checkpoint_step_00000550.pt`

Common protocol:
- terminal step: `700`
- eval: alpha sweep (`2k` samples/alpha, pretrained Inception, cached ImageNet val stats)
- NN audit: `alpha=1.5`, `max_generated=512`, `max_reference=10000`
- device: `cuda:0` (this machine currently exposes a single GPU)

## How it is launched
- tmux session: `recovery_matrix_2x2`
- runner: `scripts/run_recovery_matrix_2x2.sh`
- base output: `outputs/imagenet/recovery_matrix_2x2_20260213_restart_r3`

## Variants
| Variant | Optimizer state | LR on resume |
| :--- | :--- | :--- |
| `A_restoreopt_lr8e5` | restored (with scheduler reset + optimizer lr reset) | `8e-5` |
| `B_restoreopt_lr2e4` | restored (with scheduler reset + optimizer lr reset) | `2e-4` |
| `C_modelonly_lr8e5` | reset (`--resume-model-only`) | `8e-5` |
| `D_modelonly_lr2e4` | reset (`--resume-model-only`) | `2e-4` |

## Expected artifacts (per variant)
- Train summary: `.../<variant>/latent_summary.json`
- Terminal checkpoint: `.../<variant>/checkpoints/checkpoint_step_00000700.pt`
- Alpha sweep summary: `.../<variant>_alpha_sweep_s2k/alpha_sweep_summary.json`
- NN audit: `.../<variant>_alpha_sweep_s2k/alpha_1p5/nn_audit.json`

## Results
After runs finish, generate a comparison table:

```bash
uv run python scripts/summarize_recovery_matrix_2x2.py \
  --base-out outputs/imagenet/recovery_matrix_2x2_20260213_restart_r3 \
  --output-md docs/imagenet_recovery_matrix_2x2_20260213_restart_r3_results.md
```

