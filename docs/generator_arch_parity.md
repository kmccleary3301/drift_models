# Generator Architecture Parity Notes (DiT-like)

This note records the evidence that the repo’s generator matches the paper’s stated DiT-like architecture for ImageNet latent generation.

Primary paper reference sections:
- main text “Architecture” (Section 4)
- Appendix A.2 “Transformer” (scan text around `Drift_Models/Drift_Models.md` “Transformer.”)

## Evidence classification (PAPER-BACKED vs INFERRED vs OPEN)

This table is the “audit spine” for the generator. Each row must be traceable to either:
- scanned paper text/tables (PAPER-BACKED),
- strongly implied behavior needed for the pipeline (INFERRED),
- or an ambiguity requiring an explicit decision + ablation (OPEN).

| Component | Status | Paper evidence | Repo implementation pointer | Notes |
| :-- | :-- | :-- | :-- | :-- |
| Latent I/O shape (`32x32x4`) | PAPER-BACKED | latent protocol description | `drifting_models/models/dit_like.py::DiTLikeConfig` | Must match SD-VAE latents. |
| Patch size `2x2` (latent) | PAPER-BACKED | appendix architecture table | `DiTLikeGenerator.patch_embed` | Implies `16x16=256` tokens. |
| DiT-like transformer backbone | PAPER-BACKED | Architecture section | `drifting_models/models/dit_like.py` | Concrete block structure in appendix. |
| Class conditioning | PAPER-BACKED | conditional generation protocol | `DiTLikeGenerator._build_conditioning()` | Embedded as class embedding. |
| Alpha conditioning (`α`) | PAPER-BACKED | CFG-by-negative mixing description | `DiTLikeGenerator._build_conditioning()` | Scalar → MLP → conditioning vector. |
| adaLN / adaLN-zero modulation | PAPER-BACKED | Appendix transformer description | `AdaLNZeroBlock` | Final projection is zero-init (paper-aligned). |
| SwiGLU FFN | PAPER-BACKED | Appendix transformer details | `SwiGLUFeedForward` |  |
| RMSNorm | PAPER-BACKED | Appendix normalization (“RM-SNorm”) | `RMSNorm` / `make_norm(..., "rmsnorm")` |  |
| RoPE positional strategy | PAPER-BACKED | Appendix positional strategy | `DiTAttention` (`use_rope`) | Config-gated. |
| QK-Norm | PAPER-BACKED | Appendix stabilization | `DiTAttention` (`use_qk_norm`) | Config-gated. |
| Register tokens (16) | PAPER-BACKED | Appendix table | `DiTLikeConfig.register_tokens` | Implemented as prepended tokens. |
| “In-context conditioning tokens” injection | INFERRED | paper phrasing; details unclear | `DiTLikeGenerator.condition_to_register` | We map this onto register-token shifting; verify vs paper wording. |
| Random style embeddings (32 from 64) | PAPER-BACKED | appendix “random style embeddings” | `style_vocab_size`, `style_token_count` | Summed into conditioning. |
| Exact init constants (`N(0,0.02)` etc.) | INFERRED | paper typically references DiT init; not always explicit | `DiTLikeGenerator.__init__` | Captured below. |

## Checklist (paper → repo)

### Input/output shape (latent protocol)
- Paper: input noise \(\epsilon\) is `32x32x4`, output latent \(\mathbf{x}\) is `32x32x4`.
- Repo:
  - `drifting_models/models/dit_like.py::DiTLikeConfig(in_channels=4, out_channels=4, image_size=32)`
  - `drifting_models/models/dit_like.py::DiTLikeGenerator.forward()` expects `[B, C, H, W]` and unpatchifies back to `[B, out_channels, image_size, image_size]`.

### Patchification / token count
- Paper: patch size `2x2` for latent, yielding `16x16=256` tokens.
- Repo:
  - `drifting_models/models/dit_like.py::DiTLikeGenerator.patch_embed` uses `kernel_size=stride=patch_size`.
  - Token count is `grid_size=(image_size/patch_size)` and `num_patches=grid_size^2`.

### Conditioning: class \(c\) and CFG scale \(\alpha\)
- Paper: class-conditioning and \(\alpha\) are processed by adaLN / adaLN-zero.
- Repo:
  - `drifting_models/models/dit_like.py::DiTLikeGenerator._build_conditioning()` sums:
    - `class_embedding(c)`
    - MLP embedding of scalar `alpha`
    - optional style embedding sum (see below)
  - `drifting_models/models/dit_like.py::AdaLNZeroBlock` uses a zero-initialized modulation projection producing shift/scale/gates for attn+MLP.

### Feedforward: SwiGLU
- Paper: uses SwiGLU.
- Repo: `drifting_models/models/dit_like.py::SwiGLUFeedForward`.

### Positional strategy: RoPE
- Paper: uses RoPE.
- Repo: optional `use_rope` in `DiTAttention`; enabled in ImageNet configs (e.g. `use-rope: true`).

### Normalization: RMSNorm (“RM-SNorm”)
- Paper: uses RMSNorm.
- Repo:
  - `drifting_models/models/dit_like.py::RMSNorm`
  - `make_norm(..., norm_type="rmsnorm")`
  - enabled in ImageNet configs (`norm-type: rmsnorm`).

### Attention stabilization: QK-Norm
- Paper: uses QK-Norm.
- Repo: optional `use_qk_norm` in `DiTAttention`; enabled in ImageNet configs (`use-qk-norm: true`).

### “Random style embeddings”
- Paper: adds 32 random style token indices into a 64-entry codebook, summed into the conditioning vector.
- Repo:
  - `DiTLikeConfig.style_vocab_size=64`, `style_token_count=32`
  - `DiTLikeGenerator._build_conditioning()` sums embeddings over the style-token axis and adds into conditioning.

### Register tokens / in-context conditioning tokens
- Paper: uses register tokens (reported as 16 in the appendix table).
- Repo:
  - `DiTLikeConfig.register_tokens` (commonly set to 16 in ImageNet configs)
  - `DiTLikeGenerator` prepends `register_tokens` to the token sequence and injects conditioning via `condition_to_register`.

## Parameter counts (repo implementation)

Regenerate:
```bash
uv run python scripts/report_ditlike_params.py \
  --hidden-dim 768 --depth 12 --num-heads 12 --use-qk-norm --use-rope --ffn-inner-dim 2184

uv run python scripts/report_ditlike_params.py \
  --hidden-dim 1024 --depth 24 --num-heads 16 --use-qk-norm --use-rope --ffn-inner-dim 2824
```

Paper reference (scan): Table 5 reports:
- Drifting Model, B/2: `133M` (generator) + `49M` (SD-VAE)
- Drifting Model, L/2: `463M` (generator) + `49M` (SD-VAE)

Observed (repo, 2026-02-14):
- DiT-B/2-like (H=768, D=12, heads=12, `ffn-inner-dim=2184`): `133,088,720` params
- DiT-L/2-like (H=1024, D=24, heads=16, `ffn-inner-dim=2824`): `462,922,384` params

Notes:
- The paper does not specify the exact SwiGLU intermediate dimension. The repo supports an explicit `ffn-inner-dim` override so the Table-8 configs can match the paper’s reported parameter counts tightly.

## Generator ablation matrix (repo artifacts)

- Latest short ablation bundle (cuda:0): `outputs/stage2_generator_arch_ablations_cuda0_20260215_003154/generator_arch_ablation_summary.md`
- Prior bundle (cuda:1 host): `outputs/stage2_generator_arch_ablations_cuda1/generator_arch_ablation_summary.md`

## Initialization (repo implementation)

Key explicitly-specified initializations:
- Patch positional embeddings: `N(0, 0.02)` (`DiTLikeGenerator.patch_positional_embedding`)
- Register base + register positional embeddings: `N(0, 0.02)` (`register_base`, `register_positional_embedding`)
- adaLN-zero modulation projection: final linear in `AdaLNZeroBlock.modulation` is zero-initialized
  (`nn.init.zeros_(weight/bias)`), so the network starts near an identity residual update.

All other layers use PyTorch module defaults unless explicitly overridden (e.g., Linear/Conv default initializers).

## Notes / residual parity questions
- The paper describes “in-context conditioning tokens”; the repo’s implementation maps this onto register tokens whose base embeddings are shifted by a learned projection of the conditioning vector. If the paper uses a different tokenization or injection scheme, we should align it explicitly.
