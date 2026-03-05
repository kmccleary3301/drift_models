# Generator Parity Audit (Paper Appendix A.2)

This doc audits our generator implementation against the scanned paper text in `Drift_Models/Drift_Models.md` (Appendix A.2 “Generator Architecture”).

## Scope

- Paper: Appendix A.2 “Generator Architecture”
- Code: `drifting_models/models/dit_like.py`
- Primary target: latent-space ImageNet 256x256 generation (SD-VAE latents, 32x32x4)

## Paper Requirements (Appendix A.2)

From `Drift_Models/Drift_Models.md`:

- DiT-style Transformer backbone.
- Patchify Gaussian noise into `16 x 16 = 256` tokens.
  - Latent: patch size `2 x 2` (on `32 x 32` latent) -> `16 x 16` tokens.
  - Pixel: patch size `16 x 16` (on `256 x 256`) -> `16 x 16` tokens.
- Conditioning inputs: `(c, alpha)`; processed by adaLN as well as by in-context conditioning tokens.
- “Following (Yao et al., 2025)”:
  - SwiGLU MLP.
  - RoPE in attention.
  - RM-SNorm (RMSNorm).
  - QK-Norm.
- In-context tokens:
  - Prepend `16` learnable tokens (a.k.a. “register tokens” in Table 8).
  - Tokens are formed by adding a projected conditioning vector and positional embeddings.
- Random style embeddings:
  - `32` “style tokens”: each is a random index into a codebook of `64` learnable embeddings.
  - Style embeddings are summed and added to the conditioning vector (no sequence-length change).

## What We Implement Today

Implementation: `drifting_models/models/dit_like.py`.

- Patchify/unpatchify:
  - `patch_embed: Conv2d(kernel=stride=patch_size)`, flattened into tokens.
  - `output_projection -> _unpatchify` back to `[B, C, H, W]`.
- Conditioning vector:
  - Class embedding + configurable alpha embedding (`mlp` or `fourier_mlp`) + (optional) style embedding sum.
- In-context conditioning tokens:
  - Implemented as `register_tokens` (default `16` in Table 8 configs).
  - `register_base + register_positional_embedding + condition_to_register(condition)`.
  - Prepended to the patch token sequence.
- adaLN-zero:
  - Two norms + attention/MLP residuals gated and modulated by `(shift, scale, gate)` from conditioning.
  - Modulation final layer is zero-initialized.
- Transformer components:
  - SwiGLU feed-forward (`SwiGLUFeedForward`).
  - RoPE configurable via `rope_mode` (`none`, `1d_flat`, `2d_axial`; `auto` keeps legacy `use_rope` behavior).
  - QK-Norm configurable via `qk_norm_mode` (`none`, `l2`; `auto` keeps legacy `use_qk_norm` behavior).
  - RMSNorm available via `norm_type=rmsnorm` with optional affine gain toggle (`rmsnorm_affine`).
  - Patch absolute positional embeddings can be toggled on/off (`use_patch_positional_embedding`).

Config parity for paper Table 8 is represented in configs like:
- `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`
  - `hidden-dim: 1024`, `depth: 24`, `num-heads: 16`
  - `register-tokens: 16`
  - `norm-type: rmsnorm`
  - `use-qk-norm: true`
  - `use-rope: true`

## Differences / Ambiguities

These are places the paper scan does not fully specify details, so we may not be identical to the authors’ code:

1. Alpha embedding micro-variant:
   - Paper does not fully pin the exact conditioning parameterization details.
   - Repo now provides explicit selectable variants (`mlp`, `fourier_mlp`) so runs can match a chosen interpretation and record it.

2. Positional/RoPE micro-variant:
   - Paper does not clearly specify absolute patch pos-emb presence with RoPE or 1D vs 2D RoPE layout.
   - Repo now exposes `use_patch_positional_embedding` and `rope_mode` (`1d_flat`/`2d_axial`) so both plausible interpretations are testable.

3. QK/RMSNorm micro-variant:
   - Paper cites QK-Norm + RMSNorm but not all implementation details.
   - Repo now exposes `qk_norm_mode` and `rmsnorm_affine` to cover the main plausible variants.

## Verdict

“Paper-backed” generator architectural features listed in Appendix A.2 are implemented and configurable:

- SwiGLU: yes
- RoPE: yes (optional)
- RMSNorm: yes (optional)
- QK-Norm: yes (optional)
- adaLN-zero: yes
- In-context/register tokens (16): yes
- Style embeddings (32 tokens, vocab 64): yes

The remaining uncertainty is primarily at the **evidence-selection** level (which variant best matches the authors), not at missing implementation capability.

## Recommended Next Actions (If We Want Closer Parity)

1. For each claim-facing run, explicitly pin and log:
   - `alpha_embedding_type`, `rope_mode`, `qk_norm_mode`, `use_patch_positional_embedding`, `rmsnorm_affine`.
2. Add a small sweep over these variants on short runs and compare with nearest-neighbor / proxy metrics to choose default claim-facing micro-variant.
3. Keep `tests/unit/test_dit_like.py` parity checks as required CI coverage for new generator options.

Latest short-run micro-variant selection artifact:
- `docs/generator_microvariant_selection_20260303.md`
