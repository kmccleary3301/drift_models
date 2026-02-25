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
  - Class embedding + alpha MLP embedding + (optional) style embedding sum.
- In-context conditioning tokens:
  - Implemented as `register_tokens` (default `16` in Table 8 configs).
  - `register_base + register_positional_embedding + condition_to_register(condition)`.
  - Prepended to the patch token sequence.
- adaLN-zero:
  - Two norms + attention/MLP residuals gated and modulated by `(shift, scale, gate)` from conditioning.
  - Modulation final layer is zero-initialized.
- Transformer components:
  - SwiGLU feed-forward (`SwiGLUFeedForward`).
  - RoPE (`apply_rope`) optional via `use_rope`.
  - QK-Norm optional via `use_qk_norm` (L2-normalize q/k).
  - RMSNorm available via `norm_type=rmsnorm`.

Config parity for paper Table 8 is represented in configs like:
- `configs/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`
  - `hidden-dim: 1024`, `depth: 24`, `num-heads: 16`
  - `register-tokens: 16`
  - `norm-type: rmsnorm`
  - `use-qk-norm: true`
  - `use-rope: true`

## Differences / Ambiguities

These are places the paper scan does not fully specify details, so we may not be identical to the authors’ code:

1. Patch positional embeddings:
   - We add a learnable `patch_positional_embedding` to patch tokens.
   - Paper explicitly mentions positional embeddings for in-context tokens; it does not clearly state whether patch tokens also have absolute positional embeddings when RoPE is used.

2. RMSNorm “affine” behavior under adaLN:
   - Our `RMSNorm` includes a learnable per-dim `weight`.
   - For `LayerNorm`, we use `elementwise_affine=False` (consistent with DiT-style modulation).
   - Paper cites RMSNorm but does not specify whether its gain is present/absent/frozen under adaLN.

3. RoPE layout:
   - Our RoPE is 1D over the flattened token index.
   - Paper does not specify 1D vs 2D RoPE for the 2D patch grid.

## Verdict

“Paper-backed” generator architectural features listed in Appendix A.2 are implemented and configurable:

- SwiGLU: yes
- RoPE: yes (optional)
- RMSNorm: yes (optional)
- QK-Norm: yes (optional)
- adaLN-zero: yes
- In-context/register tokens (16): yes
- Style embeddings (32 tokens, vocab 64): yes

The remaining gaps are mostly *spec ambiguities* (pos-emb and norm details), not missing modules.

## Recommended Next Actions (If We Want Closer Parity)

1. Make patch positional embedding optional (e.g., `use_abs_pos_emb: bool`) and test with RoPE-only.
2. Add an RMSNorm “no-affine” option (or freeze gain) to mirror LayerNorm’s `elementwise_affine=False` behavior.
3. Add a 2D RoPE option for the patch grid (latent and pixel variants).
4. Add a small “generator parity unit test” to assert:
   - token counts match expected `(register_tokens + num_patches)`
   - style embedding path produces deterministic shapes
   - RoPE requires even head_dim

