# Generator Micro-Variant Selection (2026-03-03)

This note records the short no-heavy-compute sweep used to choose a default generator micro-variant tuple for claim-facing runs.

## Run artifact

- Summary JSON: `outputs/feature_ablations/generator_microvariant_sweep_20260303/generator_arch_ablation_summary.json`
- Summary markdown: `outputs/feature_ablations/generator_microvariant_sweep_20260303/generator_arch_ablation_summary.md`
- Runner: `scripts/experimental/ablations/generator_arch.py` (compatibility wrapper: `scripts/run_generator_arch_ablations.py`)

## Sweep scope

- Device: `cuda:0`
- Steps per variant: `24`
- Seed: `1337`
- Latent synthetic ablation profile (short-run stability/perf proxy, not paper-metric evidence)
- Variants compared:
  - `baseline_table8_like_auto`
  - `explicit_legacy_modes`
  - `rope2d_axial`
  - `rope2d_no_abs_pos`
  - `rope2d_no_abs_pos_rms_noaffine`
  - `fourier_alpha_rope2d_no_abs_pos`

## Result summary

- All variants were stable (`success=1`, finite metrics).
- `fourier_alpha_rope2d_no_abs_pos` had the weakest drift proxy in this sweep.
- `rope2d_no_abs_pos` and `rope2d_no_abs_pos_rms_noaffine` had the best drift-proxy values.
- `rope2d_no_abs_pos_rms_noaffine` additionally reduced peak VRAM materially vs other 2D-RoPE variants.

## Provisional default tuple

Use this as the default claim-facing micro-variant tuple unless future evidence (longer run + eval package) contradicts it:

- `alpha_embedding_type: mlp`
- `qk_norm_mode: l2`
- `rope_mode: 2d_axial`
- `use_patch_positional_embedding: false`
- `rmsnorm_affine: false`

Equivalent CLI flags:

- `--alpha-embedding-type mlp`
- `--qk-norm-mode l2`
- `--rope-mode 2d_axial`
- `--disable-patch-positional-embedding`
- `--disable-rmsnorm-affine`

## Pinning status

- This tuple is now pinned in faithful latent templates:
  - `configs/experimental/latent/imagenet1k_sdvae_latents_table8_ablation_default_template.yaml`
  - `configs/experimental/latent/imagenet1k_sdvae_latents_table8_b2_template.yaml`
  - `configs/experimental/latent/imagenet1k_sdvae_latents_table8_l2_template.yaml`
  - compatibility wrappers remain at the legacy `configs/latent/*template.yaml` paths.
- Contract enforcement:
  - `tests/unit/test_table8_faithful_config_contract.py`

## Evidence caution

This selection is based on short synthetic latent runs; it closes implementation selection for defaults, but not paper-metric parity evidence.
