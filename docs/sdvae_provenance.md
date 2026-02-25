# SD‑VAE Provenance (Latent Tokenizer / Decoder)

This repo uses the SD‑VAE (as referenced by the paper) via **diffusers** `AutoencoderKL.from_pretrained`.

## Chosen weights source (pinned)
- Model ID: `stabilityai/sd-vae-ft-mse`
- Pinned revision (commit SHA): `31f26fdeee1355a5c34592e401dd41e45d25a493`

Rationale:
- Using `revision=None` implicitly tracks “latest”, which is not reproducible.
- The pinned commit SHA is stable and can be used across machines.

## Capture provenance into an artifact (recommended)

This writes a JSON containing the revision and **sha256** hashes of the cached files:
```bash
uv run python scripts/capture_sdvae_provenance.py \
  --model-id stabilityai/sd-vae-ft-mse \
  --revision 31f26fdeee1355a5c34592e401dd41e45d25a493 \
  --output-path outputs/datasets/sdvae_provenance.json
```

## Where it’s used
- Latent export: `scripts/export_sd_vae_latents_tensor_file.py` (`--model-id/--revision`)
- Decoding for sampling/eval: `scripts/sample_latent.py` (`--sd-vae-model-id/--sd-vae-revision`)

