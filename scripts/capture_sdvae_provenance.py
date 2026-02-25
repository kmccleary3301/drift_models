from __future__ import annotations

import argparse
import json
from pathlib import Path

from drifting_models.utils import environment_fingerprint, file_sha256, write_json


def main() -> None:
    args = _parse_args()
    model_id = str(args.model_id)
    revision = None if args.revision is None or not str(args.revision).strip() else str(args.revision)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("huggingface_hub is required (it should come with the sdvae extra).") from error

    allow_patterns = [
        ".gitattributes",
        "README.md",
        "config.json",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors",
    ]
    snapshot_dir = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    ).resolve()

    files: list[dict[str, object]] = []
    for rel in allow_patterns:
        path = snapshot_dir / rel
        if not path.exists():
            continue
        files.append(
            {
                "path": rel,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )

    payload = {
        "kind": "sd_vae_provenance",
        "model_id": model_id,
        "revision": revision,
        "snapshot_dir": str(snapshot_dir),
        "files": files,
        "env_fingerprint": environment_fingerprint(),
    }

    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, payload)
    print(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture SD-VAE (diffusers AutoencoderKL) provenance from HF cache.")
    p.add_argument("--model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--revision", type=str, default="31f26fdeee1355a5c34592e401dd41e45d25a493")
    p.add_argument("--output-path", type=str, default="outputs/datasets/sdvae_provenance.json")
    return p.parse_args()


if __name__ == "__main__":
    main()

