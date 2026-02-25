from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    args = _parse_args()
    archives_root = Path(args.archives_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    required = {
        "train": "ILSVRC2012_img_train.tar",
        "val": "ILSVRC2012_img_val.tar",
        "devkit": "ILSVRC2012_devkit_t12.tar.gz",
    }
    sources = {
        "train": Path(args.train_archive_path).expanduser().resolve()
        if args.train_archive_path is not None
        else (archives_root / required["train"]),
        "val": Path(args.val_archive_path).expanduser().resolve()
        if args.val_archive_path is not None
        else (archives_root / required["val"]),
        "devkit": Path(args.devkit_archive_path).expanduser().resolve()
        if args.devkit_archive_path is not None
        else (archives_root / required["devkit"]),
    }

    missing = []
    for name, path in sources.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Missing required ImageNet archives:\n" + "\n".join(missing))

    output_root.mkdir(parents=True, exist_ok=True)

    train_dir = output_root / "train"
    val_dir = output_root / "val"
    if not args.allow_existing:
        collisions = [p for p in (train_dir, val_dir) if p.exists()]
        if collisions:
            raise FileExistsError(
                "Output already contains extracted split(s):\n"
                + "\n".join(str(p) for p in collisions)
                + "\nPass --allow-existing to proceed."
            )

    _ensure_archives_available(
        sources=sources,
        dest_names=required,
        output_root=output_root,
        link_mode=str(args.link_mode),
    )
    if not args.skip_provenance:
        provenance_path = (
            Path(args.provenance_path).expanduser().resolve()
            if args.provenance_path is not None
            else (output_root.parent / "imagenet1k_provenance.json")
        )
        _write_archive_provenance(sources=sources, provenance_path=provenance_path)

    if args.skip_md5:
        _extract_without_md5(args=args, output_root=output_root)
        return

    # Use torchvision's official ImageNet parsing logic so that:
    # - devkit parsing produces meta.bin consistent with torchvision ImageNet
    # - train archive extraction expands synset tarballs into directories
    # - val archive extraction uses devkit ground truth mapping
    from torchvision.datasets.imagenet import parse_devkit_archive, parse_train_archive, parse_val_archive

    split = args.split
    if split in ("all", "devkit"):
        parse_devkit_archive(str(output_root))
    if split in ("all", "train"):
        parse_train_archive(str(output_root))
    if split in ("all", "val"):
        parse_val_archive(str(output_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ImageNet-1k ImageFolder from official ILSVRC2012 archives.")
    parser.add_argument("--archives-root", type=str, default="data", help="Directory holding the downloaded archives.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/datasets/imagenet1k_raw",
        help="Directory to populate with train/ and val/ folders.",
    )
    parser.add_argument(
        "--train-archive-path",
        type=str,
        default=None,
        help="Optional override path to ILSVRC2012_img_train.tar (useful if the file is named differently).",
    )
    parser.add_argument(
        "--val-archive-path",
        type=str,
        default=None,
        help="Optional override path to ILSVRC2012_img_val.tar (useful if the file is named differently).",
    )
    parser.add_argument(
        "--devkit-archive-path",
        type=str,
        default=None,
        help="Optional override path to ILSVRC2012_devkit_t12.tar.gz (useful if the file is named differently).",
    )
    parser.add_argument("--split", choices=("all", "devkit", "train", "val"), default="all")
    parser.add_argument("--allow-existing", action="store_true", help="Allow output train/val dirs to already exist.")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip torchvision MD5 verification and extract archives directly (useful for non-standard variants).",
    )
    parser.add_argument(
        "--provenance-path",
        type=str,
        default=None,
        help="Optional output path for archive checksum provenance JSON. "
        "Defaults to <output-root-parent>/imagenet1k_provenance.json.",
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Disable checksum provenance writeout.",
    )
    return parser.parse_args()


def _ensure_archives_available(
    *,
    sources: dict[str, Path],
    dest_names: dict[str, str],
    output_root: Path,
    link_mode: str,
) -> None:
    import shutil

    for key, filename in dest_names.items():
        src = sources[key]
        dst = output_root / filename
        if dst.exists():
            continue
        if link_mode == "symlink":
            os.symlink(src, dst)
        elif link_mode == "hardlink":
            os.link(src, dst)
        elif link_mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unsupported link_mode: {link_mode}")


def _extract_without_md5(*, args: argparse.Namespace, output_root: Path) -> None:
    # Minimal extraction fallback without MD5 checks. This is not identical to torchvision's ImageNet
    # parsing when devkit is involved; prefer default mode for paper-faithful ILSVRC2012.
    from torchvision.datasets.utils import extract_archive

    split = args.split
    if split in ("all", "devkit"):
        # Resume-friendly: if meta.bin already exists, assume devkit already extracted.
        if not (output_root / "meta.bin").exists():
            extract_archive(str(output_root / "ILSVRC2012_devkit_t12.tar.gz"), str(output_root))
    if split in ("all", "val"):
        val_root = output_root / "val"
        # Resume-friendly: skip extraction if val_root already contains class dirs.
        has_val_dirs = val_root.exists() and any(p.is_dir() for p in val_root.iterdir())
        if not has_val_dirs:
            extract_archive(str(output_root / "ILSVRC2012_img_val.tar"), str(val_root))
    if split in ("all", "train"):
        train_root = output_root / "train"
        # Resume-friendly: avoid re-extracting the giant train tar if it looks like we've already done it.
        # After the first extraction, train_root contains many `n*.tar` synset archives and/or extracted wnid dirs.
        has_synset_tars = train_root.exists() and any(p.is_file() and p.suffix == ".tar" for p in train_root.iterdir())
        has_wnid_dirs = train_root.exists() and any(p.is_dir() and p.name.startswith("n") for p in train_root.iterdir())
        if not (has_synset_tars or has_wnid_dirs):
            extract_archive(str(output_root / "ILSVRC2012_img_train.tar"), str(train_root))

        # Expand per-synset tarballs (idempotent; remove_finished=True will clean up as we go).
        for archive in train_root.iterdir():
            if archive.is_file() and archive.suffix == ".tar":
                extract_archive(str(archive), str(archive.with_suffix("")), remove_finished=True)


def _write_archive_provenance(*, sources: dict[str, Path], provenance_path: Path) -> None:
    existing = _load_existing_provenance(provenance_path)
    archives: list[dict[str, object]] = []
    for role, src in sorted(sources.items(), key=lambda item: item[0]):
        stat = src.stat()
        src_path = str(src)
        size_bytes = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        cached = existing.get(src_path)
        md5_hash: str | None = None
        sha256_hash: str | None = None
        if (
            cached is not None
            and int(cached.get("size_bytes", -1)) == size_bytes
            and int(cached.get("mtime_ns", -1)) == mtime_ns
            and isinstance(cached.get("md5"), str)
            and isinstance(cached.get("sha256"), str)
        ):
            md5_hash = str(cached["md5"])
            sha256_hash = str(cached["sha256"])
        else:
            md5_hash, sha256_hash = _compute_hashes(src)
        archives.append(
            {
                "role": role,
                "path": src_path,
                "filename": src.name,
                "size_bytes": size_bytes,
                "mtime_ns": mtime_ns,
                "hashes": {"md5": md5_hash, "sha256": sha256_hash},
            }
        )

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "generator": "scripts/prepare_imagenet1k_from_archives.py",
        "archives": archives,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_existing_provenance(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    archives = payload.get("archives")
    if not isinstance(archives, list):
        return {}
    cached: dict[str, dict[str, object]] = {}
    for entry in archives:
        if not isinstance(entry, dict):
            continue
        src_path = entry.get("path")
        hashes = entry.get("hashes")
        if not isinstance(src_path, str) or not isinstance(hashes, dict):
            continue
        md5_hash = hashes.get("md5")
        sha256_hash = hashes.get("sha256")
        if not isinstance(md5_hash, str) or not isinstance(sha256_hash, str):
            continue
        cached[src_path] = {
            "size_bytes": entry.get("size_bytes"),
            "mtime_ns": entry.get("mtime_ns"),
            "md5": md5_hash,
            "sha256": sha256_hash,
        }
    return cached


def _compute_hashes(path: Path) -> tuple[str, str]:
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    return md5_hash.hexdigest(), sha256_hash.hexdigest()


if __name__ == "__main__":
    main()
