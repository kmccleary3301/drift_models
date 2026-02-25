from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


_STEP_RE = re.compile(r"checkpoint_step_(\d+)\.pt$")


@dataclass(frozen=True)
class PruneResult:
    run_dir: str
    kept_step: int | None
    kept_checkpoint: str | None
    removed_checkpoints: int
    removed_bytes: int
    made_checkpoint_symlink: bool
    pruned_sample_images_dirs: int
    pruned_sample_images_files: int
    pruned_sample_images_bytes: int


def main() -> None:
    args = _parse_args()
    roots = [Path(p).expanduser().resolve() for p in args.roots]
    results: list[PruneResult] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(str(root))
        for run_dir in _iter_run_dirs(root):
            results.append(
                prune_run_dir(
                    run_dir=run_dir,
                    keep_last_n=int(args.keep_last_n),
                    prune_decoded_images=bool(args.prune_decoded_images),
                    dry_run=bool(args.dry_run),
                )
            )
        if args.prune_decoded_images:
            results.append(
                prune_decoded_images_under_root(
                    root=root,
                    dry_run=bool(args.dry_run),
                )
            )

    total_removed = sum(r.removed_bytes + r.pruned_sample_images_bytes for r in results)
    print(f"pruned_runs={len(results)} total_removed_bytes={total_removed}")
    if args.report_path:
        report_path = Path(args.report_path).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(results, dry_run=bool(args.dry_run)), encoding="utf-8")


def _iter_run_dirs(root: Path) -> list[Path]:
    if root.is_dir() and (root / "checkpoints").is_dir():
        return [root]
    candidates: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "checkpoints").is_dir():
            candidates.append(child)
    return sorted(candidates)


def prune_run_dir(*, run_dir: Path, keep_last_n: int, prune_decoded_images: bool, dry_run: bool) -> PruneResult:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_pt = run_dir / "checkpoint.pt"

    step_files: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint_step_*.pt"):
        match = _STEP_RE.search(path.name)
        if not match:
            continue
        step_files.append((int(match.group(1)), path))
    step_files.sort(key=lambda item: item[0])

    kept_step = None
    kept_checkpoint = None
    removed_bytes = 0
    removed_count = 0
    made_symlink = False

    if step_files:
        keep_last_n = max(1, int(keep_last_n))
        kept = step_files[-keep_last_n:]
        kept_step = kept[-1][0]
        kept_checkpoint = str(kept[-1][1])
        keep_paths = {p for _, p in kept}
        for step, path in step_files:
            if path in keep_paths:
                continue
            removed_count += 1
            removed_bytes += int(path.stat().st_size)
            if not dry_run:
                path.unlink()

        # Replace checkpoint.pt with symlink to the latest kept step checkpoint (saves space vs duplicate files).
        target = kept[-1][1]
        if checkpoint_pt.exists() and not checkpoint_pt.is_symlink():
            removed_bytes += int(checkpoint_pt.stat().st_size)
            if not dry_run:
                checkpoint_pt.unlink()
        if (not checkpoint_pt.exists()) or checkpoint_pt.is_symlink():
            if not dry_run:
                if checkpoint_pt.exists() and checkpoint_pt.is_symlink():
                    # Refresh if it points elsewhere.
                    checkpoint_pt.unlink()
                checkpoint_pt.symlink_to(target.relative_to(run_dir))
            made_symlink = True

    pruned_dirs = 0
    pruned_files = 0
    pruned_bytes = 0
    if prune_decoded_images:
        for images_dir in run_dir.rglob("samples/images"):
            if not images_dir.is_dir():
                continue
            pruned_dirs += 1
            for path in images_dir.rglob("*"):
                if not path.is_file():
                    continue
                pruned_files += 1
                pruned_bytes += int(path.stat().st_size)
            if not dry_run:
                shutil.rmtree(images_dir)

    return PruneResult(
        run_dir=str(run_dir),
        kept_step=kept_step,
        kept_checkpoint=kept_checkpoint,
        removed_checkpoints=int(removed_count),
        removed_bytes=int(removed_bytes),
        made_checkpoint_symlink=bool(made_symlink),
        pruned_sample_images_dirs=int(pruned_dirs),
        pruned_sample_images_files=int(pruned_files),
        pruned_sample_images_bytes=int(pruned_bytes),
    )


def prune_decoded_images_under_root(*, root: Path, dry_run: bool) -> PruneResult:
    pruned_dirs = 0
    pruned_files = 0
    pruned_bytes = 0
    for images_dir in root.rglob("samples/images"):
        if not images_dir.is_dir():
            continue
        pruned_dirs += 1
        for path in images_dir.rglob("*"):
            if not path.is_file():
                continue
            pruned_files += 1
            pruned_bytes += int(path.stat().st_size)
        if not dry_run:
            shutil.rmtree(images_dir)
    return PruneResult(
        run_dir=str(root),
        kept_step=None,
        kept_checkpoint=None,
        removed_checkpoints=0,
        removed_bytes=0,
        made_checkpoint_symlink=False,
        pruned_sample_images_dirs=int(pruned_dirs),
        pruned_sample_images_files=int(pruned_files),
        pruned_sample_images_bytes=int(pruned_bytes),
    )


def _render_report(results: list[PruneResult], *, dry_run: bool) -> str:
    lines: list[str] = []
    lines.append("# Disk Cleanup Report")
    lines.append("")
    lines.append(f"- Mode: {'DRY_RUN' if dry_run else 'APPLIED'}")
    lines.append(f"- Runs processed: {len(results)}")
    lines.append("")
    total_ckpt_bytes = sum(r.removed_bytes for r in results)
    total_img_bytes = sum(r.pruned_sample_images_bytes for r in results)
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Removed checkpoint bytes: {total_ckpt_bytes}")
    lines.append(f"- Removed decoded-image bytes: {total_img_bytes}")
    lines.append(f"- Removed total bytes: {total_ckpt_bytes + total_img_bytes}")
    lines.append("")
    lines.append("## Per-run")
    lines.append("")
    for r in results:
        lines.append(f"- `{r.run_dir}`")
        lines.append(f"  - kept_step: `{r.kept_step}`")
        lines.append(f"  - removed_checkpoints: `{r.removed_checkpoints}` ({r.removed_bytes} bytes)")
        lines.append(
            f"  - pruned_samples_images: {r.pruned_sample_images_dirs} dirs, {r.pruned_sample_images_files} files ({r.pruned_sample_images_bytes} bytes)"
        )
        lines.append(f"  - checkpoint_symlinked: `{r.made_checkpoint_symlink}`")
    lines.append("")
    return "\\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune per-step checkpoints and/or decoded sample imagefolders.")
    p.add_argument("--roots", nargs="+", required=True, help="Run root(s) or a parent containing run directories.")
    p.add_argument("--keep-last-n", type=int, default=1, help="Keep the last N checkpoint_step files per run dir.")
    p.add_argument("--prune-decoded-images", action="store_true", help="Delete any `samples/images` trees under each run dir.")
    p.add_argument("--dry-run", action="store_true", help="Report actions without deleting files.")
    p.add_argument("--report-path", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
