from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariantArtifacts:
    name: str
    train_summary: Path
    alpha_sweep_summary: Path
    nn_audit: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _extract_lr(train_summary: dict[str, Any]) -> float | None:
    train_config = train_summary.get("train_config")
    if not isinstance(train_config, dict):
        return None
    lr = train_config.get("learning_rate")
    return float(lr) if isinstance(lr, (int, float)) else None


def _extract_resume_mode(train_summary: dict[str, Any]) -> str:
    train_config = train_summary.get("train_config")
    if not isinstance(train_config, dict):
        return "unknown"
    if train_config.get("resume_model_only") is True:
        return "model_only"
    reset_sched = train_config.get("resume_reset_scheduler") is True
    reset_lr = train_config.get("resume_reset_optimizer_lr") is True
    if reset_sched or reset_lr:
        return "restore_opt+resets"
    if train_summary.get("resume_from") is not None:
        return "restore_opt"
    return "fresh"


def _best_by_fid(alpha_sweep_summary: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    results = alpha_sweep_summary.get("results")
    if not isinstance(results, list) or not results:
        return None, None, None
    best_alpha: float | None = None
    best_fid: float | None = None
    best_is: float | None = None
    for entry in results:
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        fid = metrics.get("fid_pretrained", metrics.get("fid"))
        is_mean = metrics.get("inception_score_pretrained_mean", metrics.get("inception_score_mean"))
        if not isinstance(fid, (int, float)):
            continue
        alpha = entry.get("alpha")
        if not isinstance(alpha, (int, float)):
            alpha = None
        if best_fid is None or float(fid) < best_fid:
            best_fid = float(fid)
            best_alpha = float(alpha) if alpha is not None else None
            best_is = float(is_mean) if isinstance(is_mean, (int, float)) else None
    return best_alpha, best_fid, best_is


def _extract_nn_mean_cosine(nn_audit: dict[str, Any]) -> float | None:
    value = nn_audit.get("mean_cosine_similarity")
    return float(value) if isinstance(value, (int, float)) else None


def _variant_artifacts(base_out: Path, variant: str) -> VariantArtifacts:
    return VariantArtifacts(
        name=variant,
        train_summary=base_out / variant / "latent_summary.json",
        alpha_sweep_summary=base_out / f"{variant}_alpha_sweep_s2k" / "alpha_sweep_summary.json",
        nn_audit=base_out / f"{variant}_alpha_sweep_s2k" / "alpha_1p5" / "nn_audit.json",
    )


def _render_markdown(*, base_out: Path, variants: list[str]) -> str:
    rows: list[str] = []
    for variant in variants:
        art = _variant_artifacts(base_out, variant)

        lr: float | None = None
        resume_mode = "pending"
        if art.train_summary.exists():
            train = _read_json(art.train_summary)
            lr = _extract_lr(train)
            resume_mode = _extract_resume_mode(train)

        best_alpha: float | None = None
        best_fid: float | None = None
        best_is: float | None = None
        if art.alpha_sweep_summary.exists():
            sweep = _read_json(art.alpha_sweep_summary)
            best_alpha, best_fid, best_is = _best_by_fid(sweep)

        nn_mean_cosine: float | None = None
        if art.nn_audit.exists():
            nn = _read_json(art.nn_audit)
            nn_mean_cosine = _extract_nn_mean_cosine(nn)

        rows.append(
            "| "
            + " | ".join(
                [
                    variant,
                    resume_mode,
                    _format_float(lr, digits=6),
                    _format_float(best_alpha, digits=2),
                    _format_float(best_fid, digits=2),
                    _format_float(best_is, digits=5),
                    _format_float(nn_mean_cosine, digits=4),
                ]
            )
            + " |"
        )

    header = "\n".join(
        [
            "# ImageNet Recovery Matrix 2x2 Summary",
            "",
            f"- Base output: `{base_out}`",
            "",
            "## Table (best-by-FID per sweep)",
            "| Variant | Resume mode | LR | Best alpha | Best FID | IS@best | NN mean cosine |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    return header + "\n" + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a recovery_matrix_2x2 run directory.")
    parser.add_argument("--base-out", type=str, required=True)
    parser.add_argument("--output-md", type=str, default=None, help="Optional path to write markdown summary.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "A_restoreopt_lr8e5",
            "B_restoreopt_lr2e4",
            "C_modelonly_lr8e5",
            "D_modelonly_lr2e4",
        ],
    )
    args = parser.parse_args()

    base_out = Path(args.base_out)
    md = _render_markdown(base_out=base_out, variants=list(args.variants))
    if args.output_md is None:
        print(md, end="")
        return
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

