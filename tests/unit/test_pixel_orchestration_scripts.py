from __future__ import annotations

import importlib
import json
from pathlib import Path


def _argv_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def test_paper_facing_sample_eval_package_reads_written_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("scripts.run_pixel_paper_facing_package")
    command_records: list[object] = []

    def fake_run(*, name: str, argv: list[str], cwd: Path):
        del cwd
        if "scripts/sample_pixel.py" in argv:
            sample_root = Path(_argv_value(argv, "--output-root"))
            sample_root.mkdir(parents=True, exist_ok=True)
            (sample_root / "sample_summary.json").write_text(
                json.dumps({"saved_images": 7}),
                encoding="utf-8",
            )
        if "scripts/eval_fid_is.py" in argv:
            eval_summary_path = Path(_argv_value(argv, "--output-path"))
            eval_summary_path.parent.mkdir(parents=True, exist_ok=True)
            eval_summary_path.write_text(
                json.dumps(
                    {
                        "fid": 123.4,
                        "inception_score_mean": 1.2,
                        "inception_score_std": 0.02,
                    }
                ),
                encoding="utf-8",
            )
        return module.CommandRecord(name=name, argv=argv, returncode=0, duration_seconds=0.01)

    monkeypatch.setattr(module, "_run", fake_run)

    output_root = tmp_path / "pkg"
    result = module._sample_eval_package(
        repo_root=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        config_path=None,
        output_root=output_root,
        alpha=1.5,
        n_samples=7,
        sample_batch_size=4,
        eval_batch_size=4,
        device="cpu",
        eval_profile="proxy",
        reference_imagefolder_root=tmp_path / "ref_imagefolder",
        reference_tensor_file=tmp_path / "ref.pt",
        load_reference_stats=tmp_path / "ref_stats.pt",
        allow_reference_contract_mismatch=False,
        command_records=command_records,
        name_prefix="alpha_1p5",
    )

    assert result["alpha"] == 1.5
    assert result["n_samples"] == 7
    assert result["sample_saved_images"] == 7
    assert result["fid"] == 123.4
    assert result["inception_score_mean"] == 1.2
    assert len(command_records) == 2


def test_paper_facing_markdown_contains_claim_and_tables() -> None:
    module = importlib.import_module("scripts.run_pixel_paper_facing_package")
    markdown = module._to_markdown(
        {
            "eval_profile": "pretrained_cached",
            "checkpoint_path": "/tmp/checkpoint.pt",
            "claim_result": {"alpha": 1.5, "n_samples": 64, "fid": 100.0, "inception_score_mean": 1.05},
            "last_k_results": [{"step": 100, "alpha": 1.5, "n_samples": 64, "fid": 99.0, "inception_score_mean": 1.06}],
            "alpha_sweep_results": [{"alpha": 1.0, "n_samples": 64, "fid": 101.0, "inception_score_mean": 1.01}],
        }
    )
    assert "Pixel Paper-Facing Package Summary" in markdown
    assert "## Claim" in markdown
    assert "## Last-K" in markdown
    assert "## Alpha Sweep" in markdown
    assert "99.0" in markdown


def test_proxy_ablation_variants_pin_expected_encoder_args() -> None:
    module = importlib.import_module("scripts.run_pixel_proxy_ablation_package")
    variants = module._build_variants()
    variant_by_name = {str(item["name"]): item for item in variants}

    assert set(variant_by_name) == {"tiny", "mae", "convnext_tiny"}
    mae_args = list(map(str, variant_by_name["mae"]["extra_train_args"]))
    convnext_args = list(map(str, variant_by_name["convnext_tiny"]["extra_train_args"]))
    assert "--mae-encoder-arch" in mae_args
    assert "paper_resnet34_unet" in mae_args
    assert "--convnext-weights" in convnext_args
    assert "none" in convnext_args


def test_feature_encoder_ablation_variants_include_convnext_and_paper_mae() -> None:
    module = importlib.import_module("scripts.run_pixel_feature_encoder_ablations")
    variants = module._build_variants()
    variant_by_name = {str(item["name"]): item for item in variants}

    assert set(variant_by_name) == {"tiny", "mae", "convnext_tiny"}
    mae_args = list(map(str, variant_by_name["mae"]["extra_args"]))
    convnext_args = list(map(str, variant_by_name["convnext_tiny"]["extra_args"]))
    assert "--mae-encoder-arch" in mae_args
    assert "paper_resnet34_unet" in mae_args
    assert "--convnext-weights" in convnext_args
    assert "none" in convnext_args


def test_paper_facing_eval_proxy_contract_args(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("scripts.run_pixel_paper_facing_package")
    command_records: list[object] = []

    def fake_run(*, name: str, argv: list[str], cwd: Path):
        del cwd
        if "scripts/sample_pixel.py" in argv:
            sample_root = Path(_argv_value(argv, "--output-root"))
            sample_root.mkdir(parents=True, exist_ok=True)
            (sample_root / "sample_summary.json").write_text(json.dumps({"saved_images": 5}), encoding="utf-8")
        if "scripts/eval_fid_is.py" in argv:
            output_path = Path(_argv_value(argv, "--output-path"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"fid": 1.0, "inception_score_mean": 1.0, "inception_score_std": 0.0}),
                encoding="utf-8",
            )
        return module.CommandRecord(name=name, argv=argv, returncode=0, duration_seconds=0.01)

    monkeypatch.setattr(module, "_run", fake_run)
    output_root = tmp_path / "proxy_contract"
    module._sample_eval_package(
        repo_root=Path("."),
        checkpoint_path=Path("checkpoint.pt"),
        config_path=None,
        output_root=output_root,
        alpha=1.5,
        n_samples=5,
        sample_batch_size=2,
        eval_batch_size=2,
        device="cpu",
        eval_profile="proxy",
        reference_imagefolder_root=Path("ref_images"),
        reference_tensor_file=Path("ref_tensor.pt"),
        load_reference_stats=Path("ref_stats.pt"),
        allow_reference_contract_mismatch=False,
        command_records=command_records,
        name_prefix="claim",
    )

    eval_argv = list(command_records[1].argv)  # type: ignore[attr-defined]
    assert "--inception-weights" in eval_argv
    assert _argv_value(eval_argv, "--inception-weights") == "none"
    assert _argv_value(eval_argv, "--reference-source") == "tensor_file"
    assert _argv_value(eval_argv, "--reference-tensor-file-path") == "ref_tensor.pt"
    assert _argv_value(eval_argv, "--max-reference-samples") == "2048"
    assert "--load-reference-stats" not in eval_argv
    assert "--allow-reference-contract-mismatch" not in eval_argv


def test_paper_facing_eval_pretrained_cached_contract_args(tmp_path: Path, monkeypatch) -> None:
    module = importlib.import_module("scripts.run_pixel_paper_facing_package")
    command_records: list[object] = []

    def fake_run(*, name: str, argv: list[str], cwd: Path):
        del cwd
        if "scripts/sample_pixel.py" in argv:
            sample_root = Path(_argv_value(argv, "--output-root"))
            sample_root.mkdir(parents=True, exist_ok=True)
            (sample_root / "sample_summary.json").write_text(json.dumps({"saved_images": 6}), encoding="utf-8")
        if "scripts/eval_fid_is.py" in argv:
            output_path = Path(_argv_value(argv, "--output-path"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"fid": 1.0, "inception_score_mean": 1.0, "inception_score_std": 0.0}),
                encoding="utf-8",
            )
        return module.CommandRecord(name=name, argv=argv, returncode=0, duration_seconds=0.01)

    monkeypatch.setattr(module, "_run", fake_run)
    output_root = tmp_path / "pretrained_contract"
    module._sample_eval_package(
        repo_root=Path("."),
        checkpoint_path=Path("checkpoint.pt"),
        config_path=None,
        output_root=output_root,
        alpha=1.5,
        n_samples=6,
        sample_batch_size=2,
        eval_batch_size=2,
        device="cpu",
        eval_profile="pretrained_cached",
        reference_imagefolder_root=Path("ref_images"),
        reference_tensor_file=Path("ref_tensor.pt"),
        load_reference_stats=Path("ref_stats.pt"),
        allow_reference_contract_mismatch=True,
        command_records=command_records,
        name_prefix="claim",
    )

    eval_argv = list(command_records[1].argv)  # type: ignore[attr-defined]
    assert "--inception-weights" in eval_argv
    assert _argv_value(eval_argv, "--inception-weights") == "pretrained"
    assert _argv_value(eval_argv, "--reference-source") == "imagefolder"
    assert _argv_value(eval_argv, "--reference-imagefolder-root") == "ref_images"
    assert _argv_value(eval_argv, "--load-reference-stats") == "ref_stats.pt"
    assert "--reference-tensor-file-path" not in eval_argv
    assert "--allow-reference-contract-mismatch" in eval_argv
