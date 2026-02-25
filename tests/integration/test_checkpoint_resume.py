import json
import subprocess
import sys
from pathlib import Path

import torch


def test_train_latent_checkpoint_resume_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "latent_ckpt.pt"

    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    first_payload = json.loads(first.stdout)
    assert checkpoint_path.exists()
    assert first_payload["resume_step"] == 0.0

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["resume_step"] == 2.0
    assert second_payload["logs"][0]["step"] == 3.0


def test_train_latent_resume_model_only_resets_optimizer_schedule(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "latent_ckpt_model_only.pt"

    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--learning-rate",
            "1e-5",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--allow-resume-config-mismatch",
            "--resume-model-only",
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["resume_step"] == 2.0
    assert second_payload["logs"][0]["step"] == 3.0
    assert abs(second_payload["logs"][0]["lr"] - 1e-5) < 1e-12


def test_train_latent_resume_reset_optimizer_lr_overrides_state(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "latent_ckpt_resume_lr_override.pt"

    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--learning-rate",
            "1e-4",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert checkpoint_path.exists()

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--learning-rate",
            "1e-5",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--allow-resume-config-mismatch",
            "--resume-reset-optimizer-lr",
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["resume_step"] == 2.0
    assert second_payload["logs"][0]["step"] == 3.0
    assert abs(second_payload["logs"][0]["lr"] - 1e-5) < 1e-12


def test_train_latent_queue_resume_restores_queue_state(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "latent_queue_resume.pt"

    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "1",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-queue",
            "--queue-prime-samples",
            "16",
            "--queue-push-batch",
            "8",
            "--queue-per-class-capacity",
            "16",
            "--queue-global-capacity",
            "64",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(checkpoint_payload.get("queue_state"), dict)
    assert checkpoint_payload["queue_state"]["global_labels"] is not None

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_latent.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "1",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-queue",
            "--queue-prime-samples",
            "16",
            "--queue-push-batch",
            "8",
            "--queue-per-class-capacity",
            "16",
            "--queue-global-capacity",
            "64",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["resume_step"] == 2.0
    assert second_payload["train_config"]["queue_resumed_from_checkpoint"] is True
    assert second_payload["queue_warmup_report"]["mode"] == "resume_restore"


def test_train_pixel_queue_resume_restores_queue_state(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_path = tmp_path / "pixel_queue_resume.pt"

    first = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "1",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-queue",
            "--queue-prime-samples",
            "16",
            "--queue-push-batch",
            "8",
            "--queue-per-class-capacity",
            "16",
            "--queue-global-capacity",
            "64",
            "--checkpoint-path",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(checkpoint_payload.get("queue_state"), dict)
    assert checkpoint_payload["queue_state"]["global_labels"] is not None

    second = subprocess.run(
        [
            sys.executable,
            "scripts/train_pixel.py",
            "--device",
            "cpu",
            "--steps",
            "3",
            "--log-every",
            "1",
            "--groups",
            "2",
            "--negatives-per-group",
            "2",
            "--positives-per-group",
            "2",
            "--unconditional-per-group",
            "1",
            "--image-size",
            "16",
            "--patch-size",
            "4",
            "--hidden-dim",
            "64",
            "--depth",
            "2",
            "--num-heads",
            "4",
            "--use-queue",
            "--queue-prime-samples",
            "16",
            "--queue-push-batch",
            "8",
            "--queue-per-class-capacity",
            "16",
            "--queue-global-capacity",
            "64",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from",
            str(checkpoint_path),
            "--save-every",
            "1",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["resume_step"] == 2.0
    assert second_payload["train_config"]["queue_resumed_from_checkpoint"] is True
    assert second_payload["queue_warmup_report"]["mode"] == "resume_restore"
