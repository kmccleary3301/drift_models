from pathlib import Path

from drifting_models.utils.checkpoints import list_step_checkpoints, parse_step_checkpoint, select_last_k_checkpoints


def test_parse_step_checkpoint() -> None:
    assert parse_step_checkpoint(Path("checkpoint_step_00000001.pt")).step == 1
    assert parse_step_checkpoint(Path("checkpoint_step_00000123.pt")).step == 123
    assert parse_step_checkpoint(Path("checkpoint.pt")) is None
    assert parse_step_checkpoint(Path("checkpoint_step_1.pt")) is None


def test_list_and_select(tmp_path: Path) -> None:
    for step in [3, 1, 2]:
        (tmp_path / f"checkpoint_step_{step:08d}.pt").write_bytes(b"dummy")
    (tmp_path / "other.txt").write_text("ignore", encoding="utf-8")
    items = list_step_checkpoints(tmp_path)
    assert [item.step for item in items] == [1, 2, 3]
    last = select_last_k_checkpoints(tmp_path, k=2)
    assert [item.step for item in last] == [2, 3]

