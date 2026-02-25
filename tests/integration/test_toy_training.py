from pathlib import Path

import torch

from drifting_models.train import (
    AblationConfig,
    parse_simple_yaml_config,
    run_toy_suite,
)


def test_toy_baseline_beats_attraction_only(tmp_path: Path) -> None:
    config = parse_simple_yaml_config(Path("configs/toy/quick.yaml"))
    summary = run_toy_suite(
        config=config,
        ablations=[
            AblationConfig(name="baseline", attraction_scale=1.0, repulsion_scale=1.0),
            AblationConfig(name="attraction_only", attraction_scale=1.0, repulsion_scale=0.0),
        ],
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    by_name = {entry["ablation"]: entry for entry in summary["results"]}
    baseline_distance = by_name["baseline"]["final_metrics"]["mean_distance_to_target"]
    attraction_only_distance = by_name["attraction_only"]["final_metrics"]["mean_distance_to_target"]
    assert baseline_distance < attraction_only_distance
    assert (tmp_path / "toy_results.json").exists()
    assert (tmp_path / "toy_ablation_table.md").exists()
    baseline_dir = tmp_path / "baseline"
    assert baseline_dir.exists()
    assert any(path.suffix == ".pt" for path in baseline_dir.iterdir())
    assert any(path.suffix == ".png" for path in baseline_dir.iterdir())


def test_toy_baseline_training_improves_loss() -> None:
    config = parse_simple_yaml_config(Path("configs/toy/quick.yaml"))
    summary = run_toy_suite(
        config=config,
        ablations=[AblationConfig(name="baseline", attraction_scale=1.0, repulsion_scale=1.0)],
        output_dir=None,
        device=torch.device("cpu"),
    )
    history = summary["results"][0]["history"]
    assert len(history) >= 2
    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    first_drift = history[0]["drift_norm"]
    last_drift = history[-1]["drift_norm"]
    assert last_loss < first_loss
    assert last_drift < first_drift


def test_toy_replay_is_deterministic_for_fixed_seed() -> None:
    config = parse_simple_yaml_config(Path("configs/toy/quick.yaml"))
    ablations = [AblationConfig(name="baseline", attraction_scale=1.0, repulsion_scale=1.0)]
    summary_a = run_toy_suite(
        config=config,
        ablations=ablations,
        output_dir=None,
        device=torch.device("cpu"),
    )
    summary_b = run_toy_suite(
        config=config,
        ablations=ablations,
        output_dir=None,
        device=torch.device("cpu"),
    )
    history_a = summary_a["results"][0]["history"]
    history_b = summary_b["results"][0]["history"]
    assert history_a == history_b
    metrics_a = summary_a["results"][0]["final_metrics"]
    metrics_b = summary_b["results"][0]["final_metrics"]
    assert metrics_a == metrics_b
