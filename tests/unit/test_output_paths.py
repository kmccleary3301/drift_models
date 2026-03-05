from __future__ import annotations

from pathlib import Path

from drifting_models.utils.output_paths import (
    build_experimental_run_root,
    build_stable_run_root,
    is_experimental_run_name,
    is_stable_run_name,
    make_experimental_run_name,
    make_stable_run_name,
    slugify_experiment_name,
)


def test_stable_name_helpers_use_expected_format() -> None:
    name = make_stable_run_name(timestamp="20260304_235959")
    assert name == "stable_20260304_235959"
    assert is_stable_run_name(name)
    assert not is_experimental_run_name(name)


def test_experimental_name_helpers_slugify_and_match_pattern() -> None:
    slug = slugify_experiment_name("Table8 Proxy/Bench")
    assert slug == "table8_proxy_bench"
    name = make_experimental_run_name(name="Table8 Proxy/Bench", timestamp="20260304_235959")
    assert name == "exp_table8_proxy_bench_20260304_235959"
    assert is_experimental_run_name(name)
    assert not is_stable_run_name(name)


def test_build_run_roots_append_names_to_base_dir() -> None:
    base = Path("/tmp/outputs/imagenet")
    stable_root = build_stable_run_root(base_dir=base, timestamp="20260304_101112")
    experimental_root = build_experimental_run_root(
        base_dir=base,
        name="queue hotpath",
        timestamp="20260304_101112",
    )
    assert stable_root == base / "stable_20260304_101112"
    assert experimental_root == base / "exp_queue_hotpath_20260304_101112"
