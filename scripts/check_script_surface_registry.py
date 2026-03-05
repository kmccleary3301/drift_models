from __future__ import annotations

import argparse
import json
from pathlib import Path

from drifting_models.utils import discover_repo_root


_CATEGORIES: tuple[str, ...] = (
    "stable_entrypoints",
    "stable_support",
    "maintenance_wrappers",
    "deprecated_wrappers",
    "legacy_misc",
    "experimental_top_level",
    "internal",
)

_STABLE_PREFIXES: tuple[str, ...] = ("train_", "sample_", "eval_", "runtime_")
_EXPERIMENTAL_PREFIXES: tuple[str, ...] = ("exp_", "ablation_")


def main() -> None:
    args = _parse_args()
    repo_root = discover_repo_root(Path(__file__))
    registry_path = (repo_root / args.registry_path).resolve()
    scripts_dir = (repo_root / "scripts").resolve()
    failures: list[str] = []

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing registry file: {registry_path}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _validate_registry_shape(registry=registry, failures=failures)

    registered = _flatten_registry(registry=registry)
    duplicates = _find_duplicates(registry=registry)
    if duplicates:
        failures.append(f"duplicate registry entries: {', '.join(sorted(duplicates))}")

    discovered = sorted(path.name for path in scripts_dir.glob("*.py"))
    missing = sorted(set(discovered) - registered)
    extra = sorted(registered - set(discovered))
    if missing:
        failures.append(f"unregistered top-level scripts: {', '.join(missing)}")
    if extra:
        failures.append(f"registry entries missing on disk: {', '.join(extra)}")

    for name in registry["stable_entrypoints"]:
        if not _matches_any_prefix(name=name, prefixes=_STABLE_PREFIXES):
            failures.append(
                f"{name}: stable entrypoint must use one of prefixes {', '.join(_STABLE_PREFIXES)}"
            )
    for name in registry["experimental_top_level"]:
        if not _matches_any_prefix(name=name, prefixes=_EXPERIMENTAL_PREFIXES):
            failures.append(
                f"{name}: experimental top-level script must use one of prefixes {', '.join(_EXPERIMENTAL_PREFIXES)}"
            )

    for name in [*registry["maintenance_wrappers"], *registry["deprecated_wrappers"]]:
        script_path = scripts_dir / name
        if not script_path.exists():
            continue
        text = script_path.read_text(encoding="utf-8")
        if "_TARGET_PATH" not in text or "forward_target(" not in text:
            failures.append(f"{name}: wrapper entry must define _TARGET_PATH and call forward_target()")

    _validate_docs_alignment(repo_root=repo_root, registry=registry, failures=failures)

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "ok",
                "registry_path": str(registry_path),
                "top_level_scripts": len(discovered),
                "categories": {category: len(registry[category]) for category in _CATEGORIES},
            },
            indent=2,
        )
    )


def _validate_docs_alignment(*, repo_root: Path, registry: dict[str, list[str]], failures: list[str]) -> None:
    scripts_readme = (repo_root / "scripts/README.md").read_text(encoding="utf-8")
    deprecation_matrix = (repo_root / "docs/deprecation_matrix.md").read_text(encoding="utf-8")

    for name in registry["stable_entrypoints"]:
        if f"`{name}`" not in scripts_readme:
            failures.append(f"scripts/README.md missing stable entrypoint: {name}")
    for name in registry["stable_support"]:
        if f"`{name}`" not in scripts_readme:
            failures.append(f"scripts/README.md missing stable support script: {name}")

    for name in registry["deprecated_wrappers"]:
        full_path = f"`scripts/{name}`"
        if full_path not in deprecation_matrix:
            failures.append(f"docs/deprecation_matrix.md missing deprecated wrapper row: scripts/{name}")


def _matches_any_prefix(*, name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _validate_registry_shape(*, registry: dict[str, object], failures: list[str]) -> None:
    for category in _CATEGORIES:
        if category not in registry:
            failures.append(f"missing category: {category}")
            continue
        value = registry[category]
        if not isinstance(value, list):
            failures.append(f"{category}: expected list, got {type(value).__name__}")
            continue
        for name in value:
            if not isinstance(name, str):
                failures.append(f"{category}: non-string entry {name!r}")
            elif not name.endswith(".py"):
                failures.append(f"{category}: non-python entry {name!r}")


def _flatten_registry(*, registry: dict[str, list[str]]) -> set[str]:
    flattened: set[str] = set()
    for category in _CATEGORIES:
        flattened.update(registry.get(category, []))
    return flattened


def _find_duplicates(*, registry: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for category in _CATEGORIES:
        for name in registry.get(category, []):
            if name in seen:
                duplicates.add(name)
            else:
                seen.add(name)
    return duplicates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate top-level script taxonomy registration and naming policy.")
    parser.add_argument("--registry-path", type=str, default="scripts/script_surface_registry.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
