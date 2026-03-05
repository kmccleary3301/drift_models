from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tier metadata on config files.")
    parser.add_argument("--configs-root", type=str, default="configs")
    args = parser.parse_args()

    root = Path(args.configs_root).resolve()
    config_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    )

    failures: list[str] = []
    for path in config_paths:
        entries = _load_raw_simple_kv(path)
        tier = entries.get("tier")
        rel = path.relative_to(root)
        if tier is None:
            failures.append(f"{rel}: missing tier")
            continue
        if tier not in {"stable", "experimental"}:
            failures.append(f"{rel}: invalid tier={tier!r} (expected stable|experimental)")
            continue
        first = rel.parts[0] if rel.parts else ""
        if first == "stable" and tier != "stable":
            failures.append(f"{rel}: configs/stable/* must declare tier: stable")
        if first == "experimental" and tier != "experimental":
            failures.append(f"{rel}: configs/experimental/* must declare tier: experimental")

    if failures:
        print("Config tier check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Config tier check passed ({len(config_paths)} files).")


def _load_raw_simple_kv(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line in {path}: {raw_line}")
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


if __name__ == "__main__":
    main()
