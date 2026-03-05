from __future__ import annotations

from pathlib import Path


_INCLUDE_KEYS: frozenset[str] = frozenset({"include", "extends", "base-config"})
_METADATA_KEYS: frozenset[str] = frozenset({"tier", "canonical-config", "notes"})


def load_simple_kv_config(
    path: Path,
    *,
    allow_includes: bool = True,
    strip_metadata: bool = True,
    _visited: set[Path] | None = None,
) -> dict[str, str]:
    resolved_path = path.resolve()
    visited = set() if _visited is None else _visited
    if resolved_path in visited:
        raise ValueError(f"Config include cycle detected at {resolved_path}")
    visited.add(resolved_path)

    include_paths: list[Path] = []
    local_entries: dict[str, str] = {}
    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in _INCLUDE_KEYS:
            if allow_includes:
                include_paths.append(_resolve_include_path(base_path=resolved_path, include_value=value))
            continue
        local_entries[key] = value

    merged: dict[str, str] = {}
    for include_path in include_paths:
        included_entries = load_simple_kv_config(
            include_path,
            allow_includes=allow_includes,
            strip_metadata=False,
            _visited=visited,
        )
        merged.update(included_entries)
    merged.update(local_entries)

    if strip_metadata:
        for key in _METADATA_KEYS:
            merged.pop(key, None)
    return merged


def _resolve_include_path(*, base_path: Path, include_value: str) -> Path:
    raw = include_value.strip().strip("'").strip('"')
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base_path.parent / candidate).resolve()
    return candidate
