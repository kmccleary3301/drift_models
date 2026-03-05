from __future__ import annotations

from pathlib import Path
from typing import Any
import sys


def forward_target(
    *,
    target_path: Path,
    namespace: dict[str, Any],
    run_main: bool,
    deprecated_entrypoint: str | None = None,
    migration_target: str | None = None,
    removal_version: str | None = None,
) -> None:
    if not target_path.exists():
        raise FileNotFoundError(f"Missing forwarded target: {target_path}")
    if run_main and deprecated_entrypoint is not None and migration_target is not None:
        target_removal = removal_version if removal_version is not None else "a future release"
        print(
            f"[DEPRECATED] `{deprecated_entrypoint}` is deprecated and will be removed in {target_removal}. "
            f"Use `{migration_target}` instead.",
            file=sys.stderr,
        )

    original_name = str(namespace.get("__name__", ""))
    original_file = namespace.get("__file__", None)
    namespace["__name__"] = original_name
    namespace["__file__"] = str(target_path)

    source = target_path.read_text(encoding="utf-8")
    code = compile(source, str(target_path), "exec")
    exec(code, namespace)

    namespace["__name__"] = original_name
    if original_file is not None:
        namespace["__file__"] = original_file
