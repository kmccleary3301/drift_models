from __future__ import annotations

from pathlib import Path

try:
    from ._legacy_forward import forward_target
except ImportError:  # pragma: no cover
    from _legacy_forward import forward_target

_TARGET_PATH = Path(__file__).resolve().parent / "experimental" / "pipelines" / "pixel_mae_export.py"

forward_target(
    target_path=_TARGET_PATH,
    namespace=globals(),
    run_main=__name__ == "__main__",
    deprecated_entrypoint="scripts/run_pixel_mae_export_pipeline.py",
    migration_target="scripts/experimental/pipelines/pixel_mae_export.py",
    removal_version="v0.3.0",
)
