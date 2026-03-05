from __future__ import annotations

from pathlib import Path

try:
    from ._legacy_forward import forward_target
except ImportError:  # pragma: no cover
    from _legacy_forward import forward_target

_TARGET_PATH = Path(__file__).resolve().parent / "experimental" / "checks" / "feature_x2_toggle_effect.py"

forward_target(
    target_path=_TARGET_PATH,
    namespace=globals(),
    run_main=__name__ == "__main__",
    deprecated_entrypoint="scripts/check_feature_x2_toggle_effect.py",
    migration_target="scripts/experimental/checks/feature_x2_toggle_effect.py",
    removal_version="v0.3.0",
)
