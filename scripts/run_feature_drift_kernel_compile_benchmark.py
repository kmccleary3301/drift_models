from __future__ import annotations

from pathlib import Path

try:
    from ._legacy_forward import forward_target
except ImportError:  # pragma: no cover
    from _legacy_forward import forward_target

_TARGET_PATH = Path(__file__).resolve().parent / "experimental" / "benchmarks" / "feature_drift_kernel_compile.py"

forward_target(target_path=_TARGET_PATH, namespace=globals(), run_main=__name__ == "__main__")
