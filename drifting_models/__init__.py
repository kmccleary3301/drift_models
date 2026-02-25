from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from drifting_models.drift_field import (
    DriftFieldConfig,
    build_negative_log_weights,
    cfg_alpha_to_unconditional_weight,
    compute_affinity_matrices,
    compute_drift_components,
    compute_v,
)
from drifting_models.drift_loss import (
    DriftingLossConfig,
    FeatureDriftingConfig,
    feature_space_drifting_loss,
)

try:
    __version__ = package_version("drift-models")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "DriftFieldConfig",
    "DriftingLossConfig",
    "FeatureDriftingConfig",
    "build_negative_log_weights",
    "cfg_alpha_to_unconditional_weight",
    "compute_affinity_matrices",
    "compute_drift_components",
    "compute_v",
    "feature_space_drifting_loss",
]
