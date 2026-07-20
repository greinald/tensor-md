"""Public package API for the tensor-valued anomaly detector."""

__version__ = "0.1.5"

from .Data_Loading import (
    PatchExtractionConfig,
    extract_cnn_feature_maps,
    dark_foreground_orientation_context,
    light_background_orientation_context,
    load_patch_datasets,
    load_normal_patches,
    load_image_patches,
    make_cnn_feature_extractor,
)
from .location_aware_tensor_mahalanobis_detector import (
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
)
from .diagnostics import save_score_diagnostics
from .patch_estimators import TensorGaussianState

__all__ = [
    "__version__",
    "PatchExtractionConfig",
    "extract_cnn_feature_maps",
    "dark_foreground_orientation_context",
    "light_background_orientation_context",
    "make_cnn_feature_extractor",
    "save_score_diagnostics",
    "load_patch_datasets",
    "load_normal_patches",
    "load_image_patches",
    "LocationAwareTensorMahalanobisDetector",
    "NeighborhoodScoreLocationAwareTensorMahalanobisDetector",
    "TensorGaussianState",
]
