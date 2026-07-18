"""Public package API for the tensor-valued anomaly detector."""

__version__ = "0.1.1"

from .Data_Loading import (
    PatchExtractionConfig,
    extract_cnn_feature_maps,
    load_patch_datasets,
    load_normal_patches,
    make_cnn_feature_extractor,
)
from .location_aware_tensor_mahalanobis_detector import (
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
)
from .patch_estimators import TensorGaussianState

__all__ = [
    "__version__",
    "PatchExtractionConfig",
    "extract_cnn_feature_maps",
    "make_cnn_feature_extractor",
    "load_patch_datasets",
    "load_normal_patches",
    "LocationAwareTensorMahalanobisDetector",
    "NeighborhoodScoreLocationAwareTensorMahalanobisDetector",
    "TensorGaussianState",
]
