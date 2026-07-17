"""Public package API for the tensor-valued anomaly detector."""

__version__ = "0.1.1"

from .Data_Loading import PatchExtractionConfig, load_patch_datasets
from .location_aware_tensor_mahalanobis_detector import (
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
)
from .patch_estimators import TensorGaussianState

__all__ = [
    "__version__",
    "PatchExtractionConfig",
    "load_patch_datasets",
    "LocationAwareTensorMahalanobisDetector",
    "NeighborhoodScoreLocationAwareTensorMahalanobisDetector",
    "TensorGaussianState",
]
