from __future__ import annotations

import time

import numpy as np
from sklearn.covariance import LedoitWolf


class VectorizedMahalanobisDetector:
    """Gaussian patch detector after vectorizing tensor observations."""

    def __init__(self) -> None:
        # Ledoit-Wolf gives a shrinkage covariance estimate, which is much more
        # stable than a raw empirical covariance in high dimension.
        self.estimator = LedoitWolf(assume_centered=False)
        # Filled after fit()/score() for timing inspection in notebooks.
        self.fit_timing: dict[str, float] = {}
        self.score_timing: dict[str, float] = {}

    def fit(self, patches: np.ndarray) -> "VectorizedMahalanobisDetector":
        fit_start = time.perf_counter()
        # Collapse each tensor patch into one long feature vector.
        # A (16, 16, 3) patch becomes a vector of length 768.
        reshape_start = time.perf_counter()
        flat_patches = patches.reshape(patches.shape[0], -1)
        reshape_seconds = time.perf_counter() - reshape_start

        estimator_start = time.perf_counter()
        self.estimator.fit(flat_patches)
        estimator_seconds = time.perf_counter() - estimator_start
        self.fit_timing = {
            "reshape_seconds": reshape_seconds,
            "estimator_fit_seconds": estimator_seconds,
            "total_seconds": time.perf_counter() - fit_start,
        }
        return self

    def score(self, patches: np.ndarray) -> np.ndarray:
        score_start = time.perf_counter()
        # Use the fitted Gaussian model to return squared Mahalanobis distance
        # for each test patch. Larger means more anomalous.
        # Internally this is the standard vector formula
        # (x - mu)^T Sigma^(-1) (x - mu),
        # but sklearn handles the covariance inversion details for us.
        reshape_start = time.perf_counter()
        flat_patches = patches.reshape(patches.shape[0], -1)
        reshape_seconds = time.perf_counter() - reshape_start

        estimator_start = time.perf_counter()
        scores = self.estimator.mahalanobis(flat_patches)
        estimator_seconds = time.perf_counter() - estimator_start
        self.score_timing = {
            "reshape_seconds": reshape_seconds,
            "estimator_score_seconds": estimator_seconds,
            "total_seconds": time.perf_counter() - score_start,
        }
        return scores
