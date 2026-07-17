from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np

try:
    from .patch_estimators import (
        EPS,
        TensorGaussianState,
        _blend_tensor_separable_covariances,
        _fit_tensor_separable_model_from_centered,
        _fit_tensor_separable_model_from_centered_batches,
        _matrix_inverse_square_root,
        _normalize_trace,
        _regularize_covariance,
        _score_tensor_separable_model,
    )
except ImportError:
    from patch_estimators import (
        EPS,
        TensorGaussianState,
        _blend_tensor_separable_covariances,
        _fit_tensor_separable_model_from_centered,
        _fit_tensor_separable_model_from_centered_batches,
        _matrix_inverse_square_root,
        _normalize_trace,
        _regularize_covariance,
        _score_tensor_separable_model,
    )


def location_neighbors(grid_h: int, grid_w: int, radius: int) -> list[np.ndarray]:
    """Return flattened location indices inside each location's square neighborhood."""

    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")

    neighbors = []
    for row in range(grid_h):
        for col in range(grid_w):
            row_lo = max(0, row - radius)
            row_hi = min(grid_h, row + radius + 1)
            col_lo = max(0, col - radius)
            col_hi = min(grid_w, col + radius + 1)
            neighbors.append(
                np.asarray(
                    [
                        r * grid_w + c
                        for r in range(row_lo, row_hi)
                        for c in range(col_lo, col_hi)
                    ],
                    dtype=np.int64,
                )
            )
    return neighbors


def location_neighbor_bilateral_weights(
    grid_h: int,
    grid_w: int,
    location_means: np.ndarray,
    radius: int,
    sigma_spatial: float | None = None,
    sigma_range: float | None = None,
) -> list[np.ndarray]:
    """Return bilateral spatial-feature weights aligned with location_neighbors()."""

    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    sigma_spatial = float(max(radius, 1) if sigma_spatial is None else sigma_spatial)
    sigma_range = float(sigma_spatial if sigma_range is None else sigma_range)
    if sigma_spatial <= 0.0:
        raise ValueError(f"sigma_spatial must be positive, got {sigma_spatial}.")
    if sigma_range <= 0.0:
        raise ValueError(f"sigma_range must be positive, got {sigma_range}.")
    if location_means.shape[0] != grid_h * grid_w:
        raise ValueError(
            "location_means count does not match grid size: "
            f"{location_means.shape[0]} vs {grid_h * grid_w}."
        )

    flattened_means = location_means.reshape(location_means.shape[0], -1).astype(np.float64, copy=False)
    weights_by_location = []
    for row in range(grid_h):
        for col in range(grid_w):
            location_index = row * grid_w + col
            row_lo = max(0, row - radius)
            row_hi = min(grid_h, row + radius + 1)
            col_lo = max(0, col - radius)
            col_hi = min(grid_w, col + radius + 1)
            target_mean = flattened_means[location_index]
            weights = []
            for r in range(row_lo, row_hi):
                for c in range(col_lo, col_hi):
                    neighbor_index = r * grid_w + c
                    neighbor_mean = flattened_means[neighbor_index]
                    spatial_distance_sq = float((r - row) ** 2 + (c - col) ** 2)
                    feature_delta = target_mean - neighbor_mean
                    feature_distance_sq = float(feature_delta @ feature_delta)
                    spatial_weight = np.exp(-spatial_distance_sq / (2.0 * sigma_spatial * sigma_spatial))
                    range_weight = np.exp(-feature_distance_sq / (2.0 * sigma_range * sigma_range))
                    weights.append(spatial_weight * range_weight)

            weights = np.asarray(weights, dtype=np.float64)
            weight_sum = float(weights.sum())
            if weight_sum > 0.0:
                weights = weights / weight_sum
            else:
                weights.fill(1.0 / max(len(weights), 1))
            weights_by_location.append(weights.astype(np.float32))
    return weights_by_location


def aggregate_location_scores(
    scores_by_image: np.ndarray,
    neighbor_indices_by_location: list[np.ndarray],
    pooling: str,
    neighbor_weights_by_location: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Pool already-computed location scores over spatial neighborhoods."""

    if pooling not in {"mean", "max", "weighted_mean"}:
        raise ValueError(
            "pooling must be 'mean', 'max', or 'weighted_mean', "
            f"got {pooling!r}."
        )
    if scores_by_image.shape[1] != len(neighbor_indices_by_location):
        raise ValueError(
            "scores_by_image location dimension does not match neighbor list: "
            f"{scores_by_image.shape[1]} vs {len(neighbor_indices_by_location)}."
        )
    if pooling == "weighted_mean":
        if neighbor_weights_by_location is None:
            raise ValueError("neighbor_weights_by_location is required for weighted_mean pooling.")
        if len(neighbor_weights_by_location) != len(neighbor_indices_by_location):
            raise ValueError(
                "neighbor weight count does not match neighbor list count: "
                f"{len(neighbor_weights_by_location)} vs {len(neighbor_indices_by_location)}."
            )

    pooled = np.empty_like(scores_by_image)
    for location_index, neighbor_indices in enumerate(neighbor_indices_by_location):
        neighbor_scores = scores_by_image[:, neighbor_indices]
        if pooling == "mean":
            pooled[:, location_index] = neighbor_scores.mean(axis=1)
        elif pooling == "max":
            pooled[:, location_index] = neighbor_scores.max(axis=1)
        else:
            weights = neighbor_weights_by_location[location_index]
            if len(weights) != len(neighbor_indices):
                raise ValueError(
                    f"Location {location_index} has {len(neighbor_indices)} neighbors "
                    f"but {len(weights)} weights."
                )
            pooled[:, location_index] = neighbor_scores @ weights
    return pooled


def aggregate_regular_grid_scores(
    scores_by_image: np.ndarray,
    grid_shape: tuple[int, int],
    radius: int,
    pooling: str,
    sigma: float | None = None,
) -> np.ndarray:
    """Vectorized neighborhood pooling for a regular spatial score grid."""

    scores_by_image = np.asarray(scores_by_image)
    grid_h, grid_w = grid_shape
    if scores_by_image.ndim != 2 or scores_by_image.shape[1] != grid_h * grid_w:
        raise ValueError(
            "scores_by_image must have one column per grid location: "
            f"got {scores_by_image.shape} for grid {grid_shape}."
        )
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}.")
    if pooling not in {"mean", "max", "weighted_mean"}:
        raise ValueError(
            "pooling must be 'mean', 'max', or 'weighted_mean', "
            f"got {pooling!r}."
        )
    if radius == 0:
        return scores_by_image.copy()

    grid = scores_by_image.reshape(-1, grid_h, grid_w)
    window_size = 2 * radius + 1
    if pooling == "max":
        padded = np.pad(
            grid,
            ((0, 0), (radius, radius), (radius, radius)),
            mode="constant",
            constant_values=-np.inf,
        )
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (window_size, window_size),
            axis=(1, 2),
        )
        return windows.max(axis=(-2, -1)).reshape(scores_by_image.shape)

    if pooling == "mean":
        kernel = np.ones((window_size, window_size), dtype=np.float64)
    else:
        sigma = float(max(radius, 1) if sigma is None else sigma)
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        distances_sq = offsets[:, None] ** 2 + offsets[None, :] ** 2
        kernel = np.exp(-distances_sq / (2.0 * sigma * sigma))

    padded = np.pad(
        grid,
        ((0, 0), (radius, radius), (radius, radius)),
        mode="constant",
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (window_size, window_size),
        axis=(1, 2),
    )
    numerator = np.einsum("nhwij,ij->nhw", windows, kernel, optimize=True)

    valid = np.pad(
        np.ones((grid_h, grid_w), dtype=np.float64),
        ((radius, radius), (radius, radius)),
        mode="constant",
    )
    valid_windows = np.lib.stride_tricks.sliding_window_view(
        valid,
        (window_size, window_size),
    )
    denominator = np.einsum("hwij,ij->hw", valid_windows, kernel, optimize=True)
    pooled = numerator / denominator[None, :, :]
    return pooled.astype(scores_by_image.dtype, copy=False).reshape(scores_by_image.shape)


class LocationAwareTensorMahalanobisDetector:
    """Location-aware tensor Mahalanobis detector with one mean and covariance per location."""

    def __init__(
        self,
        patches_per_image: int,
        iterations: int = 6,
        eps: float = EPS,
        convergence_tol: float = 1e-4,
        mean_shrinkage: float = 0.0,
        covariance_shrinkage: float = 0.0,
        score_normalization: str = "none",
        score_normalization_eps: float = 1e-8,
        verbose: bool = False,
        shared_score_location_batch_size: int = 32,
        location_fit_workers: int | None = None,
    ) -> None:
        self.iterations = iterations
        self.eps = eps
        self.convergence_tol = convergence_tol
        self.mean_shrinkage = mean_shrinkage
        self.covariance_shrinkage = covariance_shrinkage
        self.score_normalization = score_normalization
        self.score_normalization_eps = score_normalization_eps
        self.verbose = verbose
        self.shared_score_location_batch_size = shared_score_location_batch_size
        self.location_fit_workers = (
            min(2, os.cpu_count() or 1)
            if location_fit_workers is None
            else location_fit_workers
        )
        # One mean patch per spatial patch location.
        self.location_means: np.ndarray | None = None
        # One covariance model per spatial patch location.
        self.location_covariance_states: list[TensorGaussianState] | None = None
        # Backward-compatible summary over all location covariance states.
        self.location_aware_covariance_state: TensorGaussianState | None = None
        # Backward-compatible alias for older call sites.
        self.covariance_state: TensorGaussianState | None = None
        # Optional global covariance fit used only as a shrinkage target.
        self.global_covariance_state: TensorGaussianState | None = None
        # Optional per-location train-score statistics for score calibration.
        self.location_score_statistics: dict[str, np.ndarray] | None = None
        # Every image contributes this many local patch positions.
        self.patches_per_image = patches_per_image
        self.fit_timing: dict[str, float | int | bool | None] = {}
        self.score_timing: dict[str, float] = {}

        if not isinstance(self.patches_per_image, (int, np.integer)) or self.patches_per_image <= 0:
            raise ValueError(
                "patches_per_image must be a positive integer, "
                f"got {self.patches_per_image!r}."
            )
        if not isinstance(self.iterations, (int, np.integer)) or self.iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {self.iterations!r}.")
        if (
            not isinstance(self.shared_score_location_batch_size, (int, np.integer))
            or self.shared_score_location_batch_size <= 0
        ):
            raise ValueError(
                "shared_score_location_batch_size must be a positive integer, "
                f"got {self.shared_score_location_batch_size!r}."
            )
        if (
            not isinstance(self.location_fit_workers, (int, np.integer))
            or self.location_fit_workers <= 0
        ):
            raise ValueError(
                "location_fit_workers must be a positive integer or None, "
                f"got {self.location_fit_workers!r}."
            )
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {self.eps}.")
        if not np.isfinite(self.convergence_tol) or self.convergence_tol < 0.0:
            raise ValueError(
                "convergence_tol must be finite and non-negative, "
                f"got {self.convergence_tol}."
            )

    def _validate_mean_shrinkage(self) -> None:
        if not 0.0 <= self.mean_shrinkage <= 1.0:
            raise ValueError(
                "mean_shrinkage must lie in [0, 1], "
                f"got {self.mean_shrinkage}."
            )

    def _validate_shrinkage(self) -> None:
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError(
                "covariance_shrinkage must lie in [0, 1], "
                f"got {self.covariance_shrinkage}."
            )

    def _validate_score_normalization(self) -> None:
        if self.score_normalization not in {"none", "zscore"}:
            raise ValueError(
                "score_normalization must be 'none' or 'zscore', "
                f"got {self.score_normalization!r}."
            )
        if self.score_normalization_eps <= 0.0:
            raise ValueError(
                "score_normalization_eps must be positive, "
                f"got {self.score_normalization_eps}."
            )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LocationAwareTensorMahalanobisDetector] {message}")

    def _log_batch_progress(
        self,
        phase: str,
        batch_index: int,
        batch_count: int | None,
        iteration: int | None = None,
        dimension: int | None = None,
        location_index: int | None = None,
    ) -> None:
        if not self.verbose:
            return
        location_suffix = (
            ""
            if location_index is None
            else f" loc={location_index + 1}/{self.patches_per_image}"
        )
        if iteration is None:
            print(
                f"[LocationAwareTensorMahalanobisDetector] {phase}{location_suffix} "
                f"batch {batch_index}/{batch_count if batch_count is not None else '?'}"
            )
            return
        mode_names = ("height", "width", "channel", "layer")
        dimension_name = (
            mode_names[dimension] if dimension is not None and dimension < len(mode_names)
            else f"mode-{dimension}"
        )
        print(
            f"[LocationAwareTensorMahalanobisDetector] {phase}{location_suffix} "
            f"iter={iteration} dim={dimension_name} "
            f"batch {batch_index}/{batch_count if batch_count is not None else '?'}"
        )

    def _reshape_by_image(self, patches: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches)
        if patches.ndim < 1:
            raise ValueError(f"patches must have a sample axis, got shape {patches.shape}.")
        if len(patches) == 0:
            raise ValueError("patches must not be empty.")
        if len(patches) % self.patches_per_image != 0:
            raise ValueError(
                f"Expected patch count to be a multiple of {self.patches_per_image}, "
                f"but got {len(patches)}."
            )
        n_images = len(patches) // self.patches_per_image
        return patches.reshape(n_images, self.patches_per_image, *patches.shape[1:])

    def _fit_global_covariance_from_array(
        self,
        residuals_by_image: np.ndarray,
        sample_shape: tuple[int, ...],
    ) -> tuple[TensorGaussianState | None, float]:
        if self.covariance_shrinkage <= 0.0:
            return None, 0.0

        global_start = time.perf_counter()
        global_residuals = residuals_by_image.reshape(-1, *sample_shape)
        global_mean = np.zeros(sample_shape, dtype=np.float32)
        global_center_start = time.perf_counter()
        global_centered = global_residuals
        global_center_seconds = time.perf_counter() - global_center_start
        state = _fit_tensor_separable_model_from_centered(
            centered=global_centered,
            mean=global_mean,
            mean_seconds=0.0,
            center_seconds=global_center_seconds,
            iterations=self.iterations,
            eps=self.eps,
            convergence_tol=self.convergence_tol,
            fit_start=global_start,
        )
        return state, time.perf_counter() - global_start

    def _shrink_location_means(self, location_means: np.ndarray) -> np.ndarray:
        """Blend per-location means toward a single global mean."""

        if self.mean_shrinkage == 0.0:
            return location_means.astype(np.float32, copy=False)

        global_mean = location_means.mean(axis=0, keepdims=True)
        blended = (
            (1.0 - self.mean_shrinkage) * location_means
            + self.mean_shrinkage * global_mean
        )
        return blended.astype(np.float32, copy=False)

    def _fit_location_covariance_states_from_array(
        self,
        residuals_by_image: np.ndarray,
        sample_shape: tuple[int, ...],
    ) -> tuple[list[TensorGaussianState], float, float]:
        zero_residual_mean = np.zeros(sample_shape, dtype=np.float32)
        covariance_seconds = 0.0
        shrinkage_seconds = 0.0
        states: list[TensorGaussianState] = []

        if self.covariance_shrinkage == 1.0:
            if self.global_covariance_state is None:
                raise RuntimeError("Global covariance state is missing for shrinkage.")
            for _ in range(self.patches_per_image):
                states.append(dict(self.global_covariance_state))
            return states, covariance_seconds, shrinkage_seconds

        def fit_location(
            location_index: int,
        ) -> tuple[TensorGaussianState, float]:
            self._log(f"fitting covariance for location {location_index + 1}/{self.patches_per_image}")
            covariance_start = time.perf_counter()
            state = _fit_tensor_separable_model_from_centered(
                centered=residuals_by_image[:, location_index, ...],
                mean=zero_residual_mean,
                mean_seconds=0.0,
                center_seconds=0.0,
                iterations=self.iterations,
                eps=self.eps,
                convergence_tol=self.convergence_tol,
                fit_start=covariance_start,
            )
            location_shrinkage_seconds = 0.0

            if self.covariance_shrinkage > 0.0:
                if self.global_covariance_state is None:
                    raise RuntimeError("Global covariance state is missing for shrinkage.")
                shrinkage_start = time.perf_counter()
                state = _blend_tensor_separable_covariances(
                    base_state=state,
                    bleed_state=self.global_covariance_state,
                    shrinkage=self.covariance_shrinkage,
                    eps=self.eps,
                )
                location_shrinkage_seconds = time.perf_counter() - shrinkage_start
            return state, location_shrinkage_seconds

        covariance_start = time.perf_counter()
        worker_count = min(self.location_fit_workers, self.patches_per_image)
        if worker_count == 1:
            fitted = map(fit_location, range(self.patches_per_image))
            for state, location_shrinkage_seconds in fitted:
                states.append(state)
                shrinkage_seconds += location_shrinkage_seconds
        else:
            # Threads share the large residual tensor without serialization or
            # process-level copies. NumPy releases the GIL for the matrix-heavy
            # work, while the worker cap avoids excessive BLAS oversubscription.
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                fitted = executor.map(fit_location, range(self.patches_per_image))
                for state, location_shrinkage_seconds in fitted:
                    states.append(state)
                    shrinkage_seconds += location_shrinkage_seconds
        covariance_seconds = time.perf_counter() - covariance_start

        return states, covariance_seconds, shrinkage_seconds

    def _fit_global_covariance_from_batches(
        self,
        patch_batch_factory,
        sample_shape: tuple[int, ...],
        observations_count: int,
        batch_count: int,
    ) -> tuple[TensorGaussianState | None, float]:
        if self.covariance_shrinkage <= 0.0:
            return None, 0.0
        if self.location_means is None:
            raise RuntimeError("location_means must be available before global fit.")

        global_start = time.perf_counter()
        global_mean = np.zeros(sample_shape, dtype=np.float32)

        def global_centered_batches():
            for patches_by_image in patch_batch_factory():
                residuals_by_image = patches_by_image - self.location_means[None, ...]
                yield residuals_by_image.reshape(-1, *sample_shape)

        state = _fit_tensor_separable_model_from_centered_batches(
            centered_batch_factory=global_centered_batches,
            sample_shape=sample_shape,
            mean=global_mean,
            observations_count=observations_count,
            batch_count=batch_count,
            batch_progress_callback=lambda iteration, dimension, batch_index, batch_count: (
                self._log_batch_progress(
                    "global-covariance",
                    batch_index,
                    batch_count,
                    iteration=iteration,
                    dimension=dimension,
                )
            ),
            iterations=self.iterations,
            eps=self.eps,
            convergence_tol=self.convergence_tol,
        )
        return state, time.perf_counter() - global_start

    def _fit_location_covariance_states_from_batches(
        self,
        patch_batch_factory,
        sample_shape: tuple[int, ...],
        image_count: int,
        batch_count: int,
    ) -> tuple[list[TensorGaussianState], float, float]:
        zero_residual_mean = np.zeros(sample_shape, dtype=np.float32)
        covariance_seconds = 0.0
        shrinkage_seconds = 0.0
        states: list[TensorGaussianState] = []

        if self.covariance_shrinkage == 1.0:
            if self.global_covariance_state is None:
                raise RuntimeError("Global covariance state is missing for shrinkage.")
            for _ in range(self.patches_per_image):
                states.append(dict(self.global_covariance_state))
            return states, covariance_seconds, shrinkage_seconds

        if self.location_means is None:
            raise RuntimeError("location_means must be available before location fits.")

        for location_index in range(self.patches_per_image):
            covariance_start = time.perf_counter()

            def location_centered_batches(location_index: int = location_index):
                for patches_by_image in patch_batch_factory():
                    yield patches_by_image[:, location_index, ...] - self.location_means[location_index]

            state = _fit_tensor_separable_model_from_centered_batches(
                centered_batch_factory=location_centered_batches,
                sample_shape=sample_shape,
                mean=zero_residual_mean,
                observations_count=image_count,
                batch_count=batch_count,
                batch_progress_callback=lambda iteration, dimension, batch_index, batch_count, location_index=location_index: (
                    self._log_batch_progress(
                        "location-covariance",
                        batch_index,
                        batch_count,
                        iteration=iteration,
                        dimension=dimension,
                        location_index=location_index,
                    )
                ),
                iterations=self.iterations,
                eps=self.eps,
                convergence_tol=self.convergence_tol,
            )
            covariance_seconds += time.perf_counter() - covariance_start

            if self.covariance_shrinkage > 0.0:
                if self.global_covariance_state is None:
                    raise RuntimeError("Global covariance state is missing for shrinkage.")
                shrinkage_start = time.perf_counter()
                state = _blend_tensor_separable_covariances(
                    base_state=state,
                    bleed_state=self.global_covariance_state,
                    shrinkage=self.covariance_shrinkage,
                    eps=self.eps,
                )
                shrinkage_seconds += time.perf_counter() - shrinkage_start
            states.append(state)

        return states, covariance_seconds, shrinkage_seconds

    def _summarize_location_covariance_states(
        self,
        states: list[TensorGaussianState],
        sample_shape: tuple[int, ...],
    ) -> TensorGaussianState:
        if not states:
            raise ValueError("states must not be empty.")

        mean = np.zeros(sample_shape, dtype=np.float32)
        summarized_covariances: list[np.ndarray] = []
        for dimension in range(len(sample_shape)):
            covariance_stack = np.stack(
                [np.asarray(state["covariances"][dimension], dtype=np.float64) for state in states],
                axis=0,
            )
            summarized_covariance = covariance_stack.mean(axis=0)
            summarized_covariance = _regularize_covariance(summarized_covariance, eps=self.eps)
            summarized_covariance = _normalize_trace(summarized_covariance)
            summarized_covariances.append(summarized_covariance)

        converged = all(bool(state["converged"]) for state in states)
        last_max_relative_change = max(
            float(state["last_max_relative_change"])
            for state in states
            if state["last_max_relative_change"] is not None
        )
        iterations_run = max(int(state["iterations_run"]) for state in states)
        variance_scale = float(
            np.mean([float(state.get("variance_scale", 1.0)) for state in states])
        )
        return {
            "last_channel_contrib": None,
            "last_spatial_contrib": None,
            "mean": mean,
            "covariances": summarized_covariances,
            "variance_scale": variance_scale,
            "inverse_square_roots": [
                _matrix_inverse_square_root(covariance, eps=self.eps)
                for covariance in summarized_covariances
            ],
            "converged": converged,
            "last_max_relative_change": last_max_relative_change,
            "iterations_run": iterations_run,
            "fit_timing": {},
            "score_timing": {},
            "location_count": len(states),
        }

    def _compute_location_score_statistics(
        self,
        residuals_by_image: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        if self.score_normalization == "none":
            return None
        if self.location_covariance_states is None:
            raise RuntimeError("location_covariance_states must be available before score calibration.")

        location_means = np.empty(self.patches_per_image, dtype=np.float64)
        location_stds = np.empty(self.patches_per_image, dtype=np.float64)
        for location_index, state in enumerate(self.location_covariance_states):
            location_scores = _score_tensor_separable_model(
                state,
                residuals_by_image[:, location_index, ...],
            )
            location_means[location_index] = float(np.mean(location_scores))
            location_stds[location_index] = float(np.std(location_scores))
        return {
            "mean": location_means,
            "std": location_stds,
        }

    def _compute_location_score_statistics_from_batches(
        self,
        patch_batch_factory,
    ) -> dict[str, np.ndarray] | None:
        if self.score_normalization == "none":
            return None
        if self.location_covariance_states is None or self.location_means is None:
            raise RuntimeError("location covariance states and means must be available before score calibration.")

        score_sum = np.zeros(self.patches_per_image, dtype=np.float64)
        score_sum_sq = np.zeros(self.patches_per_image, dtype=np.float64)
        score_count = np.zeros(self.patches_per_image, dtype=np.int64)

        for patches_by_image in patch_batch_factory():
            residuals_by_image = patches_by_image - self.location_means[None, ...]
            for location_index, state in enumerate(self.location_covariance_states):
                location_scores = _score_tensor_separable_model(
                    state,
                    residuals_by_image[:, location_index, ...],
                )
                score_sum[location_index] += float(np.sum(location_scores))
                score_sum_sq[location_index] += float(np.sum(location_scores * location_scores))
                score_count[location_index] += len(location_scores)

        means = score_sum / np.maximum(score_count, 1)
        variances = (score_sum_sq / np.maximum(score_count, 1)) - (means * means)
        variances = np.maximum(variances, 0.0)
        return {
            "mean": means,
            "std": np.sqrt(variances),
        }

    def _normalize_scores_by_location(
        self,
        scores_by_image: np.ndarray,
    ) -> np.ndarray:
        if self.score_normalization == "none":
            return scores_by_image
        if self.location_score_statistics is None:
            raise RuntimeError("location_score_statistics are missing for score normalization.")

        if self.score_normalization == "zscore":
            mean = self.location_score_statistics["mean"][None, :]
            std = self.location_score_statistics["std"][None, :]
            return (scores_by_image - mean) / np.maximum(std, self.score_normalization_eps)

        raise RuntimeError(f"Unsupported score_normalization {self.score_normalization!r}.")

    def fit_from_patch_batches(self, patch_batch_factory) -> "LocationAwareTensorMahalanobisDetector":
        """Fit from repeatable batches shaped (batch_images, locations, h, w, c)."""

        fit_start = time.perf_counter()
        self._validate_mean_shrinkage()
        self._validate_shrinkage()
        self._validate_score_normalization()

        mean_start = time.perf_counter()
        location_sum: np.ndarray | None = None
        image_count = 0
        batch_count = 0
        sample_shape: tuple[int, ...] | None = None

        for batch_index, patches_by_image in enumerate(patch_batch_factory(), start=1):
            self._log_batch_progress("location-means", batch_index, None)
            patches_by_image = np.asarray(patches_by_image)
            if patches_by_image.ndim < 5:
                raise ValueError(
                    "patch batches must have shape (batch_images, patches_per_image, "
                    "patch_h, patch_w, channels[, additional_modes...])."
                )
            if patches_by_image.shape[1] != self.patches_per_image:
                raise ValueError(
                    f"Expected {self.patches_per_image} patch locations per image, "
                    f"got {patches_by_image.shape[1]}."
                )
            if patches_by_image.shape[0] == 0:
                raise ValueError("patch batches must contain at least one image.")
            if location_sum is None:
                sample_shape = tuple(patches_by_image.shape[2:])
                location_sum = np.zeros(patches_by_image.shape[1:], dtype=np.float64)
            elif tuple(patches_by_image.shape[2:]) != sample_shape:
                raise ValueError(
                    "All patch batches must use the same sample shape: "
                    f"expected {sample_shape}, got {tuple(patches_by_image.shape[2:])}."
                )
            location_sum += patches_by_image.sum(axis=0, dtype=np.float64)
            image_count += patches_by_image.shape[0]
            batch_count += 1

        if location_sum is None or sample_shape is None or image_count == 0:
            raise ValueError("patch_batch_factory produced no batches.")

        raw_location_means = location_sum / image_count
        self.location_means = self._shrink_location_means(raw_location_means)
        mean_seconds = time.perf_counter() - mean_start
        total_observations = image_count * self.patches_per_image

        self.global_covariance_state, global_covariance_seconds = (
            self._fit_global_covariance_from_batches(
                patch_batch_factory=patch_batch_factory,
                sample_shape=sample_shape,
                observations_count=total_observations,
                batch_count=batch_count,
            )
        )
        self.location_covariance_states, covariance_seconds, shrinkage_seconds = (
            self._fit_location_covariance_states_from_batches(
                patch_batch_factory=patch_batch_factory,
                sample_shape=sample_shape,
                image_count=image_count,
                batch_count=batch_count,
            )
        )
        self.location_aware_covariance_state = self._summarize_location_covariance_states(
            self.location_covariance_states,
            sample_shape=sample_shape,
        )
        self.covariance_state = self.location_aware_covariance_state
        self.location_score_statistics = self._compute_location_score_statistics_from_batches(
            patch_batch_factory=patch_batch_factory,
        )

        self.fit_timing = {
            "streaming_fit": True,
            "image_count": float(image_count),
            "batch_count": float(batch_count),
            "location_count": float(self.patches_per_image),
            "location_mean_seconds": mean_seconds,
            "global_covariance_fit_seconds": global_covariance_seconds,
            "location_covariance_fit_seconds": covariance_seconds,
            "shared_covariance_fit_seconds": covariance_seconds,
            "covariance_shrinkage_seconds": shrinkage_seconds,
            "mean_shrinkage": self.mean_shrinkage,
            "covariance_shrinkage": self.covariance_shrinkage,
            "score_normalization": self.score_normalization,
            "location_fit_workers": self.location_fit_workers,
            "location_aware_converged": all(
                bool(state["converged"]) for state in self.location_covariance_states
            ),
            "location_aware_iterations_run": max(
                int(state["iterations_run"]) for state in self.location_covariance_states
            ),
            "location_aware_last_max_relative_change": max(
                float(state["last_max_relative_change"])
                for state in self.location_covariance_states
                if state["last_max_relative_change"] is not None
            ),
            "global_converged": (
                None
                if self.global_covariance_state is None
                else bool(self.global_covariance_state["converged"])
            ),
            "global_iterations_run": (
                None
                if self.global_covariance_state is None
                else int(self.global_covariance_state["iterations_run"])
            ),
            "global_last_max_relative_change": (
                None
                if self.global_covariance_state is None
                else self.global_covariance_state["last_max_relative_change"]
            ),
            "total_seconds": time.perf_counter() - fit_start,
        }
        return self

    def fit(self, patches: np.ndarray) -> "LocationAwareTensorMahalanobisDetector":
        fit_start = time.perf_counter()
        self._validate_mean_shrinkage()
        self._validate_shrinkage()
        self._validate_score_normalization()

        patches = np.asarray(patches)
        if patches.ndim < 4:
            raise ValueError(
                "patches must have shape (samples, patch_h, patch_w, channels"
                "[, additional_modes...]), "
                f"got {patches.shape}."
            )
        reshape_start = time.perf_counter()
        patches_by_image = self._reshape_by_image(patches)
        reshape_seconds = time.perf_counter() - reshape_start

        mean_start = time.perf_counter()
        raw_location_means = patches_by_image.mean(axis=0)
        self.location_means = self._shrink_location_means(raw_location_means)
        mean_seconds = time.perf_counter() - mean_start

        residual_start = time.perf_counter()
        residuals_by_image = patches_by_image - self.location_means[None, ...]
        residual_seconds = time.perf_counter() - residual_start
        sample_shape = tuple(patches.shape[1:])

        self.global_covariance_state, global_covariance_seconds = (
            self._fit_global_covariance_from_array(
                residuals_by_image=residuals_by_image,
                sample_shape=sample_shape,
            )
        )
        self.location_covariance_states, covariance_seconds, shrinkage_seconds = (
            self._fit_location_covariance_states_from_array(
                residuals_by_image=residuals_by_image,
                sample_shape=sample_shape,
            )
        )
        self.location_aware_covariance_state = self._summarize_location_covariance_states(
            self.location_covariance_states,
            sample_shape=sample_shape,
        )
        self.covariance_state = self.location_aware_covariance_state
        self.location_score_statistics = self._compute_location_score_statistics(
            residuals_by_image=residuals_by_image,
        )

        self.fit_timing = {
            "reshape_seconds": reshape_seconds,
            "location_mean_seconds": mean_seconds,
            "residual_build_seconds": residual_seconds,
            "location_count": float(self.patches_per_image),
            "global_covariance_fit_seconds": global_covariance_seconds,
            "location_covariance_fit_seconds": covariance_seconds,
            "shared_covariance_fit_seconds": covariance_seconds,
            "covariance_shrinkage_seconds": shrinkage_seconds,
            "mean_shrinkage": self.mean_shrinkage,
            "covariance_shrinkage": self.covariance_shrinkage,
            "score_normalization": self.score_normalization,
            "location_fit_workers": self.location_fit_workers,
            "location_aware_converged": all(
                bool(state["converged"]) for state in self.location_covariance_states
            ),
            "location_aware_iterations_run": max(
                int(state["iterations_run"]) for state in self.location_covariance_states
            ),
            "location_aware_last_max_relative_change": max(
                float(state["last_max_relative_change"])
                for state in self.location_covariance_states
                if state["last_max_relative_change"] is not None
            ),
            "global_converged": (
                None
                if self.global_covariance_state is None
                else bool(self.global_covariance_state["converged"])
            ),
            "global_iterations_run": (
                None
                if self.global_covariance_state is None
                else int(self.global_covariance_state["iterations_run"])
            ),
            "global_last_max_relative_change": (
                None
                if self.global_covariance_state is None
                else self.global_covariance_state["last_max_relative_change"]
            ),
            "total_seconds": time.perf_counter() - fit_start,
        }
        return self

    def score(self, patches: np.ndarray) -> np.ndarray:
        if self.location_means is None or self.location_covariance_states is None:
            raise RuntimeError("Call fit() before score().")

        patches = np.asarray(patches)
        expected_sample_shape = tuple(self.location_means.shape[1:])
        if tuple(patches.shape[1:]) != expected_sample_shape:
            raise ValueError(
                "Scoring patch shape does not match the fitted model: "
                f"expected {expected_sample_shape}, got {tuple(patches.shape[1:])}."
            )
        score_start = time.perf_counter()
        reshape_start = time.perf_counter()
        patches_by_image = self._reshape_by_image(patches)
        reshape_seconds = time.perf_counter() - reshape_start

        residual_start = time.perf_counter()
        residuals_by_image = patches_by_image - self.location_means[None, ...]
        residual_seconds = time.perf_counter() - residual_start

        covariance_start = time.perf_counter()
        scores_by_image = np.empty(
            (patches_by_image.shape[0], self.patches_per_image),
            dtype=np.float32,
        )
        score_center_seconds = 0.0
        score_whitening_seconds = 0.0
        score_norm_seconds = 0.0
        if self.covariance_shrinkage == 1.0 and self.global_covariance_state is not None:
            # Every location shares the same separable tensor covariance. Batch
            # locations along the sample axis so whitening remains tensor-valued
            # but avoids one tiny Python/BLAS call per spatial position.
            for location_start in range(
                0,
                self.patches_per_image,
                self.shared_score_location_batch_size,
            ):
                location_end = min(
                    location_start + self.shared_score_location_batch_size,
                    self.patches_per_image,
                )
                location_count = location_end - location_start
                residual_batch = residuals_by_image[
                    :, location_start:location_end, ...
                ].reshape(
                    patches_by_image.shape[0] * location_count,
                    *expected_sample_shape,
                )
                location_scores = _score_tensor_separable_model(
                    self.global_covariance_state,
                    residual_batch,
                    store_contributions=False,
                ).reshape(patches_by_image.shape[0], location_count)
                scores_by_image[:, location_start:location_end] = location_scores.astype(
                    np.float32,
                    copy=False,
                )
                state_score_timing = self.global_covariance_state.get("score_timing", {})
                score_center_seconds += float(state_score_timing.get("center_seconds", 0.0))
                score_whitening_seconds += float(
                    state_score_timing.get("whitening_seconds", 0.0)
                )
                score_norm_seconds += float(state_score_timing.get("norm_seconds", 0.0))
        else:
            for location_index, state in enumerate(self.location_covariance_states):
                location_scores = _score_tensor_separable_model(
                    state,
                    residuals_by_image[:, location_index, ...],
                )
                scores_by_image[:, location_index] = location_scores.astype(
                    np.float32,
                    copy=False,
                )
                state_score_timing = state.get("score_timing", {})
                score_center_seconds += float(state_score_timing.get("center_seconds", 0.0))
                score_whitening_seconds += float(
                    state_score_timing.get("whitening_seconds", 0.0)
                )
                score_norm_seconds += float(state_score_timing.get("norm_seconds", 0.0))

        covariance_seconds = time.perf_counter() - covariance_start
        self.score_timing = {
            "reshape_seconds": reshape_seconds,
            "residual_build_seconds": residual_seconds,
            "location_covariance_score_seconds": covariance_seconds,
            "shared_covariance_score_seconds": covariance_seconds,
            "score_normalization": self.score_normalization,
            "center_seconds": score_center_seconds,
            "whitening_seconds": score_whitening_seconds,
            "norm_seconds": score_norm_seconds,
            "total_seconds": time.perf_counter() - score_start,
        }
        normalized_scores_by_image = self._normalize_scores_by_location(scores_by_image)
        return normalized_scores_by_image.reshape(-1)


class NeighborhoodScoreLocationAwareTensorMahalanobisDetector(LocationAwareTensorMahalanobisDetector):
    """Apply spatial-neighborhood pooling to finished location Mahalanobis scores."""

    def __init__(
        self,
        *args,
        grid_shape: tuple[int, int],
        score_neighbor_radius: int = 1,
        score_neighbor_pooling: str = "weighted_mean",
        score_neighbor_sigma: float | None = None,
        score_neighbor_sigma_range: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.grid_shape = grid_shape
        self.score_neighbor_radius = score_neighbor_radius
        self.score_neighbor_pooling = score_neighbor_pooling
        self.score_neighbor_sigma = score_neighbor_sigma
        self.score_neighbor_sigma_range = score_neighbor_sigma_range
        if (
            len(grid_shape) != 2
            or not all(isinstance(size, (int, np.integer)) and size > 0 for size in grid_shape)
        ):
            raise ValueError(f"grid_shape must contain two positive integers, got {grid_shape!r}.")
        if score_neighbor_pooling not in {"mean", "max", "weighted_mean"}:
            raise ValueError(
                "score_neighbor_pooling must be 'mean', 'max', or 'weighted_mean', "
                f"got {score_neighbor_pooling!r}."
            )
        self.neighbor_indices_by_location = location_neighbors(
            grid_shape[0],
            grid_shape[1],
            score_neighbor_radius,
        )
        if score_neighbor_pooling == "weighted_mean" and score_neighbor_sigma_range is None:
            sigma_spatial = float(
                max(score_neighbor_radius, 1)
                if score_neighbor_sigma is None
                else score_neighbor_sigma
            )
            if sigma_spatial <= 0.0:
                raise ValueError(f"score_neighbor_sigma must be positive, got {sigma_spatial}.")
            spatial_weights = []
            for row in range(grid_shape[0]):
                for col in range(grid_shape[1]):
                    row_lo = max(0, row - score_neighbor_radius)
                    row_hi = min(grid_shape[0], row + score_neighbor_radius + 1)
                    col_lo = max(0, col - score_neighbor_radius)
                    col_hi = min(grid_shape[1], col + score_neighbor_radius + 1)
                    distances_sq = np.asarray(
                        [
                            (r - row) ** 2 + (c - col) ** 2
                            for r in range(row_lo, row_hi)
                            for c in range(col_lo, col_hi)
                        ],
                        dtype=np.float64,
                    )
                    weights = np.exp(-distances_sq / (2.0 * sigma_spatial * sigma_spatial))
                    weight_sum = float(weights.sum())
                    if weight_sum > 0.0:
                        weights /= weight_sum
                    else:
                        weights.fill(1.0 / max(len(weights), 1))
                    spatial_weights.append(weights.astype(np.float32))
            self.neighbor_weights_by_location = spatial_weights
        else:
            self.neighbor_weights_by_location = None
        if len(self.neighbor_indices_by_location) != self.patches_per_image:
            raise ValueError(
                "grid_shape does not match patches_per_image: "
                f"{grid_shape} gives {len(self.neighbor_indices_by_location)} locations, "
                f"but patches_per_image={self.patches_per_image}."
            )

    def fit(self, patches: np.ndarray) -> "NeighborhoodScoreLocationAwareTensorMahalanobisDetector":
        super().fit(patches)
        if (
            self.score_neighbor_pooling == "weighted_mean"
            and self.score_neighbor_sigma_range is not None
        ):
            if self.location_means is None:
                raise RuntimeError("location_means must be available before bilateral weights.")
            self.neighbor_weights_by_location = location_neighbor_bilateral_weights(
                self.grid_shape[0],
                self.grid_shape[1],
                self.location_means,
                self.score_neighbor_radius,
                sigma_spatial=self.score_neighbor_sigma,
                sigma_range=self.score_neighbor_sigma_range,
            )
        return self

    def fit_from_patch_batches(
        self,
        patch_batch_factory,
    ) -> "NeighborhoodScoreLocationAwareTensorMahalanobisDetector":
        super().fit_from_patch_batches(patch_batch_factory)
        if (
            self.score_neighbor_pooling == "weighted_mean"
            and self.score_neighbor_sigma_range is not None
        ):
            if self.location_means is None:
                raise RuntimeError("location_means must be available before bilateral weights.")
            self.neighbor_weights_by_location = location_neighbor_bilateral_weights(
                self.grid_shape[0],
                self.grid_shape[1],
                self.location_means,
                self.score_neighbor_radius,
                sigma_spatial=self.score_neighbor_sigma,
                sigma_range=self.score_neighbor_sigma_range,
            )
        return self

    def score(self, patches: np.ndarray) -> np.ndarray:
        base_scores = super().score(patches)
        pooling_start = time.perf_counter()
        scores_by_image = self._reshape_by_image(base_scores[:, None])[..., 0]
        if self.score_neighbor_sigma_range is None:
            pooled_scores = aggregate_regular_grid_scores(
                scores_by_image=scores_by_image,
                grid_shape=self.grid_shape,
                radius=self.score_neighbor_radius,
                pooling=self.score_neighbor_pooling,
                sigma=self.score_neighbor_sigma,
            )
        else:
            pooled_scores = aggregate_location_scores(
                scores_by_image=scores_by_image,
                neighbor_indices_by_location=self.neighbor_indices_by_location,
                pooling=self.score_neighbor_pooling,
                neighbor_weights_by_location=self.neighbor_weights_by_location,
            )
        pooling_seconds = time.perf_counter() - pooling_start
        self.score_timing.update(
            {
                "score_neighbor_radius": float(self.score_neighbor_radius),
                "score_neighbor_pooling": self.score_neighbor_pooling,
                "score_neighbor_sigma": self.score_neighbor_sigma,
                "score_neighbor_sigma_range": self.score_neighbor_sigma_range,
                "neighborhood_pooling_seconds": pooling_seconds,
                "total_seconds": self.score_timing.get("total_seconds", 0.0)
                + pooling_seconds,
            }
        )
        return pooled_scores.reshape(-1)
