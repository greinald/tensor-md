from __future__ import annotations

import numpy as np
import time


EPS = 1e-6


def _matrix_inverse_square_root(matrix: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute a stable symmetric inverse square root of a covariance matrix."""

    # For Mahalanobis distance we need the inverse covariance. A numerically
    # convenient equivalent is to whiten with Sigma^(-1/2) first and then take a
    # plain squared Euclidean norm.
    #
    # If Sigma = Q diag(lambda) Q^T, then
    # Sigma^(-1/2) = Q diag(1 / sqrt(lambda)) Q^T.
    #
    # After whitening, z = Sigma^(-1/2) (x - mu), so
    #
    # ||z||^2
    # = z^T z
    # = (x - mu)^T Sigma^(-1/2)^T Sigma^(-1/2) (x - mu)
    # = (x - mu)^T Sigma^(-1) (x - mu),
    #
    # because for a symmetric covariance matrix, Sigma^(-1/2)^T Sigma^(-1/2)
    # equals Sigma^(-1). So this is the same Mahalanobis distance, just written
    # in whitened coordinates.

    # eight is used for symmetry and numerical stability
    # but any orthogonal diagonalization would work here.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clamp tiny eigenvalues so whitening does not explode numerically.
    eigenvalues = np.maximum(eigenvalues, eps)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _regularize_covariance(covariance: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Add a small ridge term so covariance inversion is numerically stable."""

    # This is standard ridge regularization for covariance matrices:
    #
    # Sigma_reg = Sigma + alpha I.
    #
    # Adding a small diagonal term pushes all eigenvalues upward. That matters
    # because later we invert the covariance, or take its inverse square root,
    # Scale the ridge by the average variance so eps stays meaningful across
    # different covariance magnitudes.
    # trace(Sigma) / d is the average variance across all dimensions, 
    # so it is a natural scale for regularization.
    # Using that as a scale makes the ridge adapt to the overall size of the
    # covariance instead of being a raw fixed constant
    # Eye just used to add the ridge to the diagonal and not off diagonal.
    scale = np.trace(covariance) / covariance.shape[0]
    return covariance + eps * max(scale, eps) * np.eye(covariance.shape[0])


def _normalize_trace(covariance: np.ndarray) -> np.ndarray:
    """Normalize covariance scale so dimension covariances remain identifiable."""

    trace = np.trace(covariance)
    # If the covariance is zero, we can't rescale it.
    if trace <= 0:
        return covariance
    # In the separable model, only the relative shape of each dimension covariance
    # matters. Trace normalization removes arbitrary scaling drift.
    #
    # Example: in a Kronecker-style product, multiplying one dimension covariance by
    # 10 and another by 1/10 can leave the combined model essentially unchanged.
    # So the scale of each individual dimension covariance is not unique by itself.
    #
    # We fix that ambiguity by rescaling the matrix so that its trace equals its
    # dimension. Since trace is the sum of diagonal variances, this sets a
    # consistent total-variance scale without changing the covariance shape.
    return covariance * (covariance.shape[0] / trace)


def _dimension_multiply(tensor: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Multiply a batch of tensors by a matrix along one non-batch dimension."""

    # Move the target axis to the end so a standard matrix multiply applies to
    # every patch in the batch at once.
    moved = np.moveaxis(tensor, axis, -1)
    multiplied = moved @ matrix.T
    return np.moveaxis(multiplied, -1, axis)


def _unfold_dimension(patches: np.ndarray, dimension: int) -> np.ndarray:
    """Unfold all patches along one tensor dimension for covariance estimation."""
    # The dimension index in the patch tensor (excluding batch axis 0)
    axis = dimension + 1
    
    # Move the target axis to the very front (axis 0)
    moved = np.moveaxis(patches, axis, 0)
    
    # Reshape keeping the target axis intact, flattening all other axes into columns
    return moved.reshape(moved.shape[0], -1)

TensorGaussianState = dict[str, object]


def _relative_matrix_change(old: np.ndarray, new: np.ndarray, eps: float) -> float:
    """Return the relative Frobenius-norm change between two matrices."""

    old_norm = np.linalg.norm(old, ord="fro")
    diff_norm = np.linalg.norm(new - old, ord="fro")
    return float(diff_norm / max(old_norm, eps))


def _tensor_variance_scale(
    centered: np.ndarray,
    covariances: list[np.ndarray],
    eps: float,
) -> float:
    """Estimate the scalar variance removed by trace-normalizing every mode."""

    whitened = centered
    for dimension, covariance in enumerate(covariances):
        whitened = _dimension_multiply(
            whitened,
            _matrix_inverse_square_root(covariance, eps=eps),
            axis=dimension + 1,
        )
    mean_square = float(np.mean(np.asarray(whitened, dtype=np.float64) ** 2))
    return max(mean_square, eps)


def _tensor_variance_scale_from_batches(
    centered_batch_factory,
    covariances: list[np.ndarray],
    eps: float,
) -> float:
    """Streaming counterpart of _tensor_variance_scale()."""

    inverse_square_roots = [
        _matrix_inverse_square_root(covariance, eps=eps)
        for covariance in covariances
    ]
    squared_sum = 0.0
    value_count = 0
    for centered_batch in centered_batch_factory():
        whitened = centered_batch
        for dimension, inverse_square_root in enumerate(inverse_square_roots):
            whitened = _dimension_multiply(
                whitened,
                inverse_square_root,
                axis=dimension + 1,
            )
        whitened64 = np.asarray(whitened, dtype=np.float64)
        squared_sum += float(np.sum(whitened64 * whitened64))
        value_count += whitened64.size

    if value_count == 0:
        raise ValueError("centered_batch_factory produced no batches.")
    return max(squared_sum / value_count, eps)


def _fit_tensor_separable_model(
    patches: np.ndarray,
    iterations: int = 6,
    eps: float = EPS,
    convergence_tol: float = 1e-4,
) -> TensorGaussianState:
    """Fit one separable tensor Gaussian model and return its state."""

    fit_start = time.perf_counter()
    mean_start = time.perf_counter()
    mean = patches.mean(axis=0)
    mean_seconds = time.perf_counter() - mean_start

    center_start = time.perf_counter()
    centered = patches - mean
    center_seconds = time.perf_counter() - center_start
    return _fit_tensor_separable_model_from_centered(
        centered=centered,
        mean=mean,
        mean_seconds=mean_seconds,
        center_seconds=center_seconds,
        iterations=iterations,
        eps=eps,
        convergence_tol=convergence_tol,
        fit_start=fit_start,
    )


def _fit_tensor_separable_model_from_centered(
    centered: np.ndarray,
    mean: np.ndarray,
    mean_seconds: float = 0.0,
    center_seconds: float = 0.0,
    iterations: int = 6,
    eps: float = EPS,
    convergence_tol: float = 1e-4,
    fit_start: float | None = None,
) -> TensorGaussianState:
    """Fit one separable tensor Gaussian model from pre-centered patches."""

    sample_shape = centered.shape[1:]
    covariances = [np.eye(size) for size in sample_shape]
    whitening_seconds = 0.0
    unfold_seconds = 0.0
    covariance_update_seconds = 0.0
    converged = False
    last_max_relative_change: float | None = None
    iterations_run = 0

    for _ in range(iterations):
        iterations_run += 1
        max_relative_change = 0.0
        for dimension, dimension_size in enumerate(sample_shape):
            # A singleton tensor mode has the only possible trace-normalized
            # covariance [[1]]. Re-estimating it cannot change the model and
            # needlessly whitens the full tensor through every other mode.
            if dimension_size == 1:
                continue
            whitened = centered
            for other_dimension, covariance in enumerate(covariances):
                if other_dimension == dimension:
                    continue
                whitening_start = time.perf_counter()
                inv_sqrt = _matrix_inverse_square_root(covariance, eps=eps)
                whitened = _dimension_multiply(
                    whitened,
                    inv_sqrt,
                    axis=other_dimension + 1,
                )
                whitening_seconds += time.perf_counter() - whitening_start

            unfold_start = time.perf_counter()
            unfolded = _unfold_dimension(whitened, dimension)
            unfold_seconds += time.perf_counter() - unfold_start

            covariance_start = time.perf_counter()
            covariance = unfolded @ unfolded.T / unfolded.shape[1]
            covariance = _regularize_covariance(covariance, eps=eps)
            covariance = _normalize_trace(covariance)
            covariance_update_seconds += time.perf_counter() - covariance_start

            relative_change = _relative_matrix_change(
                covariances[dimension],
                covariance,
                eps=eps,
            )
            max_relative_change = max(max_relative_change, relative_change)
            covariances[dimension] = covariance

        last_max_relative_change = max_relative_change
        if max_relative_change < convergence_tol:
            converged = True
            break

    scale_start = time.perf_counter()
    variance_scale = _tensor_variance_scale(centered, covariances, eps=eps)
    scale_seconds = time.perf_counter() - scale_start

    inverse_start = time.perf_counter()
    inverse_square_roots = [
        _matrix_inverse_square_root(covariance, eps=eps)
        for covariance in covariances
    ]
    inverse_seconds = time.perf_counter() - inverse_start
    return {
        "last_channel_contrib": None,
        "last_spatial_contrib": None,
        "mean": mean,
        "covariances": covariances,
        "variance_scale": variance_scale,
        "inverse_square_roots": inverse_square_roots,
        "converged": converged,
        "last_max_relative_change": last_max_relative_change,
        "iterations_run": iterations_run,
        "fit_timing": {
            "mean_seconds": mean_seconds,
            "center_seconds": center_seconds,
            "whitening_seconds": whitening_seconds,
            "unfold_seconds": unfold_seconds,
            "covariance_update_seconds": covariance_update_seconds,
            "variance_scale_seconds": scale_seconds,
            "inverse_square_root_seconds": inverse_seconds,
            "iterations_run": float(iterations_run),
            "total_seconds": (
                time.perf_counter() - fit_start
                if fit_start is not None
                else mean_seconds
                + center_seconds
                + whitening_seconds
                + unfold_seconds
                + covariance_update_seconds
                + scale_seconds
                + inverse_seconds
            ),
        },
        "score_timing": {},
    }


def _fit_tensor_separable_model_from_centered_batches(
    centered_batch_factory,
    sample_shape: tuple[int, ...],
    mean: np.ndarray,
    observations_count: int,
    batch_count: int | None = None,
    batch_progress_callback=None,
    iterations: int = 6,
    eps: float = EPS,
    convergence_tol: float = 1e-4,
) -> TensorGaussianState:
    """Fit a tensor Gaussian model by repeatedly streaming centered patch batches."""

    fit_start = time.perf_counter()
    covariances = [np.eye(size) for size in sample_shape]
    whitening_seconds = 0.0
    unfold_seconds = 0.0
    covariance_update_seconds = 0.0
    converged = False
    last_max_relative_change: float | None = None
    iterations_run = 0

    if observations_count <= 0:
        raise ValueError("observations_count must be positive.")

    for _ in range(iterations):
        iterations_run += 1
        max_relative_change = 0.0
        for dimension, dimension_size in enumerate(sample_shape):
            if dimension_size == 1:
                continue
            covariance_sum = np.zeros((dimension_size, dimension_size), dtype=np.float64)
            column_count = 0

            for batch_index, centered_batch in enumerate(centered_batch_factory(), start=1):
                if batch_progress_callback is not None:
                    batch_progress_callback(
                        iteration=iterations_run,
                        dimension=dimension,
                        batch_index=batch_index,
                        batch_count=batch_count,
                    )
                whitened = centered_batch
                for other_dimension, covariance in enumerate(covariances):
                    if other_dimension == dimension:
                        continue
                    whitening_start = time.perf_counter()
                    inv_sqrt = _matrix_inverse_square_root(covariance, eps=eps)
                    whitened = _dimension_multiply(
                        whitened,
                        inv_sqrt,
                        axis=other_dimension + 1,
                    )
                    whitening_seconds += time.perf_counter() - whitening_start

                unfold_start = time.perf_counter()
                unfolded = _unfold_dimension(whitened, dimension)
                unfold_seconds += time.perf_counter() - unfold_start

                covariance_sum += unfolded @ unfolded.T
                column_count += unfolded.shape[1]

            if column_count == 0:
                raise ValueError("centered_batch_factory produced no batches.")

            covariance_start = time.perf_counter()
            covariance = covariance_sum / column_count
            covariance = _regularize_covariance(covariance, eps=eps)
            covariance = _normalize_trace(covariance)
            covariance_update_seconds += time.perf_counter() - covariance_start

            relative_change = _relative_matrix_change(
                covariances[dimension],
                covariance,
                eps=eps,
            )
            max_relative_change = max(max_relative_change, relative_change)
            covariances[dimension] = covariance

        last_max_relative_change = max_relative_change
        if max_relative_change < convergence_tol:
            converged = True
            break

    scale_start = time.perf_counter()
    variance_scale = _tensor_variance_scale_from_batches(
        centered_batch_factory,
        covariances,
        eps=eps,
    )
    scale_seconds = time.perf_counter() - scale_start

    inverse_start = time.perf_counter()
    inverse_square_roots = [
        _matrix_inverse_square_root(covariance, eps=eps)
        for covariance in covariances
    ]
    inverse_seconds = time.perf_counter() - inverse_start
    return {
        "last_channel_contrib": None,
        "last_spatial_contrib": None,
        "mean": mean,
        "covariances": covariances,
        "variance_scale": variance_scale,
        "inverse_square_roots": inverse_square_roots,
        "converged": converged,
        "last_max_relative_change": last_max_relative_change,
        "iterations_run": iterations_run,
        "fit_timing": {
            "mean_seconds": 0.0,
            "center_seconds": 0.0,
            "whitening_seconds": whitening_seconds,
            "unfold_seconds": unfold_seconds,
            "covariance_update_seconds": covariance_update_seconds,
            "variance_scale_seconds": scale_seconds,
            "inverse_square_root_seconds": inverse_seconds,
            "iterations_run": float(iterations_run),
            "observations_count": float(observations_count),
            "total_seconds": time.perf_counter() - fit_start,
        },
        "score_timing": {},
    }


def _score_tensor_separable_model(
    model: TensorGaussianState,
    patches: np.ndarray,
    store_contributions: bool = True,
) -> np.ndarray:
    """Score patches with a fitted separable tensor Gaussian state."""

    mean = model.get("mean")
    inverse_square_roots = model.get("inverse_square_roots")
    variance_scale = model.get("variance_scale", 1.0)
    if mean is None or inverse_square_roots is None:
        raise RuntimeError("Tensor Gaussian state is not fitted.")
    variance_scale = float(variance_scale)
    if not np.isfinite(variance_scale) or variance_scale <= 0.0:
        raise RuntimeError(
            f"Tensor Gaussian state has invalid variance_scale={variance_scale}."
        )

    score_start = time.perf_counter()
    center_start = time.perf_counter()
    whitened = patches - mean
    center_seconds = time.perf_counter() - center_start

    whitening_seconds = 0.0
    for dimension, inv_sqrt in enumerate(inverse_square_roots):
        whitening_start = time.perf_counter()
        whitened = _dimension_multiply(whitened, inv_sqrt, axis=dimension + 1)
        whitening_seconds += time.perf_counter() - whitening_start
    whitened = whitened / np.sqrt(variance_scale)

    norm_start = time.perf_counter()
    scores = np.sum(whitened * whitened, axis=tuple(range(1, whitened.ndim)))
    if store_contributions:
        model["last_channel_contrib"] = np.sum(
            whitened * whitened,
            axis=(1, 2),
        )
        model["last_spatial_contrib"] = np.sum(
            whitened * whitened,
            axis=tuple(range(3, whitened.ndim)),
        )
    norm_seconds = time.perf_counter() - norm_start
    model["score_timing"] = {
        "center_seconds": center_seconds,
        "whitening_seconds": whitening_seconds,
        "norm_seconds": norm_seconds,
        "total_seconds": time.perf_counter() - score_start,
    }
    return scores


def _blend_tensor_separable_covariances(
    base_state: TensorGaussianState,
    bleed_state: TensorGaussianState,
    shrinkage: float,
    eps: float = EPS,
) -> TensorGaussianState:
    """Blend one tensor-separable covariance state toward another."""

    base_covariances = base_state.get("covariances")
    bleed_covariances = bleed_state.get("covariances")
    if base_covariances is None or bleed_covariances is None:
        raise RuntimeError("Both tensor covariance states must be fitted before blending.")

    blended_covariances: list[np.ndarray] = []
    for base_covariance, bleed_covariance in zip(
        base_covariances,
        bleed_covariances,
        strict=True,
    ):
        blended_covariance = (
            (1.0 - shrinkage) * base_covariance
            + shrinkage * bleed_covariance
        )
        blended_covariance = _regularize_covariance(blended_covariance, eps=eps)
        blended_covariance = _normalize_trace(blended_covariance)
        blended_covariances.append(blended_covariance)

    blended_state = dict(base_state)
    blended_state["covariances"] = blended_covariances
    base_scale = float(base_state.get("variance_scale", 1.0))
    bleed_scale = float(bleed_state.get("variance_scale", 1.0))
    blended_state["variance_scale"] = (
        (1.0 - shrinkage) * base_scale + shrinkage * bleed_scale
    )
    blended_state["inverse_square_roots"] = [
        _matrix_inverse_square_root(covariance, eps=eps)
        for covariance in blended_covariances
    ]
    return blended_state


def __getattr__(name: str):
    """Keep the optional scikit-learn vector baseline from blocking tensor imports."""

    if name == "VectorizedMahalanobisDetector":
        try:
            from .vectorized_mahalanobis_detector import VectorizedMahalanobisDetector
        except ImportError:
            from vectorized_mahalanobis_detector import VectorizedMahalanobisDetector

        return VectorizedMahalanobisDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EPS",
    "TensorGaussianState",
    "_dimension_multiply",
    "_fit_tensor_separable_model",
    "_fit_tensor_separable_model_from_centered",
    "_fit_tensor_separable_model_from_centered_batches",
    "_blend_tensor_separable_covariances",
    "_matrix_inverse_square_root",
    "_normalize_trace",
    "_regularize_covariance",
    "_score_tensor_separable_model",
    "_unfold_dimension",
    "VectorizedMahalanobisDetector",
]
