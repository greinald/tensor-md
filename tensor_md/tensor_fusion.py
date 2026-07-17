from __future__ import annotations

import numpy as np

try:
    from .Data_Loading import (
        SpatialPatchLayout,
        feature_patch_layout_for_shape,
        fit_channel_pca,
        transform_channel_pca,
    )
except ImportError:
    from Data_Loading import (
        SpatialPatchLayout,
        feature_patch_layout_for_shape,
        fit_channel_pca,
        transform_channel_pca,
    )


def _import_tensorflow():
    import tensorflow as tf

    return tf


def resize_feature_maps(maps: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a batch of feature maps to a shared spatial grid."""

    tf = _import_tensorflow()
    resized = tf.image.resize(maps, target_hw, method="bilinear").numpy()
    return np.asarray(resized, dtype=np.float32)


def _validate_patch_geometry(
    map_hw: tuple[int, int],
    patch_size: tuple[int, int],
    stride: int,
) -> None:
    """Validate spatial tensor-patch geometry before creating a strided view."""

    if (
        len(patch_size) != 2
        or not all(isinstance(size, (int, np.integer)) and size > 0 for size in patch_size)
    ):
        raise ValueError(f"patch_size must contain two positive integers, got {patch_size!r}.")
    if not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride!r}.")
    if patch_size[0] > map_hw[0] or patch_size[1] > map_hw[1]:
        raise ValueError(
            "patch_size must fit inside the fused feature map: "
            f"patch_size={patch_size}, map_hw={map_hw}."
        )


def fused_patch_layout(
    fused_map_shape: tuple[int, ...],
    patch_size: tuple[int, int] = (1, 1),
    stride: int = 1,
) -> SpatialPatchLayout:
    """Return the valid-window layout used for fused tensor patches."""

    if len(fused_map_shape) != 4:
        raise ValueError(
            "fused_map_shape must be (H, W, C, L), "
            f"got {fused_map_shape}."
        )
    _validate_patch_geometry(fused_map_shape[:2], patch_size, stride)
    return feature_patch_layout_for_shape(
        fused_map_shape,
        patch_size=patch_size,
        stride=stride,
    )


def fused_patch_grid_shape(
    fused_map_shape: tuple[int, ...],
    patch_size: tuple[int, int] = (1, 1),
    stride: int = 1,
) -> tuple[int, int]:
    """Return the rows and columns in the fused tensor-patch location grid."""

    layout = fused_patch_layout(fused_map_shape, patch_size=patch_size, stride=stride)
    rows = len(range(0, layout.image_shape[0] - patch_size[0] + 1, stride))
    cols = len(range(0, layout.image_shape[1] - patch_size[1] + 1, stride))
    return rows, cols


def fused_maps_to_patch_batch(
    fused_maps: np.ndarray,
    patch_size: tuple[int, int] = (1, 1),
    stride: int = 1,
) -> np.ndarray:
    """Extract spatial tensor patches from fused maps.

    Parameters
    ----------
    fused_maps:
        Dense fused representation with shape ``(N, H, W, C, L)``.
    patch_size:
        Spatial height and width retained inside every tensor observation.
    stride:
        Step between valid-window patch locations on the feature grid.

    Returns
    -------
    np.ndarray
        Detector input with shape ``(N * P, p_h, p_w, C, L)``, where ``P``
        is the number of valid patch locations per image.  The default
        ``patch_size=(1, 1), stride=1`` is exactly backward compatible.
    """

    fused_maps = np.asarray(fused_maps)
    if fused_maps.ndim != 5:
        raise ValueError(
            "fused_maps must have shape (N, H, W, C, L), "
            f"got {fused_maps.shape}."
        )
    n_images, map_h, map_w, channels, layers = fused_maps.shape
    _validate_patch_geometry((map_h, map_w), patch_size, stride)
    patch_h, patch_w = patch_size

    # sliding_window_view appends the window axes after the untouched C and L
    # axes: (N, out_h, out_w, C, L, p_h, p_w).  Move p_h and p_w before C and
    # L so each detector observation retains the intended H x W x C x L modes.
    windows = np.lib.stride_tricks.sliding_window_view(
        fused_maps,
        window_shape=(patch_h, patch_w),
        axis=(1, 2),
    )[:, ::stride, ::stride]
    windows = windows.transpose(0, 1, 2, 5, 6, 3, 4)
    patches_per_image = windows.shape[1] * windows.shape[2]
    return windows.reshape(
        n_images * patches_per_image,
        patch_h,
        patch_w,
        channels,
        layers,
    ).astype(np.float32, copy=False)


class PerLayerPCATensorFusion:
    """Fit PCA per CNN layer, align grids, and stack layers as tensor modes."""

    def __init__(
        self,
        channels: int,
        target_hw: tuple[int, int],
        batch_size: int = 20_000,
    ):
        self.channels = channels
        self.target_hw = target_hw
        self.batch_size = batch_size
        self.name = f"clean_fusion_pca_{channels}x2"

    def fit(self, first_layer_maps: np.ndarray, second_layer_maps: np.ndarray):
        self.first_layer_pca = fit_channel_pca(
            first_layer_maps,
            n_components=self.channels,
            batch_size=self.batch_size,
        )
        self.second_layer_pca = fit_channel_pca(
            second_layer_maps,
            n_components=self.channels,
            batch_size=self.batch_size,
        )
        return self

    def transform(
        self,
        first_layer_maps: np.ndarray,
        second_layer_maps: np.ndarray,
    ) -> np.ndarray:
        reduced_first = transform_channel_pca(
            first_layer_maps,
            self.first_layer_pca,
            batch_size=self.batch_size,
        )
        reduced_second = transform_channel_pca(
            second_layer_maps,
            self.second_layer_pca,
            batch_size=self.batch_size,
        )
        resized_second = resize_feature_maps(reduced_second, self.target_hw)
        return np.stack([reduced_first, resized_second], axis=-1).astype(
            np.float32,
            copy=False,
        )
