from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from sklearn.decomposition import IncrementalPCA



IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PatchExtractionConfig:
    """Configuration for loading one MVTec category as patch tensors."""

    category: str
    data_root: Path | None = None
    image_size: tuple[int, int] = (256, 256)
    patch_size: tuple[int, int] = (16, 16)
    stride: int = 4
    anomaly_threshold: float = 0.05
    max_train_images: int | None = 300
    max_test_good_images: int | None = 300
    max_test_anomaly_images_per_type: int | None = 300
    object_crop_mode: str = "none"
    white_background_threshold: int = 245
    dark_background_threshold: int = 40
    object_crop_padding: int = 4
    center_cropped_object: bool = False
    yolo_obb_preprocessing: bool = False
    yolo_obb_model_path: str = "yolo11n-obb.pt"
    yolo_obb_device: str | None = None
    yolo_obb_padding: int = 10
    input_representation: str = "raw_pixels"
    cnn_backbone: str = "ResNet50"
    # Optional user-owned extractor. It receives a float32 NHWC batch in [0, 1]
    # and returns one NHWC feature-map batch or a list/tuple of such batches.
    # When supplied, no framework-specific backbone is constructed.
    cnn_feature_extractor: Callable[[np.ndarray], Any] | Any | None = None
    cnn_layer_name: str = "conv3_block4_out"
    cnn_layer_names: tuple[str, ...] | None = None
    cnn_pca_components: int | tuple[int, ...] | None = None
    cnn_pca_chunk_size: int = 20000
    cnn_fusion_channels: int | None = None
    cnn_fusion_seed: int = 0
    cnn_weights: str | None = "imagenet"
    yolo_model_path: str = "yolo11n.pt"
    yolo_target_layer_idx: int | None = None
    yolo_device: str | None = None
    cnn_feature_patch_size: tuple[int, int] = (1, 1)
    cnn_feature_stride: int = 1
    cnn_batch_size: int = 16
    debug_visualization: bool = False
    debug_anomalous_image_masks: bool = False
    debug_log_samples: bool = False
    debug_timing: bool = False
    debug_memory: bool = False
    debug_sample_count: int = 3
    debug_random_seed: int = 0


@dataclass(frozen=True)
class PatchDataset:
    """Container for patch tensors and their binary patch labels."""

    patches: np.ndarray
    labels: np.ndarray
    image_paths: list[Path]
    patches_per_image: int
    patch_image_indices: np.ndarray
    patch_local_indices: np.ndarray


@dataclass(frozen=True)
class PatchDatasets:
    """Bundle for the train/test patch datasets of one MVTec category."""

    category_root: Path
    train: PatchDataset
    test: PatchDataset


@dataclass(frozen=True)
class FeatureMapDataset:
    """Container for one CNN feature map tensor per image."""

    maps: np.ndarray
    image_paths: list[Path]


@dataclass(frozen=True)
class SpatialPatchLayout:
    """Patch-grid metadata used for both training and evaluation."""

    patch_size: tuple[int, int]
    stride: int
    image_shape: tuple[int, int]
    positions: list[tuple[int, int]]


def find_data_root() -> Path:
    """Locate the MVTec dataset directory used by the notebook."""

    configured_root = os.environ.get("MVTEC_DATA_ROOT")
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            "MVTEC_DATA_ROOT is set but does not point to an existing directory: "
            f"{candidate}"
        )

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    candidates = [
        Path("Code/Data"),
        Path("Data"),
        Path.cwd() / "Code" / "Data",
        Path.cwd() / "Data",
        script_dir.parent / "Data",
        repo_root / "Code" / "Data",
        repo_root / "Data",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the MVTec dataset. Expected it at Code/Data or Data "
        f"relative to the current working directory: {Path.cwd()}"
    )


def maybe_limit(items: list[Path], limit: int | None) -> list[Path]:
    """Return all items when limit is None, otherwise return the first limit."""

    return items if limit is None else items[:limit]


def list_categories(data_root: Path) -> list[str]:
    """Return MVTec category names that contain train and test folders."""

    categories = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and (path / "train").exists() and (path / "test").exists():
            categories.append(path.name)
    return categories


def list_images(folder: Path) -> list[Path]:
    """List image files recursively and keep ordering deterministic."""

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_config(config: PatchExtractionConfig) -> None:
    """Reject invalid image and patch geometry early."""

    image_w, image_h = config.image_size
    patch_h, patch_w = config.patch_size

    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"image_size must be positive, got {config.image_size}")
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size must be positive, got {config.patch_size}")
    if config.stride <= 0:
        raise ValueError(f"stride must be positive, got {config.stride}")
    if patch_h > image_h or patch_w > image_w:
        raise ValueError(
            "patch_size must fit inside image_size, "
            f"got patch_size={config.patch_size}, image_size={config.image_size}"
        )
    if not 0.0 <= config.anomaly_threshold <= 1.0:
        raise ValueError(
            "anomaly_threshold must be between 0 and 1, "
            f"got {config.anomaly_threshold}"
        )
    if config.debug_sample_count <= 0:
        raise ValueError(
            "debug_sample_count must be positive, "
            f"got {config.debug_sample_count}"
        )
    if config.object_crop_mode not in {
        "none",
        "white_background_bbox",
        "dark_background_bbox",
    }:
        raise ValueError(
            "object_crop_mode must be 'none', 'white_background_bbox', "
            "or 'dark_background_bbox', "
            f"got {config.object_crop_mode!r}"
        )
    if not 0 <= config.white_background_threshold <= 255:
        raise ValueError(
            "white_background_threshold must be between 0 and 255, "
            f"got {config.white_background_threshold}"
        )
    if not 0 <= config.dark_background_threshold <= 255:
        raise ValueError(
            "dark_background_threshold must be between 0 and 255, "
            f"got {config.dark_background_threshold}"
        )
    if config.object_crop_padding < 0:
        raise ValueError(
            "object_crop_padding must be non-negative, "
            f"got {config.object_crop_padding}"
        )
    if config.yolo_obb_padding < 0:
        raise ValueError(
            "yolo_obb_padding must be non-negative, "
            f"got {config.yolo_obb_padding}"
        )
    if config.input_representation not in {"raw_pixels", "cnn_features"}:
        raise ValueError(
            "input_representation must be 'raw_pixels' or 'cnn_features', "
            f"got {config.input_representation!r}"
        )
    if config.cnn_feature_patch_size[0] <= 0 or config.cnn_feature_patch_size[1] <= 0:
        raise ValueError(
            "cnn_feature_patch_size must be positive, "
            f"got {config.cnn_feature_patch_size}"
        )
    if config.cnn_feature_stride <= 0:
        raise ValueError(
            f"cnn_feature_stride must be positive, got {config.cnn_feature_stride}"
        )
    if config.cnn_batch_size <= 0:
        raise ValueError(f"cnn_batch_size must be positive, got {config.cnn_batch_size}")
    if config.cnn_layer_names is not None and len(config.cnn_layer_names) < 2:
        raise ValueError("cnn_layer_names must contain at least two layer names.")
    if config.cnn_pca_chunk_size <= 0:
        raise ValueError(
            f"cnn_pca_chunk_size must be positive, got {config.cnn_pca_chunk_size}"
        )
    if config.cnn_pca_components is not None:
        if isinstance(config.cnn_pca_components, int):
            pca_components = (config.cnn_pca_components,)
        else:
            pca_components = tuple(config.cnn_pca_components)
        if not pca_components:
            raise ValueError("cnn_pca_components cannot be empty.")
        if any(components <= 0 for components in pca_components):
            raise ValueError(
                "cnn_pca_components must contain only positive integers, "
                f"got {config.cnn_pca_components!r}"
            )
        if config.cnn_fusion_channels is not None:
            raise ValueError(
                "cnn_pca_components and cnn_fusion_channels are mutually exclusive. "
                "Use PCA reduction instead of random channel sub-selection."
            )
        if len(pca_components) not in {1, len(_configured_cnn_layers(config))}:
            raise ValueError(
                "cnn_pca_components must be either one integer applied to every layer "
                "or one value per configured CNN layer."
            )
        if (
            len(_configured_cnn_layers(config)) > 1
            and len(pca_components) > 1
            and len(set(pca_components)) != 1
        ):
            raise ValueError(
                "Multi-layer PCA fusion requires the same reduced channel count for "
                f"every layer, got {config.cnn_pca_components!r}."
            )
    if config.cnn_fusion_channels is not None and config.cnn_fusion_channels <= 0:
        raise ValueError(
            f"cnn_fusion_channels must be positive, got {config.cnn_fusion_channels}"
        )
    if config.cnn_backbone == "YOLO11":
        if config.cnn_layer_names is not None:
            raise ValueError("YOLO11 feature extraction supports exactly one hook layer.")
        if config.yolo_target_layer_idx is None:
            raise ValueError("yolo_target_layer_idx is required when cnn_backbone='YOLO11'.")
        if config.yolo_target_layer_idx < 0:
            raise ValueError(
                f"yolo_target_layer_idx must be non-negative, got {config.yolo_target_layer_idx}."
            )
    

def fit_channel_pca(train_patches, n_components=64, batch_size=20000):
    """Fit channel PCA on a batch of per-image feature maps with shape (N, H, W, C)."""

    _, _, _, C = train_patches.shape

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    pca = IncrementalPCA(n_components=n_components)

    X = train_patches.reshape(-1, C)  # (N*H*W, C)
    if len(X) < n_components:
        raise ValueError(
            f"PCA needs at least {n_components} observations, got {len(X)}."
        )

    effective_batch_size = max(batch_size, n_components)
    start = 0
    while start < len(X):
        end = min(start + effective_batch_size, len(X))
        if 0 < len(X) - end < n_components:
            end = len(X)
        pca.partial_fit(X[start:end])
        start = end

    return pca

def transform_channel_pca(patches, pca, batch_size=20000):
    """Apply channel PCA to per-image feature maps with shape (N, H, W, C)."""

    N, H, W, C = patches.shape
    X = patches.reshape(-1, C)

    reduced_batches = []
    for start in range(0, len(X), batch_size):
        reduced_batches.append(pca.transform(X[start:start + batch_size]))

    X_reduced = np.concatenate(reduced_batches, axis=0)
    return X_reduced.reshape(N, H, W, pca.n_components_).astype(np.float32)


def _configured_pca_components(config: PatchExtractionConfig) -> tuple[int, ...] | None:
    """Return PCA components aligned to configured CNN layers."""

    if config.cnn_pca_components is None:
        return None
    if isinstance(config.cnn_pca_components, int):
        return (config.cnn_pca_components,) * len(_configured_cnn_layers(config))

    components = tuple(config.cnn_pca_components)
    if len(components) == 1:
        return components * len(_configured_cnn_layers(config))
    return components


def _apply_channel_pca_to_map(
    feature_map: np.ndarray,
    pca: IncrementalPCA,
    batch_size: int,
) -> np.ndarray:
    """Apply a fitted channel PCA to one feature map of shape (H, W, C)."""

    height, width, channels = feature_map.shape
    flattened = feature_map.reshape(height * width, channels)
    reduced_batches = []
    for start in range(0, len(flattened), batch_size):
        reduced_batches.append(pca.transform(flattened[start:start + batch_size]))
    reduced = np.concatenate(reduced_batches, axis=0)
    return reduced.reshape(height, width, pca.n_components_).astype(np.float32, copy=False)


def center_crop_on_square_canvas(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center a cropped object region on a square background canvas."""

    crop_h, crop_w = image.shape[:2]
    canvas_size = max(crop_h, crop_w)

    image_canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    mask_canvas = np.zeros((canvas_size, canvas_size), dtype=bool)

    row_offset = (canvas_size - crop_h) // 2
    col_offset = (canvas_size - crop_w) // 2
    image_canvas[row_offset:row_offset + crop_h, col_offset:col_offset + crop_w, :] = image
    mask_canvas[row_offset:row_offset + crop_h, col_offset:col_offset + crop_w] = mask
    return image_canvas, mask_canvas


def pad_to_square_black(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter side symmetrically with black pixels to make a square crop."""

    crop_h, crop_w = image.shape[:2]
    canvas_size = max(crop_h, crop_w)

    image_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    mask_canvas = np.zeros((canvas_size, canvas_size), dtype=bool)

    row_offset = (canvas_size - crop_h) // 2
    col_offset = (canvas_size - crop_w) // 2
    image_canvas[row_offset:row_offset + crop_h, col_offset:col_offset + crop_w, :] = image
    mask_canvas[row_offset:row_offset + crop_h, col_offset:col_offset + crop_w] = mask
    return image_canvas, mask_canvas


def resolve_data_root(config: PatchExtractionConfig) -> Path:
    """Return the configured data root or discover it automatically."""

    return (
        Path(config.data_root).expanduser().resolve()
        if config.data_root is not None
        else find_data_root()
    )


def resolve_category_root(config: PatchExtractionConfig) -> Path:
    """Return the category root and check that it exists."""

    data_root = resolve_data_root(config)
    category_root = data_root / config.category
    if not category_root.exists():
        available = ", ".join(list_categories(data_root))
        raise FileNotFoundError(
            f"Category '{config.category}' was not found under {data_root}. "
            f"Available categories: {available}"
        )
    return category_root


def load_rgb_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Load one MVTec image as a resized RGB tensor in [0, 1]."""

    image = Image.open(path).convert("RGB")
    image = image.resize(image_size, resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def load_rgb_image_raw(path: Path) -> np.ndarray:
    """Load one RGB image at source resolution."""

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_binary_mask(path: Path | None, image_size: tuple[int, int]) -> np.ndarray:
    """Load a resized binary anomaly mask, or zeros for normal images."""

    if path is None:
        return np.zeros((image_size[1], image_size[0]), dtype=bool)

    mask = Image.open(path).convert("L")
    mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def load_binary_mask_raw(path: Path | None, image_shape: tuple[int, int]) -> np.ndarray:
    """Load one binary mask at source resolution."""

    image_h, image_w = image_shape
    if path is None:
        return np.zeros((image_h, image_w), dtype=bool)

    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def _to_numpy_array(value) -> np.ndarray:
    """Convert tensors or array-like values to a NumPy array without importing frameworks eagerly."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def mask_path_for_test_image(category_root: Path, image_path: Path) -> Path | None:
    """Return the matching MVTec mask path for one test image."""

    defect_type = image_path.parent.name
    if defect_type == "good":
        return None

    mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask for {image_path}: expected {mask_path}")
    return mask_path


def detect_object_bbox_from_white_background(
    image: np.ndarray,
    threshold: int,
    padding: int,
) -> tuple[int, int, int, int]:
    """Return the foreground box for images with a near-white background."""

    foreground = np.any(image < threshold, axis=2)
    if not np.any(foreground):
        return 0, 0, image.shape[0], image.shape[1]

    rows, cols = np.where(foreground)
    top = max(0, int(rows.min()) - padding)
    left = max(0, int(cols.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + 1 + padding)
    right = min(image.shape[1], int(cols.max()) + 1 + padding)
    return top, left, bottom, right


def detect_object_bbox_from_dark_background(
    image: np.ndarray,
    threshold: int,
    padding: int,
) -> tuple[int, int, int, int]:
    """Return the foreground box for images with a near-dark background."""

    foreground = np.any(image > threshold, axis=2)
    if not np.any(foreground):
        return 0, 0, image.shape[0], image.shape[1]

    rows, cols = np.where(foreground)
    top = max(0, int(rows.min()) - padding)
    left = max(0, int(cols.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + 1 + padding)
    right = min(image.shape[1], int(cols.max()) + 1 + padding)
    return top, left, bottom, right


def _resolve_local_or_repo_path(path_str: str) -> Path:
    """Resolve model paths relative to cwd first and then the repository."""

    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parent.parent.parent
    return (repo_root / path).resolve()


_YOLO_OBB_DETECTOR_CACHE: dict[tuple[str, str | None], object] = {}


def _build_yolo_obb_detector(config: PatchExtractionConfig):
    """Build and cache the Ultralytics OBB detector only when requested."""

    cache_key = (config.yolo_obb_model_path, config.yolo_obb_device)
    cached = _YOLO_OBB_DETECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        YOLO = _import_ultralytics()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yolo_obb_preprocessing=True requires the 'ultralytics' package."
        ) from exc
    model_path = _resolve_local_or_repo_path(config.yolo_obb_model_path)
    model_source = str(model_path) if model_path.exists() else config.yolo_obb_model_path
    detector = YOLO(model_source)
    _YOLO_OBB_DETECTOR_CACHE[cache_key] = detector
    return detector


def _extract_best_obb_detection(
    image: np.ndarray,
    config: PatchExtractionConfig,
) -> tuple[float, float, float, float, float] | None:
    """Return the highest-confidence OBB as (cx, cy, w, h, angle_radians)."""

    detector = _build_yolo_obb_detector(config)
    results = detector.predict(
        source=image,
        device=config.yolo_obb_device,
        verbose=False,
    )
    if not results:
        return None

    obb = getattr(results[0], "obb", None)
    if obb is None:
        return None

    xywhr = getattr(obb, "xywhr", None)
    conf = getattr(obb, "conf", None)
    if xywhr is None or conf is None:
        return None

    xywhr_array = _to_numpy_array(xywhr)
    conf_array = _to_numpy_array(conf).reshape(-1)
    if xywhr_array.size == 0 or conf_array.size == 0:
        return None

    best_index = int(np.argmax(conf_array))
    cx, cy, width, height, angle = xywhr_array[best_index].tolist()
    return float(cx), float(cy), float(width), float(height), float(angle)


def _rotate_with_fill(
    image: np.ndarray,
    angle_degrees: float,
    center: tuple[float, float],
    resample: Image.Resampling,
    fillcolor,
) -> np.ndarray:
    """Rotate one image around an arbitrary center while keeping the frame fixed."""

    pil_image = Image.fromarray(image)
    rotated = pil_image.rotate(
        angle=angle_degrees,
        resample=resample,
        center=center,
        expand=False,
        fillcolor=fillcolor,
    )
    return np.asarray(rotated)


def _crop_around_center(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_width: float,
    crop_height: float,
    padding: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a padded rectangle around a floating-point center with safe clamping."""

    image_h, image_w = image.shape[:2]
    half_w = crop_width / 2.0
    half_h = crop_height / 2.0

    left = max(0, int(math.floor(center_x - half_w - padding)))
    top = max(0, int(math.floor(center_y - half_h - padding)))
    right = min(image_w, int(math.ceil(center_x + half_w + padding)))
    bottom = min(image_h, int(math.ceil(center_y + half_h + padding)))
    return image[top:bottom, left:right, ...], (top, left, bottom, right)


def preprocess_with_yolo_obb_alignment(
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    config: PatchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align, crop, and square-pad an object using the best YOLO OBB prediction."""

    detection = _extract_best_obb_detection(raw_image, config)
    if detection is None:
        return None

    center_x, center_y, box_width, box_height, angle_radians = detection
    angle_degrees = math.degrees(angle_radians)

    # Align the predicted box axes to the image axes and force the long side vertical.
    rotation_degrees = -angle_degrees if box_height >= box_width else 90.0 - angle_degrees
    rotation_center = (center_x, center_y)

    rotated_image = _rotate_with_fill(
        image=raw_image,
        angle_degrees=rotation_degrees,
        center=rotation_center,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    rotated_mask = _rotate_with_fill(
        image=(raw_mask.astype(np.uint8) * 255),
        angle_degrees=rotation_degrees,
        center=rotation_center,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    ) > 0

    if box_width > box_height:
        box_width, box_height = box_height, box_width

    cropped_image, crop_bounds = _crop_around_center(
        image=rotated_image,
        center_x=center_x,
        center_y=center_y,
        crop_width=box_width,
        crop_height=box_height,
        padding=config.yolo_obb_padding,
    )
    top, left, bottom, right = crop_bounds
    cropped_mask = rotated_mask[top:bottom, left:right]
    if cropped_image.size == 0 or cropped_mask.size == 0:
        return None

    return pad_to_square_black(cropped_image, cropped_mask)


def preprocess_image_and_mask(
    image_path: Path,
    mask_path: Path | None,
    config: PatchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load image and mask with optional object crop before resizing."""

    raw_image = load_rgb_image_raw(image_path)
    raw_mask = load_binary_mask_raw(mask_path, raw_image.shape[:2])

    if config.yolo_obb_preprocessing:
        aligned = preprocess_with_yolo_obb_alignment(
            raw_image=raw_image,
            raw_mask=raw_mask,
            config=config,
        )
        if aligned is not None:
            raw_image, raw_mask = aligned

    if not config.yolo_obb_preprocessing and config.object_crop_mode == "white_background_bbox":
        top, left, bottom, right = detect_object_bbox_from_white_background(
            image=raw_image,
            threshold=config.white_background_threshold,
            padding=config.object_crop_padding,
        )
        raw_image = raw_image[top:bottom, left:right, :]
        raw_mask = raw_mask[top:bottom, left:right]
        if config.center_cropped_object:
            raw_image, raw_mask = center_crop_on_square_canvas(raw_image, raw_mask)
    elif not config.yolo_obb_preprocessing and config.object_crop_mode == "dark_background_bbox":
        top, left, bottom, right = detect_object_bbox_from_dark_background(
            image=raw_image,
            threshold=config.dark_background_threshold,
            padding=config.object_crop_padding,
        )
        raw_image = raw_image[top:bottom, left:right, :]
        raw_mask = raw_mask[top:bottom, left:right]
        if config.center_cropped_object:
            raw_image, raw_mask = pad_to_square_black(raw_image, raw_mask)

    resized_image = Image.fromarray(raw_image).resize(
        config.image_size,
        resample=Image.Resampling.BILINEAR,
    )
    resized_mask = Image.fromarray(raw_mask.astype(np.uint8) * 255).resize(
        config.image_size,
        resample=Image.Resampling.NEAREST,
    )
    return (
        np.asarray(resized_image, dtype=np.float32) / np.float32(255.0),
        np.asarray(resized_mask, dtype=np.uint8) > 0,
    )


def extract_patches_from_image(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int],
    stride: int,
    anomaly_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract overlapping patches and derive binary patch labels from the mask."""

    patch_h, patch_w = patch_size
    image_h, image_w = image.shape[:2]
    row_positions = range(0, image_h - patch_h + 1, stride)
    col_positions = range(0, image_w - patch_w + 1, stride)
    n_patches = len(row_positions) * len(col_positions)

    patches = np.empty(
        (n_patches, patch_h, patch_w, *image.shape[2:]), dtype=np.float32
    )
    labels = np.empty(n_patches, dtype=np.int8)

    patch_index = 0
    for row in row_positions:
        for col in col_positions:
            patch = image[row:row + patch_h, col:col + patch_w, ...]
            patches[patch_index] = patch
            mask_patch = mask[row:row + patch_h, col:col + patch_w]
            labels[patch_index] = int(mask_patch.mean() >= anomaly_threshold)
            patch_index += 1

    return patches, labels


def _import_tensorflow():
    import tensorflow as tf

    return tf


def _import_torch():
    import torch

    return torch


def _import_torchvision():
    import torchvision

    return torchvision


def _import_ultralytics():
    from ultralytics import YOLO

    return YOLO


class _YOLOForwardHookExtractor:
    """Expose one intermediate YOLO feature map through a forward hook."""

    def __init__(
        self,
        model_path: str,
        target_layer_idx: int,
        device: str | None = None,
    ) -> None:
        YOLO = _import_ultralytics()
        torch = _import_torch()

        self.yolo = YOLO(model_path)
        self.model = self.yolo.model.eval()
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self.model.parameters()).device

        modules = getattr(self.model, "model", None)
        if modules is None:
            raise RuntimeError("Unexpected Ultralytics model structure: missing .model module list.")
        if target_layer_idx >= len(modules):
            raise ValueError(
                f"yolo_target_layer_idx={target_layer_idx} is out of range for "
                f"model with {len(modules)} layers."
            )

        self.target_layer_idx = target_layer_idx
        self.extracted_features = None
        modules[target_layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output) -> None:
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise RuntimeError(
                    "Hooked YOLO layer returned multiple tensors; choose a layer "
                    "whose output is a single feature map tensor."
                )
            output = output[0]
        self.extracted_features = output.detach()

    def extract(self, image: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        chw_image = np.transpose(image.astype(np.float32, copy=False), (2, 0, 1))
        image_tensor = torch.from_numpy(chw_image).unsqueeze(0).to(self.device)

        self.extracted_features = None
        with torch.no_grad():
            _ = self.model(image_tensor)

        if self.extracted_features is None:
            raise RuntimeError(
                f"YOLO hook for layer {self.target_layer_idx} did not capture any features."
            )
        if self.extracted_features.ndim != 4:
            raise RuntimeError(
                "Expected hooked YOLO features to have shape [B, C, H, W], "
                f"got {tuple(self.extracted_features.shape)}."
            )

        features = self.extracted_features[0].detach().cpu().numpy()
        return np.transpose(features, (1, 2, 0)).astype(np.float32, copy=False)


class _TorchvisionResNetHookExtractor:
    """Expose one or more intermediate torchvision ResNet feature maps via hooks."""

    def __init__(
        self,
        backbone_name: str,
        layer_names: tuple[str, ...],
        weights: str | None = "imagenet",
        device: str | None = None,
    ) -> None:
        torchvision = _import_torchvision()
        torch = _import_torch()

        self.backbone_name = backbone_name
        self.layer_names = layer_names
        self.extracted_features: dict[str, object] = {}

        model_builders = {
            "WideResNet50_2": torchvision.models.wide_resnet50_2,
            "WideResNet101_2": torchvision.models.wide_resnet101_2,
        }
        if backbone_name not in model_builders:
            raise ValueError(
                f"Unsupported torchvision ResNet backbone {backbone_name!r}."
            )

        builder = model_builders[backbone_name]
        weights_enum = None
        if weights == "imagenet":
            if backbone_name == "WideResNet50_2":
                weights_enum = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            else:
                weights_enum = torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
        elif weights is not None:
            raise ValueError(
                "For torchvision Wide ResNets, cnn_weights must be 'imagenet' or None, "
                f"got {weights!r}."
            )

        self.model = builder(weights=weights_enum).eval()
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self.model.parameters()).device
        self.preprocess = (
            weights_enum.transforms()
            if weights_enum is not None
            else None
        )

        available_modules = dict(self.model.named_modules())
        missing_layers = [name for name in layer_names if name not in available_modules]
        if missing_layers:
            example_layers = ", ".join(
                name
                for name in ("relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
                if name in available_modules
            )
            raise ValueError(
                f"One of the layers {layer_names!r} was not found in {backbone_name}. "
                f"Example available layers: {example_layers}"
            )

        for layer_name in layer_names:
            available_modules[layer_name].register_forward_hook(
                self._hook_fn(layer_name)
            )

    def _hook_fn(self, layer_name: str):
        def hook(module, inputs, output) -> None:
            if isinstance(output, (list, tuple)):
                if len(output) != 1:
                    raise RuntimeError(
                        "Hooked torchvision layer returned multiple tensors; choose a layer "
                        "whose output is a single feature map tensor."
                    )
                output = output[0]
            self.extracted_features[layer_name] = output.detach()

        return hook

    def extract(self, image: np.ndarray) -> np.ndarray:
        torch = _import_torch()

        image_uint8 = np.clip(image * np.float32(255.0), 0, 255).astype(np.uint8)
        chw_image = np.transpose(image_uint8, (2, 0, 1))
        image_tensor = torch.from_numpy(chw_image)

        if self.preprocess is not None:
            image_tensor = self.preprocess(image_tensor)
        else:
            image_tensor = image_tensor.to(dtype=torch.float32) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        self.extracted_features = {}
        with torch.no_grad():
            _ = self.model(image_tensor)

        feature_maps = []
        for layer_name in self.layer_names:
            features = self.extracted_features.get(layer_name)
            if features is None:
                raise RuntimeError(
                    f"Wide ResNet hook for layer {layer_name!r} did not capture any features."
                )
            if features.ndim != 4:
                raise RuntimeError(
                    "Expected hooked Wide ResNet features to have shape [B, C, H, W], "
                    f"got {tuple(features.shape)} for layer {layer_name!r}."
                )
            feature_maps.append(
                np.transpose(features[0].detach().cpu().numpy(), (1, 2, 0)).astype(
                    np.float32,
                    copy=False,
                )
            )

        if len(feature_maps) == 1:
            return feature_maps[0]
        return feature_maps


_FEATURE_EXTRACTOR_CACHE: dict[tuple[object, ...], object] = {}
_CNN_LAYER_PCA_CACHE: dict[PatchExtractionConfig, tuple[IncrementalPCA, ...]] = {}
_CNN_FEATURE_MAP_SHAPE_CACHE: dict[PatchExtractionConfig, tuple[int, ...]] = {}


def _configured_cnn_layers(config: PatchExtractionConfig) -> tuple[str, ...]:
    """Return selected CNN layers while preserving the single-layer API."""

    return config.cnn_layer_names or (config.cnn_layer_name,)


def _extract_raw_cnn_feature_maps(
    image: np.ndarray,
    config: PatchExtractionConfig,
) -> np.ndarray | list[np.ndarray]:
    """Extract raw CNN feature maps before any fusion or PCA post-processing."""

    if config.cnn_feature_extractor is not None:
        raw = _call_custom_feature_extractor(
            np.expand_dims(np.asarray(image, dtype=np.float32), axis=0), config
        )
        if isinstance(raw, list):
            return [layer[0] for layer in raw]
        return raw[0]

    if config.cnn_backbone == "YOLO11":
        extractor = _build_yolo_feature_extractor(config)
        return extractor.extract(image)
    if config.cnn_backbone in {"WideResNet50_2", "WideResNet101_2"}:
        extractor = _build_torchvision_feature_extractor(config)
        model_output = extractor.extract(image)
        if not isinstance(model_output, list):
            return model_output
        return model_output

    feature_model, preprocess_input = _build_tensorflow_feature_extractor(config)
    network_input = preprocess_input(np.expand_dims(image * np.float32(255.0), axis=0))
    model_output = feature_model(network_input, training=False)
    if not isinstance(model_output, (list, tuple)):
        return np.asarray(model_output.numpy()[0], dtype=np.float32)
    return [np.asarray(output.numpy()[0], dtype=np.float32) for output in model_output]


def _extract_raw_cnn_feature_maps_batch(
    images: np.ndarray,
    config: PatchExtractionConfig,
) -> np.ndarray | list[np.ndarray]:
    """Extract raw CNN maps for a batch while preserving the layer structure."""

    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError(
            "images must have shape (batch, height, width, 3), "
            f"got {images.shape}."
        )

    if config.cnn_feature_extractor is not None:
        return _call_custom_feature_extractor(images, config)

    # TensorFlow ResNet accepts a true image batch. Other extractors currently
    # expose a single-image API, so retain their behavior while still returning
    # a consistently batched result.
    if config.cnn_backbone not in {"ResNet50"}:
        per_image = [_extract_raw_cnn_feature_maps(image, config) for image in images]
        if not isinstance(per_image[0], list):
            return np.stack(per_image, axis=0).astype(np.float32, copy=False)
        return [
            np.stack([image_maps[layer_index] for image_maps in per_image], axis=0).astype(
                np.float32,
                copy=False,
            )
            for layer_index in range(len(per_image[0]))
        ]

    feature_model, preprocess_input = _build_tensorflow_feature_extractor(config)
    network_input = preprocess_input(images * np.float32(255.0))
    model_output = feature_model(network_input, training=False)
    if not isinstance(model_output, (list, tuple)):
        return np.asarray(model_output.numpy(), dtype=np.float32)
    return [np.asarray(output.numpy(), dtype=np.float32) for output in model_output]


def _fit_pca_from_training_feature_maps(
    config: PatchExtractionConfig,
) -> tuple[IncrementalPCA, ...]:
    """Fit one channel PCA per configured CNN layer using training images only."""

    components_by_layer = _configured_pca_components(config)
    if components_by_layer is None:
        raise ValueError("cnn_pca_components must be configured before fitting layer PCA.")

    category_root = resolve_category_root(config)
    image_paths = training_image_paths(category_root, config)
    if not image_paths:
        raise ValueError(f"No training images found for category {config.category!r}.")

    pcas = [
        IncrementalPCA(n_components=n_components)
        for n_components in components_by_layer
    ]
    pending_by_layer: list[np.ndarray | None] = [None] * len(pcas)

    for batch_start in range(0, len(image_paths), config.cnn_batch_size):
        batch_paths = image_paths[batch_start:batch_start + config.cnn_batch_size]
        images = np.stack(
            [
                preprocess_image_and_mask(
                    image_path=image_path,
                    mask_path=None,
                    config=config,
                )[0]
                for image_path in batch_paths
            ],
            axis=0,
        )
        raw_maps = _extract_raw_cnn_feature_maps_batch(images, config)
        raw_maps_list = raw_maps if isinstance(raw_maps, list) else [raw_maps]
        for layer_index, (pca, feature_map_batch, n_components) in enumerate(
            zip(pcas, raw_maps_list, components_by_layer, strict=True)
        ):
            # Feed images to IncrementalPCA in the original deterministic order.
            # CNN inference is batched, but PCA chunk boundaries remain independent
            # of cnn_batch_size so changing throughput does not change the model.
            for feature_map in feature_map_batch:
                flattened = feature_map.reshape(-1, feature_map.shape[-1])
                pending = pending_by_layer[layer_index]
                pending = (
                    flattened
                    if pending is None
                    else np.concatenate((pending, flattened), axis=0)
                )
                batch_size = max(config.cnn_pca_chunk_size, n_components)
                # Keep at least n_components observations buffered so the final
                # partial_fit call can never receive an invalid undersized batch.
                while len(pending) >= batch_size + n_components:
                    pca.partial_fit(pending[:batch_size])
                    pending = pending[batch_size:]
                pending_by_layer[layer_index] = pending

    for layer_index, (pca, pending, n_components) in enumerate(
        zip(pcas, pending_by_layer, components_by_layer, strict=True)
    ):
        observation_count = 0 if pending is None else len(pending)
        if observation_count < n_components:
            raise ValueError(
                f"CNN layer {layer_index} needs at least {n_components} PCA "
                f"observations, got {observation_count}."
            )
        pca.partial_fit(pending)

    return tuple(pcas)


def _ensure_layer_pcas(config: PatchExtractionConfig) -> tuple[IncrementalPCA, ...]:
    """Return cached per-layer PCA models for this config, fitting them if needed."""

    cached = _CNN_LAYER_PCA_CACHE.get(config)
    if cached is not None:
        return cached

    fitted = _fit_pca_from_training_feature_maps(config)
    _CNN_LAYER_PCA_CACHE[config] = fitted
    return fitted


def _align_and_stack_feature_maps(
    feature_maps: list[np.ndarray],
    config: PatchExtractionConfig,
) -> np.ndarray:
    """Resize multiple feature maps to a shared grid and combine them consistently."""

    target_h, target_w = feature_maps[0].shape[:2]
    tf = _import_tensorflow()
    resized_maps = []
    for feature_map in feature_maps:
        if feature_map.shape[:2] == (target_h, target_w):
            resized_maps.append(np.asarray(feature_map, dtype=np.float32, copy=False))
        else:
            resized_maps.append(
                np.asarray(
                    tf.image.resize(feature_map, (target_h, target_w), method="bilinear").numpy(),
                    dtype=np.float32,
                )
            )

    pca_components = _configured_pca_components(config)
    if pca_components is not None:
        target_channels = resized_maps[0].shape[2]
        if any(feature_map.shape[2] != target_channels for feature_map in resized_maps):
            raise ValueError(
                "Multi-layer PCA fusion currently requires the same PCA channel count "
                "for every layer so the reduced maps can be stacked together."
            )
        return np.stack(resized_maps, axis=-1).astype(np.float32, copy=False)

    first_channels = resized_maps[0].shape[2]
    if config.cnn_fusion_channels is None:
        target_channels = max(feature_map.shape[2] for feature_map in resized_maps)
    else:
        target_channels = config.cnn_fusion_channels
        if first_channels != target_channels:
            raise ValueError(
                "The first fusion layer defines the retained channel dimension; "
                f"it has {first_channels} channels but cnn_fusion_channels={target_channels}."
            )

    rng = np.random.default_rng(config.cnn_fusion_seed)
    aligned_maps = []
    for layer_name, feature_map in zip(
        _configured_cnn_layers(config), resized_maps, strict=True
    ):
        selected = feature_map
        if config.cnn_fusion_channels is not None and selected.shape[2] != target_channels:
            if selected.shape[2] < target_channels:
                raise ValueError(
                    f"Layer {layer_name!r} has only {selected.shape[2]} channels; "
                    f"cannot retain {target_channels}."
                )
            channel_indices = rng.choice(
                selected.shape[2], size=target_channels, replace=False
            )
            selected = selected[:, :, channel_indices]
        elif config.cnn_fusion_channels is None and selected.shape[2] < target_channels:
            padded = np.zeros((target_h, target_w, target_channels), dtype=np.float32)
            padded[:, :, :selected.shape[2]] = selected
            selected = padded
        aligned_maps.append(np.asarray(selected, dtype=np.float32, copy=False))

    return np.stack(aligned_maps, axis=-1).astype(np.float32, copy=False)


def _align_and_stack_feature_map_batches(
    feature_maps: list[np.ndarray],
    config: PatchExtractionConfig,
) -> np.ndarray:
    """Batched counterpart of _align_and_stack_feature_maps()."""

    if not feature_maps:
        raise ValueError("feature_maps must not be empty.")
    batch_size = feature_maps[0].shape[0]
    if any(feature_map.ndim != 4 for feature_map in feature_maps):
        raise ValueError("Each batched feature map must have shape (N, H, W, C).")
    if any(feature_map.shape[0] != batch_size for feature_map in feature_maps):
        raise ValueError("All feature-map batches must contain the same number of images.")

    target_h, target_w = feature_maps[0].shape[1:3]
    tf = _import_tensorflow()
    resized_maps = []
    for feature_map in feature_maps:
        if feature_map.shape[1:3] == (target_h, target_w):
            resized_maps.append(np.asarray(feature_map, dtype=np.float32, copy=False))
        else:
            resized_maps.append(
                np.asarray(
                    tf.image.resize(
                        feature_map,
                        (target_h, target_w),
                        method="bilinear",
                    ).numpy(),
                    dtype=np.float32,
                )
            )

    pca_components = _configured_pca_components(config)
    if pca_components is not None:
        target_channels = resized_maps[0].shape[3]
        if any(feature_map.shape[3] != target_channels for feature_map in resized_maps):
            raise ValueError(
                "Multi-layer PCA fusion currently requires the same PCA channel count "
                "for every layer so the reduced maps can be stacked together."
            )
        return np.stack(resized_maps, axis=-1).astype(np.float32, copy=False)

    first_channels = resized_maps[0].shape[3]
    if config.cnn_fusion_channels is None:
        target_channels = max(feature_map.shape[3] for feature_map in resized_maps)
    else:
        target_channels = config.cnn_fusion_channels
        if first_channels != target_channels:
            raise ValueError(
                "The first fusion layer defines the retained channel dimension; "
                f"it has {first_channels} channels but cnn_fusion_channels={target_channels}."
            )

    rng = np.random.default_rng(config.cnn_fusion_seed)
    aligned_maps = []
    for layer_name, feature_map in zip(
        _configured_cnn_layers(config), resized_maps, strict=True
    ):
        selected = feature_map
        if config.cnn_fusion_channels is not None and selected.shape[3] != target_channels:
            if selected.shape[3] < target_channels:
                raise ValueError(
                    f"Layer {layer_name!r} has only {selected.shape[3]} channels; "
                    f"cannot retain {target_channels}."
                )
            channel_indices = rng.choice(
                selected.shape[3], size=target_channels, replace=False
            )
            selected = selected[:, :, :, channel_indices]
        elif config.cnn_fusion_channels is None and selected.shape[3] < target_channels:
            padded = np.zeros(
                (batch_size, target_h, target_w, target_channels),
                dtype=np.float32,
            )
            padded[:, :, :, :selected.shape[3]] = selected
            selected = padded
        aligned_maps.append(np.asarray(selected, dtype=np.float32, copy=False))

    return np.stack(aligned_maps, axis=-1).astype(np.float32, copy=False)


def _build_tensorflow_feature_extractor(config: PatchExtractionConfig):
    """Build and cache the TensorFlow backbone and its preprocessing function."""

    layer_names = _configured_cnn_layers(config)
    cache_key = (config.cnn_backbone, layer_names, config.cnn_weights)
    cached = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tf = _import_tensorflow()
    if config.cnn_backbone == "ResNet50":
        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=config.cnn_weights,
            input_shape=(config.image_size[1], config.image_size[0], 3),
        )
        preprocess_input = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError(
            "Only ResNet50 is currently supported for cnn_features mode, "
            f"got {config.cnn_backbone!r}."
        )
    try:
        layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    except ValueError as exc:
        available_layers = ", ".join(layer.name for layer in backbone.layers[:20])
        raise ValueError(
            f"One of the layers {layer_names!r} was not found in {config.cnn_backbone}. "
            f"Example available layers: {available_layers}"
        ) from exc

    outputs = layer_outputs[0] if len(layer_outputs) == 1 else layer_outputs
    feature_model = tf.keras.Model(inputs=backbone.input, outputs=outputs)
    _FEATURE_EXTRACTOR_CACHE[cache_key] = (feature_model, preprocess_input)
    return feature_model, preprocess_input


def _build_yolo_feature_extractor(config: PatchExtractionConfig) -> _YOLOForwardHookExtractor:
    """Build and cache a YOLO hook-based feature extractor."""

    if config.yolo_target_layer_idx is None:
        raise ValueError("yolo_target_layer_idx is required when cnn_backbone='YOLO11'.")

    cache_key = (
        config.cnn_backbone,
        config.yolo_model_path,
        config.yolo_target_layer_idx,
        config.yolo_device,
    )
    cached = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    extractor = _YOLOForwardHookExtractor(
        model_path=config.yolo_model_path,
        target_layer_idx=config.yolo_target_layer_idx,
        device=config.yolo_device,
    )
    _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def _build_torchvision_feature_extractor(
    config: PatchExtractionConfig,
) -> _TorchvisionResNetHookExtractor:
    """Build and cache a torchvision Wide ResNet hook-based feature extractor."""

    layer_names = _configured_cnn_layers(config)
    cache_key = (config.cnn_backbone, layer_names, config.cnn_weights)
    cached = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        extractor = _TorchvisionResNetHookExtractor(
            backbone_name=config.cnn_backbone,
            layer_names=layer_names,
            weights=config.cnn_weights,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{config.cnn_backbone} requires torchvision, but it is not installed."
        ) from exc

    _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
    return extractor


def extract_cnn_feature_map(
    image: np.ndarray,
    config: PatchExtractionConfig,
) -> np.ndarray:
    """Convert one preprocessed RGB image into an intermediate CNN feature map."""

    return extract_cnn_feature_maps(np.expand_dims(image, axis=0), config)[0]


def extract_cnn_feature_maps(
    images: np.ndarray,
    config: PatchExtractionConfig,
) -> np.ndarray:
    """Convert a batch of preprocessed RGB images into CNN feature maps."""

    raw_feature_maps = _extract_raw_cnn_feature_maps_batch(images, config)
    if not isinstance(raw_feature_maps, list):
        if config.cnn_pca_components is None:
            return raw_feature_maps
        layer_pca = _ensure_layer_pcas(config)[0]
        return transform_channel_pca(
            raw_feature_maps,
            layer_pca,
            batch_size=config.cnn_pca_chunk_size,
        )

    feature_maps = raw_feature_maps
    if config.cnn_pca_components is not None:
        layer_pcas = _ensure_layer_pcas(config)
        feature_maps = [
            transform_channel_pca(
                feature_map,
                layer_pca,
                batch_size=config.cnn_pca_chunk_size,
            )
            for feature_map, layer_pca in zip(feature_maps, layer_pcas, strict=True)
        ]

    return _align_and_stack_feature_map_batches(feature_maps, config)


def cnn_feature_map_shape(config: PatchExtractionConfig) -> tuple[int, ...]:
    """Return the deterministic feature-map shape for the configured backbone."""

    cached = _CNN_FEATURE_MAP_SHAPE_CACHE.get(config)
    if cached is not None:
        return cached
    sample_image = np.zeros((config.image_size[1], config.image_size[0], 3), dtype=np.float32)
    shape = tuple(extract_cnn_feature_map(sample_image, config).shape)
    _CNN_FEATURE_MAP_SHAPE_CACHE[config] = shape
    return shape


def _patch_positions(
    image_shape: tuple[int, int],
    patch_size: tuple[int, int],
    stride: int,
) -> list[tuple[int, int]]:
    image_h, image_w = image_shape
    patch_h, patch_w = patch_size
    return [
        (row, col)
        for row in range(0, image_h - patch_h + 1, stride)
        for col in range(0, image_w - patch_w + 1, stride)
    ]


def _to_numpy_feature_batch(value: Any, *, name: str) -> np.ndarray:
    """Convert common framework tensor outputs to a float32 NHWC batch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(
            f"Custom CNN extractor output {name!r} must have shape "
            f"(batch, height, width, channels), got {array.shape}."
        )
    return array


def _call_custom_feature_extractor(
    images: np.ndarray,
    config: PatchExtractionConfig,
) -> np.ndarray | list[np.ndarray]:
    """Run and validate a user-provided CNN/feature extractor.

    The callable may return a NumPy array, a framework tensor, or a list/tuple
    of those. Each output must be batched NHWC. A model exposing only
    ``predict`` is accepted as well, which covers common Keras-style models.
    """

    extractor = config.cnn_feature_extractor
    if extractor is None:  # pragma: no cover - guarded by callers
        raise RuntimeError("No custom CNN feature extractor was configured.")
    if callable(extractor):
        output = extractor(images)
    elif hasattr(extractor, "predict"):
        output = extractor.predict(images, verbose=0)
    else:
        raise TypeError(
            "cnn_feature_extractor must be callable or expose predict(batch, verbose=0)."
        )
    outputs = output if isinstance(output, (list, tuple)) else [output]
    feature_maps = [
        _to_numpy_feature_batch(value, name=f"layer_{index}")
        for index, value in enumerate(outputs)
    ]
    for feature_map in feature_maps:
        if feature_map.shape[0] != images.shape[0]:
            raise ValueError(
                "Custom CNN extractor returned a different batch size: "
                f"expected {images.shape[0]}, got {feature_map.shape[0]}."
            )
    return feature_maps if isinstance(output, (list, tuple)) else feature_maps[0]


def feature_patch_layout_for_shape(
    feature_map_shape: tuple[int, ...],
    patch_size: tuple[int, int] = (1, 1),
    stride: int = 1,
) -> SpatialPatchLayout:
    """Build patch-layout metadata for a dense CNN feature map."""

    feature_h, feature_w = feature_map_shape[:2]
    return SpatialPatchLayout(
        patch_size=patch_size,
        stride=stride,
        image_shape=(feature_h, feature_w),
        positions=_patch_positions(
            image_shape=(feature_h, feature_w),
            patch_size=patch_size,
            stride=stride,
        ),
    )


def build_feature_patch_labels_from_paths(
    image_paths: list[Path],
    category_root: Path,
    config: PatchExtractionConfig,
    feature_map_shape: tuple[int, ...],
) -> np.ndarray:
    """Build feature-grid patch labels from test masks without materializing features."""

    labels_by_image = []
    feature_h, feature_w = feature_map_shape[:2]
    patch_h, patch_w = config.cnn_feature_patch_size
    positions = _patch_positions(
        image_shape=(feature_h, feature_w),
        patch_size=(patch_h, patch_w),
        stride=config.cnn_feature_stride,
    )

    for image_path in image_paths:
        _, mask = preprocess_image_and_mask(
            image_path=image_path,
            mask_path=mask_path_for_test_image(category_root, image_path),
            config=config,
        )
        feature_mask_coverage = np.asarray(
            Image.fromarray(mask.astype(np.float32)).resize(
                (feature_w, feature_h),
                resample=Image.Resampling.BOX,
            ),
            dtype=np.float32,
        )
        image_labels = np.empty(len(positions), dtype=np.int8)
        for patch_index, (row, col) in enumerate(positions):
            mask_patch = feature_mask_coverage[row:row + patch_h, col:col + patch_w]
            image_labels[patch_index] = int(mask_patch.mean() >= config.anomaly_threshold)
        labels_by_image.append(image_labels)

    return np.concatenate(labels_by_image, axis=0)


def _extract_cnn_patches_from_feature_map(
    feature_map: np.ndarray,
    mask: np.ndarray,
    config: PatchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, SpatialPatchLayout]:
    """Extract configured patches from an already-computed CNN feature map."""

    feature_h, feature_w = feature_map.shape[:2]
    patch_h, patch_w = config.cnn_feature_patch_size
    if patch_h > feature_h or patch_w > feature_w:
        raise ValueError(
            "cnn_feature_patch_size must fit inside the selected feature map, "
            f"got patch_size={config.cnn_feature_patch_size}, "
            f"feature_map_shape={feature_map.shape[:2]}"
        )
    feature_mask = np.asarray(
        Image.fromarray(mask.astype(np.float32)).resize(
            (feature_w, feature_h),
            resample=Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )
    patches, labels = extract_patches_from_image(
        image=feature_map,
        mask=feature_mask,
        patch_size=config.cnn_feature_patch_size,
        stride=config.cnn_feature_stride,
        anomaly_threshold=config.anomaly_threshold,
    )
    positions = _patch_positions(
        image_shape=feature_map.shape[:2],
        patch_size=config.cnn_feature_patch_size,
        stride=config.cnn_feature_stride,
    )
    return patches, labels, SpatialPatchLayout(
        patch_size=config.cnn_feature_patch_size,
        stride=config.cnn_feature_stride,
        image_shape=feature_map.shape[:2],
        positions=positions,
    )


def extract_representation_patches(
    image: np.ndarray,
    mask: np.ndarray,
    config: PatchExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, SpatialPatchLayout]:
    """Extract either raw-pixel patches or CNN-feature patches from one image."""

    if config.input_representation == "raw_pixels":
        patches, labels = extract_patches_from_image(
            image=image,
            mask=mask,
            patch_size=config.patch_size,
            stride=config.stride,
            anomaly_threshold=config.anomaly_threshold,
        )
        positions = _patch_positions(
            image_shape=image.shape[:2],
            patch_size=config.patch_size,
            stride=config.stride,
        )
        return patches, labels, SpatialPatchLayout(
            patch_size=config.patch_size,
            stride=config.stride,
            image_shape=image.shape[:2],
            positions=positions,
        )

    return _extract_cnn_patches_from_feature_map(
        feature_map=extract_cnn_feature_map(image, config),
        mask=mask,
        config=config,
    )


def patches_per_image_for_config(config: PatchExtractionConfig) -> int:
    """Return the deterministic number of extracted patches per image."""

    if config.input_representation == "raw_pixels":
        image_w, image_h = config.image_size
        patch_h, patch_w = config.patch_size
        stride = config.stride
    else:
        feature_map = np.zeros(cnn_feature_map_shape(config), dtype=np.float32)
        image_h, image_w = feature_map.shape[:2]
        patch_h, patch_w = config.cnn_feature_patch_size
        stride = config.cnn_feature_stride
    n_patch_rows = len(range(0, image_h - patch_h + 1, stride))
    n_patch_cols = len(range(0, image_w - patch_w + 1, stride))
    return n_patch_rows * n_patch_cols


def estimate_patch_dataset_memory_bytes(
    image_count: int,
    config: PatchExtractionConfig,
) -> int:
    """Estimate memory for one materialized patch dataset."""

    if config.input_representation == "raw_pixels":
        patch_h, patch_w = config.patch_size
        patch_value_shape = (patch_h, patch_w, 3)
    else:
        feature_map = np.zeros(cnn_feature_map_shape(config), dtype=np.float32)
        patch_h, patch_w = config.cnn_feature_patch_size
        patch_value_shape = (patch_h, patch_w, *feature_map.shape[2:])
    patches_per_image = patches_per_image_for_config(config)
    total_patches = image_count * patches_per_image
    patch_bytes = total_patches * int(np.prod(patch_value_shape)) * np.dtype(np.float32).itemsize
    label_bytes = total_patches * np.dtype(np.int8).itemsize
    index_bytes = total_patches * 2 * np.dtype(np.int32).itemsize
    return patch_bytes + label_bytes + index_bytes


def format_memory_bytes(num_bytes: int) -> str:
    """Format bytes as MiB for loader diagnostics."""

    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def build_patch_dataset_from_paths(
    image_paths: list[Path],
    category_root: Path,
    config: PatchExtractionConfig,
    include_test_masks: bool,
    split_name: str,
) -> PatchDataset:
    """Build a patch dataset with one final allocation instead of list concatenation."""

    patches_per_image = patches_per_image_for_config(config)
    sample_image, sample_mask = preprocess_image_and_mask(
        image_path=image_paths[0],
        mask_path=(mask_path_for_test_image(category_root, image_paths[0]) if include_test_masks else None),
        config=config,
    ) if image_paths else (None, None)
    if sample_image is None or sample_mask is None:
        raise ValueError(f"No image paths were provided for split {split_name!r}.")
    sample_patches, _, _ = extract_representation_patches(sample_image, sample_mask, config)
    patch_shape = sample_patches.shape[1:]
    total_patches = len(image_paths) * patches_per_image
    if config.debug_memory:
        estimated_bytes = estimate_patch_dataset_memory_bytes(len(image_paths), config)
        print(
            f"{split_name} planned allocation: "
            f"images={len(image_paths)}, patches={total_patches}, "
            f"estimated={format_memory_bytes(estimated_bytes)}"
        )

    patches = np.empty((total_patches, *patch_shape), dtype=np.float32)
    labels = np.empty(total_patches, dtype=np.int8)
    patch_image_indices = np.repeat(
        np.arange(len(image_paths), dtype=np.int32),
        patches_per_image,
    )
    patch_local_indices = np.tile(
        np.arange(patches_per_image, dtype=np.int32),
        len(image_paths),
    )

    if config.input_representation == "cnn_features":
        for batch_start in range(0, len(image_paths), config.cnn_batch_size):
            batch_paths = image_paths[batch_start:batch_start + config.cnn_batch_size]
            preprocessed = [
                preprocess_image_and_mask(
                    image_path=image_path,
                    mask_path=(
                        mask_path_for_test_image(category_root, image_path)
                        if include_test_masks
                        else None
                    ),
                    config=config,
                )
                for image_path in batch_paths
            ]
            feature_maps = extract_cnn_feature_maps(
                np.stack([image for image, _ in preprocessed], axis=0),
                config,
            )
            for batch_index, (feature_map, (_, mask)) in enumerate(
                zip(feature_maps, preprocessed, strict=True)
            ):
                image_patches, image_labels, _ = _extract_cnn_patches_from_feature_map(
                    feature_map=feature_map,
                    mask=mask,
                    config=config,
                )
                image_index = batch_start + batch_index
                start = image_index * patches_per_image
                end = start + patches_per_image
                patches[start:end] = image_patches
                labels[start:end] = image_labels
    else:
        for image_index, image_path in enumerate(image_paths):
            mask_path = (
                mask_path_for_test_image(category_root, image_path)
                if include_test_masks
                else None
            )
            image, mask = preprocess_image_and_mask(
                image_path=image_path,
                mask_path=mask_path,
                config=config,
            )
            image_patches, image_labels, _ = extract_representation_patches(
                image=image,
                mask=mask,
                config=config,
            )
            start = image_index * patches_per_image
            end = start + patches_per_image
            patches[start:end] = image_patches
            labels[start:end] = image_labels

    return PatchDataset(
        patches=patches,
        labels=labels,
        image_paths=image_paths,
        patches_per_image=patches_per_image,
        patch_image_indices=patch_image_indices,
        patch_local_indices=patch_local_indices,
    )


def build_feature_map_dataset_from_paths(
    image_paths: list[Path],
    category_root: Path,
    config: PatchExtractionConfig,
    include_test_masks: bool,
    split_name: str,
) -> FeatureMapDataset:
    """Build one dense CNN feature map per image instead of flattening into patches."""

    if config.input_representation != "cnn_features":
        raise ValueError(
            "build_feature_map_dataset_from_paths requires input_representation='cnn_features', "
            f"got {config.input_representation!r}."
        )
    if not image_paths:
        raise ValueError(f"No image paths were provided for split {split_name!r}.")

    sample_mask_path = (
        mask_path_for_test_image(category_root, image_paths[0])
        if include_test_masks
        else None
    )
    sample_image, _ = preprocess_image_and_mask(
        image_path=image_paths[0],
        mask_path=sample_mask_path,
        config=config,
    )
    sample_map = extract_cnn_feature_map(sample_image, config)
    maps = np.empty((len(image_paths), *sample_map.shape), dtype=np.float32)

    if config.debug_memory:
        estimated_bytes = maps.size * maps.dtype.itemsize
        print(
            f"{split_name} feature-map allocation: "
            f"images={len(image_paths)}, estimated={format_memory_bytes(estimated_bytes)}"
        )

    for batch_start in range(0, len(image_paths), config.cnn_batch_size):
        batch_paths = image_paths[batch_start:batch_start + config.cnn_batch_size]
        images = np.stack(
            [
                preprocess_image_and_mask(
                    image_path=image_path,
                    mask_path=(
                        mask_path_for_test_image(category_root, image_path)
                        if include_test_masks
                        else None
                    ),
                    config=config,
                )[0]
                for image_path in batch_paths
            ],
            axis=0,
        )
        batch_maps = extract_cnn_feature_maps(images, config)
        maps[batch_start:batch_start + len(batch_paths)] = batch_maps

    return FeatureMapDataset(maps=maps, image_paths=image_paths)


def training_image_paths(category_root: Path, config: PatchExtractionConfig) -> list[Path]:
    """Return limited normal training image paths for one category."""

    return maybe_limit(
        list_images(category_root / "train" / "good"),
        config.max_train_images,
    )


def iter_training_patch_batches(
    category_root: Path,
    config: PatchExtractionConfig,
    batch_size: int = 8,
):
    """Yield normal training patches by image without materializing the full dataset."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    image_paths = training_image_paths(category_root, config)
    patches_per_image = patches_per_image_for_config(config)
    sample_paths = training_image_paths(category_root, config)
    if not sample_paths:
        return
    sample_image, sample_mask = preprocess_image_and_mask(
        image_path=sample_paths[0],
        mask_path=None,
        config=config,
    )
    sample_patches, _, _ = extract_representation_patches(sample_image, sample_mask, config)
    patch_shape = sample_patches.shape[1:]

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start:batch_start + batch_size]
        batch_patches = np.empty(
            (len(batch_paths), patches_per_image, *patch_shape),
            dtype=np.float32,
        )
        preprocessed = [
            preprocess_image_and_mask(
                image_path=image_path,
                mask_path=None,
                config=config,
            )
            for image_path in batch_paths
        ]
        if config.input_representation == "cnn_features":
            feature_maps = extract_cnn_feature_maps(
                np.stack([image for image, _ in preprocessed], axis=0),
                config,
            )
            for batch_index, (feature_map, (_, mask)) in enumerate(
                zip(feature_maps, preprocessed, strict=True)
            ):
                image_patches, _, _ = _extract_cnn_patches_from_feature_map(
                    feature_map=feature_map,
                    mask=mask,
                    config=config,
                )
                batch_patches[batch_index] = image_patches
        else:
            for batch_index, (image, mask) in enumerate(preprocessed):
                image_patches, _, _ = extract_representation_patches(
                    image=image,
                    mask=mask,
                    config=config,
                )
                batch_patches[batch_index] = image_patches
        yield batch_patches


def build_training_dataset(category_root: Path, config: PatchExtractionConfig) -> PatchDataset:
    """Build training patches from normal MVTec training images only."""

    train_paths = training_image_paths(category_root, config)
    return build_patch_dataset_from_paths(
        image_paths=train_paths,
        category_root=category_root,
        config=config,
        include_test_masks=False,
        split_name="train",
    )


def build_training_feature_map_dataset(
    category_root: Path,
    config: PatchExtractionConfig,
) -> FeatureMapDataset:
    """Build training CNN feature maps directly, without flattening into patches first."""

    train_paths = training_image_paths(category_root, config)
    return build_feature_map_dataset_from_paths(
        image_paths=train_paths,
        category_root=category_root,
        config=config,
        include_test_masks=False,
        split_name="train_feature_maps",
    )


def build_test_dataset(category_root: Path, config: PatchExtractionConfig) -> PatchDataset:
    """Build test patches from good and anomalous MVTec test images."""

    test_paths = []
    good_paths = maybe_limit(
        list_images(category_root / "test" / "good"),
        config.max_test_good_images,
    )
    test_paths.extend(good_paths)

    for defect_dir in sorted((category_root / "test").iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            defect_paths = maybe_limit(
                list_images(defect_dir),
                config.max_test_anomaly_images_per_type,
            )
            test_paths.extend(defect_paths)

    return build_patch_dataset_from_paths(
        image_paths=test_paths,
        category_root=category_root,
        config=config,
        include_test_masks=True,
        split_name="test",
    )


def build_test_feature_map_dataset(
    category_root: Path,
    config: PatchExtractionConfig,
) -> FeatureMapDataset:
    """Build test CNN feature maps directly, without flattening into patches first."""

    test_paths = []
    good_paths = maybe_limit(
        list_images(category_root / "test" / "good"),
        config.max_test_good_images,
    )
    test_paths.extend(good_paths)

    for defect_dir in sorted((category_root / "test").iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            defect_paths = maybe_limit(
                list_images(defect_dir),
                config.max_test_anomaly_images_per_type,
            )
            test_paths.extend(defect_paths)

    return build_feature_map_dataset_from_paths(
        image_paths=test_paths,
        category_root=category_root,
        config=config,
        include_test_masks=True,
        split_name="test_feature_maps",
    )


def load_patch_datasets(config: PatchExtractionConfig) -> PatchDatasets:
    """Load train and test patch datasets for one MVTec category."""

    validate_config(config)
    total_start = time.perf_counter()
    category_root = resolve_category_root(config)
    timings: dict[str, float] = {}

    train_start = time.perf_counter()
    train_data = build_training_dataset(category_root, config)
    timings["train_build"] = time.perf_counter() - train_start

    test_start = time.perf_counter()
    test_data = build_test_dataset(category_root, config)
    timings["test_build"] = time.perf_counter() - test_start

    datasets = PatchDatasets(
        category_root=category_root,
        train=train_data,
        test=test_data,
    )
    if (
        config.debug_log_samples
        or config.debug_visualization
        or config.debug_anomalous_image_masks
        or config.debug_timing
        or config.debug_memory
    ):
        try:
            from .loader_debugger import run_loader_debugger
        except ImportError:
            from loader_debugger import run_loader_debugger

        run_loader_debugger(
            datasets=datasets,
            config=config,
            timings=timings if config.debug_timing else {},
            total_seconds=time.perf_counter() - total_start,
            load_rgb_image_fn=load_rgb_image,
            load_binary_mask_fn=load_binary_mask,
            mask_path_for_test_image_fn=mask_path_for_test_image,
        )
    return datasets
