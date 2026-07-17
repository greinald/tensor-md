from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from .Data_Loading import cnn_feature_map_shape
except ImportError:
    from Data_Loading import cnn_feature_map_shape


def _sample_patch_subset(
    patches: np.ndarray,
    labels: np.ndarray | None,
    target_label: int | None,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a small subset of flat patch indices."""

    if labels is None:
        indices = np.arange(len(patches))
    else:
        indices = np.flatnonzero(labels == target_label)

    if len(indices) == 0:
        return np.asarray([], dtype=int)

    return np.sort(rng.choice(indices, size=min(sample_count, len(indices)), replace=False))


def _sample_anomalous_test_images(test_data, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample anomalous test-image indices, not patch indices."""

    anomalous_image_indices = np.asarray(
        [
            image_index
            for image_index, image_path in enumerate(test_data.image_paths)
            if image_path.parent.name != "good"
        ],
        dtype=int,
    )
    if len(anomalous_image_indices) == 0:
        return np.asarray([], dtype=int)
    return np.sort(
        rng.choice(
            anomalous_image_indices,
            size=min(sample_count, len(anomalous_image_indices)),
            replace=False,
        )
    )


def _patch_grid_geometry(
    image_shape: tuple[int, int],
    patch_size: tuple[int, int],
    stride: int,
) -> tuple[int, int]:
    """Return patch rows and columns for one image."""

    image_h, image_w = image_shape
    patch_h, patch_w = patch_size
    n_patch_rows = len(range(0, image_h - patch_h + 1, stride))
    n_patch_cols = len(range(0, image_w - patch_w + 1, stride))
    return n_patch_rows, n_patch_cols


def _representation_geometry(config) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Return image-shape, patch-size, and stride in the active representation."""

    if config.input_representation == "raw_pixels":
        image_w, image_h = config.image_size
        return (image_h, image_w), config.patch_size, config.stride

    feature_h, feature_w, _ = cnn_feature_map_shape(config)
    return (feature_h, feature_w), config.cnn_feature_patch_size, config.cnn_feature_stride


def _local_patch_index_to_position(
    local_index: int,
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: int,
) -> tuple[int, int]:
    """Convert local patch index inside an image to top-left row/column."""

    _, n_patch_cols = _patch_grid_geometry(image_size, patch_size, stride)
    patch_row_index = local_index // n_patch_cols
    patch_col_index = local_index % n_patch_cols
    return patch_row_index * stride, patch_col_index * stride


def _extract_mask_patch_for_index(
    dataset,
    category_root: Path,
    config,
    patch_index: int,
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> tuple[np.ndarray, float]:
    """Load and crop the source mask patch for one dataset index."""

    image_index = int(dataset.patch_image_indices[patch_index])
    local_index = int(dataset.patch_local_indices[patch_index])
    image_path = dataset.image_paths[image_index]
    mask_path = mask_path_for_test_image_fn(category_root, image_path)
    image_shape, patch_size, stride = _representation_geometry(config)
    mask = load_binary_mask_fn(mask_path, (image_shape[1], image_shape[0]))
    row, col = _local_patch_index_to_position(
        local_index, image_shape, patch_size, stride
    )
    patch_h, patch_w = patch_size
    mask_patch = mask[row:row + patch_h, col:col + patch_w]
    return mask_patch, float(mask_patch.mean())


def _format_patch_origin(
    dataset,
    category_root: Path,
    config,
    patch_index: int,
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> str:
    """Return human-readable metadata for one patch index."""

    image_index = int(dataset.patch_image_indices[patch_index])
    local_index = int(dataset.patch_local_indices[patch_index])
    image_path = dataset.image_paths[image_index]
    split = image_path.parent.parent.name
    defect_type = image_path.parent.name
    image_shape, patch_size, stride = _representation_geometry(config)
    row, col = _local_patch_index_to_position(local_index, image_shape, patch_size, stride)

    parts = [
        f"idx={patch_index}",
        f"split={split}",
        f"type={defect_type}",
        f"image={image_path.name}",
        f"local_patch={local_index}",
        f"row={row}",
        f"col={col}",
        f"label={int(dataset.labels[patch_index])}",
    ]

    if split == "test":
        _, anomaly_ratio = _extract_mask_patch_for_index(
            dataset,
            category_root,
            config,
            patch_index,
            load_binary_mask_fn,
            mask_path_for_test_image_fn,
        )
        parts.append(f"mask_ratio={anomaly_ratio:.4f}")

    return " | ".join(parts)


def _log_debug_patch_samples(
    train_data,
    test_data,
    category_root: Path,
    config,
    sampled_indices: dict[str, np.ndarray],
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> None:
    """Print sampled patch metadata for debug inspection."""

    print("Debug patch samples")
    print("-------------------")

    for group_name, dataset in (
        ("train patches", train_data),
        ("test normal", test_data),
        ("test anomalous", test_data),
    ):
        print(group_name)
        indices = sampled_indices[group_name]
        if len(indices) == 0:
            print("  no samples")
            continue
        for patch_index in indices:
            print(
                "  "
                + _format_patch_origin(
                    dataset,
                    category_root,
                    config,
                    int(patch_index),
                    load_binary_mask_fn,
                    mask_path_for_test_image_fn,
                )
            )


def _array_memory_bytes(array: np.ndarray) -> int:
    return int(array.nbytes)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def _process_memory_bytes() -> int | None:
    try:
        import resource
    except ModuleNotFoundError:
        return None

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return int(rss)
    if rss < 1024 ** 3:
        return int(rss * 1024)
    return int(rss)


def _log_debug_performance(train_data, test_data, timings: dict[str, float], total_seconds: float) -> None:
    """Print timing and memory diagnostics for the loaded datasets."""

    print("Loader debug metrics")
    print("--------------------")

    if timings:
        for stage_name, seconds in timings.items():
            print(f"{stage_name}: {seconds:.2f}s")
        print(f"total: {total_seconds:.2f}s")

    print(
        "train memory: "
        f"patches={_format_bytes(_array_memory_bytes(train_data.patches))}, "
        f"labels={_format_bytes(_array_memory_bytes(train_data.labels))}, "
        f"patch_image_indices={_format_bytes(_array_memory_bytes(train_data.patch_image_indices))}, "
        f"patch_local_indices={_format_bytes(_array_memory_bytes(train_data.patch_local_indices))}"
    )
    print(
        "test memory:  "
        f"patches={_format_bytes(_array_memory_bytes(test_data.patches))}, "
        f"labels={_format_bytes(_array_memory_bytes(test_data.labels))}, "
        f"patch_image_indices={_format_bytes(_array_memory_bytes(test_data.patch_image_indices))}, "
        f"patch_local_indices={_format_bytes(_array_memory_bytes(test_data.patch_local_indices))}"
    )

    process_bytes = _process_memory_bytes()
    if process_bytes is not None:
        print(f"process rss: {_format_bytes(process_bytes)}")


def _show_debug_patch_visualization(
    train_data,
    test_data,
    category_root: Path,
    config,
    sampled_indices: dict[str, np.ndarray],
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> None:
    """Show RGB patches plus mask crops for sampled anomalous test patches."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "debug_visualization=True requires matplotlib to be installed in the active kernel."
        ) from error

    n_cols = config.debug_sample_count
    fig, axes = plt.subplots(4, n_cols, figsize=(3 * n_cols, 11))
    axes = np.atleast_2d(axes)

    groups = [
        ("train patches", train_data),
        ("test normal", test_data),
        ("test anomalous", test_data),
    ]

    for row, (group_name, dataset) in enumerate(groups):
        indices = sampled_indices[group_name]
        for col in range(n_cols):
            axis = axes[row, col]
            axis.axis("off")
            if col >= len(indices):
                if col == 0:
                    axis.set_title(f"{group_name}\nno samples")
                continue

            patch_index = int(indices[col])
            image_index = int(dataset.patch_image_indices[patch_index])
            defect_type = dataset.image_paths[image_index].parent.name
            axis.imshow(np.clip(dataset.patches[patch_index], 0.0, 1.0))

            title = f"{group_name}\nidx={patch_index}\ntype={defect_type}"
            if group_name != "train patches":
                _, anomaly_ratio = _extract_mask_patch_for_index(
                    dataset,
                    category_root,
                    config,
                    patch_index,
                    load_binary_mask_fn,
                    mask_path_for_test_image_fn,
                )
                title += f"\nmask={anomaly_ratio:.3f}"
            axis.set_title(title)

    anomalous_indices = sampled_indices["test anomalous"]
    for col in range(n_cols):
        axis = axes[3, col]
        axis.axis("off")
        if col >= len(anomalous_indices):
            if col == 0:
                axis.set_title("test anomalous masks\nno samples")
            continue

        patch_index = int(anomalous_indices[col])
        mask_patch, anomaly_ratio = _extract_mask_patch_for_index(
            test_data,
            category_root,
            config,
            patch_index,
            load_binary_mask_fn,
            mask_path_for_test_image_fn,
        )
        axis.imshow(mask_patch, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"anomalous mask\nidx={patch_index}\nmask={anomaly_ratio:.3f}")

    fig.suptitle(
        f"{config.category} | patch_size={config.patch_size} | stride={config.stride}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def _show_anomalous_test_image_masks(
    test_data,
    category_root: Path,
    config,
    sampled_image_indices: np.ndarray,
    load_rgb_image_fn,
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> None:
    """Show full anomalous test images and their resized masks."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "debug_anomalous_image_masks=True requires matplotlib to be installed in the active kernel."
        ) from error

    if len(sampled_image_indices) == 0:
        print("Anomalous test image masks")
        print("-------------------------")
        print("No anomalous test images available.")
        return

    print("Anomalous test image masks")
    print("-------------------------")

    n_cols = len(sampled_image_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    axes = np.atleast_2d(axes)

    for col, image_index in enumerate(sampled_image_indices):
        image_path = test_data.image_paths[int(image_index)]
        mask_path = mask_path_for_test_image_fn(category_root, image_path)
        image = load_rgb_image_fn(image_path, config.image_size)
        mask = load_binary_mask_fn(mask_path, config.image_size)
        mask_ratio = float(mask.mean())

        print(
            f"image_idx={int(image_index)} | type={image_path.parent.name} | "
            f"image={image_path.name} | resized_mask_ratio={mask_ratio:.4f}"
        )

        image_axis = axes[0, col]
        image_axis.imshow(np.clip(image, 0.0, 1.0))
        image_axis.axis("off")
        image_axis.set_title(
            f"image idx={int(image_index)}\n{image_path.parent.name}/{image_path.name}"
        )

        mask_axis = axes[1, col]
        mask_axis.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        mask_axis.axis("off")
        mask_axis.set_title(f"mask\nratio={mask_ratio:.3f}")

    fig.suptitle(
        f"{config.category} anomalous test images | resized to {config.image_size}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def run_loader_debugger(
    datasets,
    config,
    timings: dict[str, float],
    total_seconds: float,
    load_rgb_image_fn,
    load_binary_mask_fn,
    mask_path_for_test_image_fn,
) -> None:
    """Run enabled loader debug actions."""

    rng = np.random.default_rng(config.debug_random_seed)
    sampled_indices = {
        "train patches": _sample_patch_subset(
            datasets.train.patches, None, None, config.debug_sample_count, rng
        ),
        "test normal": _sample_patch_subset(
            datasets.test.patches, datasets.test.labels, 0, config.debug_sample_count, rng
        ),
        "test anomalous": _sample_patch_subset(
            datasets.test.patches, datasets.test.labels, 1, config.debug_sample_count, rng
        ),
    }
    sampled_anomalous_image_indices = _sample_anomalous_test_images(
        datasets.test, config.debug_sample_count, rng
    )

    if config.debug_log_samples:
        _log_debug_patch_samples(
            datasets.train,
            datasets.test,
            datasets.category_root,
            config,
            sampled_indices,
            load_binary_mask_fn,
            mask_path_for_test_image_fn,
        )

    if config.debug_visualization:
        _show_debug_patch_visualization(
            datasets.train,
            datasets.test,
            datasets.category_root,
            config,
            sampled_indices,
            load_binary_mask_fn,
            mask_path_for_test_image_fn,
        )

    if config.debug_anomalous_image_masks:
        _show_anomalous_test_image_masks(
            datasets.test,
            datasets.category_root,
            config,
            sampled_anomalous_image_indices,
            load_rgb_image_fn,
            load_binary_mask_fn,
            mask_path_for_test_image_fn,
        )

    if config.debug_timing or config.debug_memory:
        _log_debug_performance(datasets.train, datasets.test, timings, total_seconds)
