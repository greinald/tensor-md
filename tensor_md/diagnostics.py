"""Optional, non-evaluation diagnostics for detector scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def save_score_diagnostics(
    scores: np.ndarray,
    output_dir: str | Path,
    *,
    patches_per_image: int,
    grid_shape: tuple[int, int] | None = None,
    image_paths: Sequence[str | Path] | None = None,
    prefix: str = "scores",
    formats: Sequence[str] | None = None,
    make_plots: bool | None = None,
    max_heatmaps: int = 16,
) -> dict[str, str]:
    """Save score arrays, metadata, distributions, and optional heatmaps.

    This is a debugging/inspection helper only; it does not change scores or
    participate in official evaluation. Heatmaps show the detector's spatial
    score grid before any resizing to an image resolution.
    """

    if formats is None:
        requested = {"npy", "csv", "json"}
        if make_plots is not False:
            requested.update({"distribution", "heatmaps"})
    else:
        requested = {str(item).lower() for item in formats}
        if "plots" in requested:
            requested.update({"distribution", "heatmaps"})
    allowed = {"npy", "csv", "json", "tiff", "distribution", "heatmaps", "plots"}
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"Unsupported diagnostic formats: {sorted(unknown)}")

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must contain at least one value.")
    if patches_per_image <= 0 or values.size % patches_per_image:
        raise ValueError(
            "scores length must be divisible by patches_per_image: "
            f"got {values.size} and {patches_per_image}."
        )
    image_scores = values.reshape(-1, patches_per_image)
    if grid_shape is not None:
        if len(grid_shape) != 2 or np.prod(grid_shape) != patches_per_image:
            raise ValueError(
                f"grid_shape={grid_shape} does not match patches_per_image={patches_per_image}."
            )
        grid_shape = (int(grid_shape[0]), int(grid_shape[1]))

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / f"{prefix}.npy"
    if "npy" in requested:
        np.save(raw_path, image_scores)

    csv_path = destination / f"{prefix}.csv"
    paths = [str(path) for path in image_paths] if image_paths is not None else []
    if paths and len(paths) != image_scores.shape[0]:
        raise ValueError("image_paths must contain one path per scored image.")
    if "csv" in requested:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_index", "location_index", "image_path", "score"])
            for image_index, row in enumerate(image_scores):
                image_path = paths[image_index] if paths else ""
                writer.writerows(
                    (image_index, location_index, image_path, float(value))
                    for location_index, value in enumerate(row)
                )

    manifest = {
        "scores_file": raw_path.name,
        "csv_file": csv_path.name,
        "image_count": int(image_scores.shape[0]),
        "patches_per_image": int(patches_per_image),
        "grid_shape": list(grid_shape) if grid_shape is not None else None,
        "image_paths": paths,
        "image_score_max": image_scores.max(axis=1).astype(float).tolist(),
        "score_min": float(values.min()) if values.size else None,
        "score_max": float(values.max()) if values.size else None,
        "score_mean": float(values.mean()) if values.size else None,
    }
    result = {}
    if "json" in requested:
        manifest_path = destination / f"{prefix}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result["manifest"] = str(manifest_path)
    if "npy" in requested:
        result["scores"] = str(raw_path)
    if "csv" in requested:
        result["csv"] = str(csv_path)

    if "tiff" in requested:
        if grid_shape is None:
            raise ValueError("The 'tiff' format requires grid_shape=(height, width).")
        try:
            import tifffile
        except ImportError:
            tifffile = None
        from PIL import Image
        tiff_dir = destination / f"{prefix}_tiff"
        tiff_dir.mkdir(parents=True, exist_ok=True)
        for image_index, row in enumerate(image_scores):
            score_grid = row.reshape(grid_shape).astype(np.float32, copy=False)
            target = tiff_dir / f"{image_index:06d}.tiff"
            if tifffile is not None:
                tifffile.imwrite(target, score_grid)
            else:
                # Pillow can write float32 TIFFs and is part of the core package.
                Image.fromarray(score_grid, mode="F").save(target, format="TIFF")
        result["tiff_directory"] = str(tiff_dir)

    if "distribution" in requested or "heatmaps" in requested:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            result["plots_skipped"] = "matplotlib is not installed"
        else:
            if "distribution" in requested:
                histogram_path = destination / f"{prefix}_distribution.png"
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(values, bins=min(50, max(10, values.size // 10)), color="#236b8e")
                ax.set(title="Detector score distribution", xlabel="score", ylabel="count")
                fig.tight_layout()
                fig.savefig(histogram_path, dpi=160)
                plt.close(fig)
                result["distribution_plot"] = str(histogram_path)

            if "heatmaps" in requested and grid_shape is not None:
                count = min(max_heatmaps, image_scores.shape[0])
                columns = min(4, max(1, count))
                rows = int(np.ceil(count / columns))
                fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
                for index, axis in enumerate(axes.flat):
                    axis.axis("off")
                    if index >= count:
                        continue
                    axis.imshow(image_scores[index].reshape(grid_shape), cmap="magma")
                    axis.set_title(f"image {index}")
                    axis.axis("off")
                fig.tight_layout()
                heatmap_path = destination / f"{prefix}_heatmaps.png"
                fig.savefig(heatmap_path, dpi=160)
                plt.close(fig)
                result["heatmaps"] = str(heatmap_path)

    return result
