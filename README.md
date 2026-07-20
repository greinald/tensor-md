# tensor-md

`tensor-md` detects and localizes unusual image regions from normal training
images. It keeps each local observation as a tensor and scores it with a
location-aware tensor Mahalanobis model using separable covariance factors.

The package is label-free during training: provide a folder of normal images,
fit the detector, and then score images from another folder. Larger scores mean
that a patch differs more strongly from the normal variation learned at the
same spatial location.

## Installation

Install the core package from PyPI:

```bash
python -m pip install tensor-md
```

The PyPI distribution is named `tensor-md`; import it in Python as
`tensor_md`, because Python module names cannot contain hyphens.

Install the optional CNN dependencies when using the built-in feature
extractors:

```bash
python -m pip install "tensor-md[cnn]"
```

Diagnostics and notebooks are available through the `evaluation` and
`notebooks` extras.

## Quick start

The following example learns directly from image patches. The training and
scoring directories may contain PNG, JPEG, BMP, or TIFF images; no category
name, labels, or MVTec layout is required.

```python
from tensor_md import (
    LocationAwareTensorMahalanobisDetector,
    PatchExtractionConfig,
    load_image_patches,
    load_normal_patches,
)

config = PatchExtractionConfig(
    train_image_dir="data/normal",
    image_size=(256, 256),
    patch_size=(16, 16),
    stride=16,
)

normal = load_normal_patches(config)
images = load_image_patches("data/to_check", config)

detector = LocationAwareTensorMahalanobisDetector(
    patches_per_image=normal.patches_per_image,
)
detector.fit(normal)
scores = detector.score(images)

# Rows are images and columns are spatial patch locations.
score_maps = scores.reshape(len(images.image_paths), images.patches_per_image)
```

Training and scoring images must use the same configuration. Location-aware
modelling is most useful when images are approximately aligned, so the same
grid location usually represents the same object part or texture region.

## Using CNN features

Set `input_representation="cnn_features"` to model intermediate CNN features
instead of RGB patches. This example uses two built-in ResNet50 layers and
reduces both to 128 channels before stacking them as a tensor mode:

```python
config = PatchExtractionConfig(
    train_image_dir="data/normal",
    input_representation="cnn_features",
    cnn_backbone="ResNet50",
    cnn_layer_names=("conv3_block4_out", "conv4_block6_out"),
    cnn_dimensionality_reduction="pca",
    cnn_reduction_dimensions=128,
    cnn_feature_patch_size=(1, 1),
)
```

Available dimensionality-reduction settings are:

- `"pca"`: fit one channel PCA per selected layer using normal training
  descriptors.
- `"random"`: retain a reproducible random subset of channels from each layer.
- `"none"` or `None`: keep all channels.

Multiple feature maps must have the same retained channel count so they can be
stacked. Without reduction, their original channel counts must already match.
The loader raises a clear error if `cnn_reduction_dimensions` is larger than
the channel count of any selected layer.

## Supplying any CNN

The tensor detector is not tied to ResNet. Pass a callable through
`cnn_feature_extractor`. It receives an NHWC `float32` batch with values in
`[0, 1]` and returns either one NHWC feature-map batch or a list of them:

```python
def extract_features(batch):
    # Return shape: (N, H, W, C), or a list of such arrays.
    return my_model(batch)

config = PatchExtractionConfig(
    train_image_dir="data/normal",
    input_representation="cnn_features",
    cnn_feature_extractor=extract_features,
)
```

Keras and PyTorch models can also be wrapped with the convenience adapter:

```python
from tensor_md import make_cnn_feature_extractor

extractor = make_cnn_feature_extractor(model, framework="pytorch")
config = PatchExtractionConfig(
    train_image_dir="data/normal",
    input_representation="cnn_features",
    cnn_feature_extractor=extractor,
)
```

The adapter handles the common NCHW/NHWC layout conversion. A custom extractor
remains appropriate for models with unusual inputs or outputs.

## Neighbourhood scoring

The neighbourhood detector pools completed Mahalanobis scores across nearby
grid locations. This can make localization less sensitive to small spatial
movements:

```python
from tensor_md import NeighborhoodScoreLocationAwareTensorMahalanobisDetector

detector = NeighborhoodScoreLocationAwareTensorMahalanobisDetector(
    patches_per_image=normal.patches_per_image,
    grid_shape=(16, 16),
    score_neighbor_radius=1,
    score_neighbor_pooling="weighted_mean",
)
detector.fit(normal)
scores = detector.score(images)
```

The product of `grid_shape` must equal `patches_per_image`.

Set `score_neighbor_pooling="median"` to suppress isolated score spikes while
retaining responses supported by several nearby grid locations. For example,
radius one applies a 3 x 3 median window. The available pooling modes are
`"mean"`, `"max"`, `"median"`, and `"weighted_mean"`.

## Optional orientation-conditioned mean

For an elongated object whose position is stable but whose orientation changes,
the detector can model the expected feature mean as a smooth function of the
object angle. The covariance model remains location-specific and is fitted to
the residuals after subtracting that conditional mean.

```python
config = PatchExtractionConfig(
    train_image_dir="data/normal",
    test_image_dir="data/to_check",
    input_representation="cnn_features",
    image_context_mode="light_background_orientation",
)
datasets = load_patch_datasets(config)

detector = LocationAwareTensorMahalanobisDetector(
    patches_per_image=datasets.train.patches_per_image,
    conditioning="fourier_mean",
    conditioning_order=4,
    conditioning_ridge=1e-3,
)
detector.fit_dataset(datasets.train)
scores = detector.score_dataset(datasets.test)
```

Use `dark_foreground_orientation` for a bright object on a dark background.
For other geometries, supply `image_context_extractor=callable`; it receives an
image path and must return one finite scalar in radians. This option does not
rotate or crop images. It is intended only when the extracted angle has a clear,
consistent meaning for every image.

## Score diagnostics

Diagnostics are optional and do not require anomaly labels. They compare the
scores of the normal training images with the images being checked and can save
arrays, distributions, heatmaps, or floating-point TIFF score maps:

```python
artifacts = detector.fit_and_save_diagnostics(
    normal,
    images,
    "outputs/diagnostics",
    grid_shape=(16, 16),
    formats=("npy", "json", "distribution", "heatmaps", "tiff"),
)
```

Available formats are `npy`, `csv`, `json`, `tiff`, `distribution`, and
`heatmaps`. Diagnostics are for inspection; benchmark metrics still require
the benchmark's official ground-truth masks and evaluator.

## Saving a fitted detector

```python
detector.save("models/detector.pkl")

restored = LocationAwareTensorMahalanobisDetector.load(
    "models/detector.pkl"
)
scores = restored.score(images)
```

Model files use Python pickle and must only be loaded from a trusted source.
Reuse the same image preprocessing and feature extractor when creating data for
the restored detector.

## MVTec AD layout

MVTec AD is optional. If the dataset is arranged as
`<root>/<category>/train/good` and `<root>/<category>/test/...`, both splits can
be loaded together:

```python
from tensor_md import load_patch_datasets

config = PatchExtractionConfig(
    category="bottle",
    data_root="/path/to/mvtec",
    input_representation="cnn_features",
    cnn_backbone="ResNet50",
    cnn_layer_names=("conv3_block4_out", "conv4_block6_out"),
    cnn_dimensionality_reduction="pca",
    cnn_reduction_dimensions=128,
)
datasets = load_patch_datasets(config)
```

The package does not download or bundle MVTec AD.

## Release Notices

git pull --ff-only
python scripts/release.py

## License

MIT; see `LICENSE`.
