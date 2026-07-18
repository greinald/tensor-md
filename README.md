# tensor-md

Location-aware tensor Mahalanobis anomaly detection for tensor-valued CNN
patch descriptors. The package contains the reusable tensor detector, its
separable covariance estimators, and the MVTec patch data loader. Exploratory
notebooks, the official evaluator, and the thesis are kept in the repository
but are not required to import the core package.

## Install from source

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[evaluation,cnn]"
```

For notebook support, use `.[evaluation,cnn,notebooks]`. The complete
environment is also documented in `REPRODUCIBILITY.md` and `environment.yml`.

## Public API

```python
from tensor_md import (
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
    PatchExtractionConfig,
    extract_cnn_feature_maps,
    load_patch_datasets,
    load_normal_patches,
    load_image_patches,
    make_cnn_feature_extractor,
)
```

The detector is fitted using normal patches only. It stores a location-specific
mean and separable covariance model and produces a scalar score for each test
patch. The neighbourhood subclass optionally pools the already-computed scores
on the spatial grid.

The image loader is not tied to one CNN. Pass any callable (or model exposing
`predict`) through `cnn_feature_extractor`; it receives a float32 image batch
with shape `(batch, height, width, 3)` and values in `[0, 1]`, and returns one
feature-map batch or a list of feature-map batches in `(batch, height, width,
channels)` format. PCA, spatial alignment, fusion, and tensor scoring then use
those maps exactly as they do for the built-in adapters. PyTorch models can be
wrapped with a small adapter that converts their NCHW output to NHWC.

```python
def my_cnn(batch):
    with torch.no_grad():
        return torch_model(to_nchw(batch)).permute(0, 2, 3, 1)

config = PatchExtractionConfig(
    category="bottle",
    data_root="/path/to/mvtec",
    input_representation="cnn_features",
    cnn_feature_extractor=my_cnn,
)
```

ResNet50, WideResNet, and YOLO adapters remain available as conveniences; they
are not requirements of the tensor detector.

The loader can also use ordinary folders instead of an MVTec category:

```python
config = PatchExtractionConfig(
    category="custom",
    train_image_dir="data/normal_train",
    test_image_dir="data/test",  # subfolders named normal/good are label 0
    input_representation="cnn_features",
    cnn_feature_extractor=extractor,
)
datasets = load_patch_datasets(config)
```

Training images are all treated as normal. Test images in `normal` or `good`
subfolders receive label 0; images in other test subfolders receive label 1.

For unsupervised use, no test split is needed:

```python
normal = load_normal_patches(
    PatchExtractionConfig(
        train_image_dir="data/normal_images",
        input_representation="cnn_features",
        cnn_feature_extractor=extractor,
    )
)
detector.fit(normal)
```

The detector learns only from these normal patches. Later images are scored
with `load_image_patches(...)` and `detector.score(...)`; no labels are
required:

```python
other = load_image_patches("data/images_to_check", config)
scores = detector.score(other)
```

Optional diagnostics can be saved in only the formats a user wants:

```python
detector.save_score_diagnostics(
    scores,
    "outputs/debug",
    grid_shape=(8, 8),
    formats=("npy", "json", "distribution", "heatmaps"),
)
```

Available formats are `npy`, `csv`, `json`, `distribution`, and `heatmaps`.
These files are for inspection only and do not replace the official evaluator.

For convenience, Keras and PyTorch models can be adapted without writing a
layout-conversion wrapper:

```python
extractor = make_cnn_feature_extractor(model, framework="pytorch")
config = PatchExtractionConfig(
    category="bottle",
    input_representation="cnn_features",
    cnn_feature_extractor=extractor,
)
```

Fitted detectors can be saved and restored, including all fitted means,
covariance factors, shrinkage state, and score-calibration statistics:

```python
detector.fit(train_patches)
detector.save("models/bottle.pkl")

restored = LocationAwareTensorMahalanobisDetector.load("models/bottle.pkl")
scores = restored.score(test_patches)
```

Model files use Python pickle and must only be loaded from a trusted source.

The MVTec dataset is not bundled with the package. See `REPRODUCIBILITY.md`
for the expected layout and the official evaluator command. The loader accepts
an explicit `PatchExtractionConfig(data_root=...)` or the `MVTEC_DATA_ROOT`
environment variable.

## Release

Build and test a distribution locally:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Publish to TestPyPI first, then PyPI:

```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Use a PyPI API token through Twine's credential prompt or `~/.pypirc`; never
commit the token to the repository.

## License

MIT; see `LICENSE`.
