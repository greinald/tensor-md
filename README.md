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
    load_patch_datasets,
)
```

The detector is fitted using normal patches only. It stores a location-specific
mean and separable covariance model and produces a scalar score for each test
patch. The neighbourhood subclass optionally pools the already-computed scores
on the spatial grid.

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
