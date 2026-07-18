import numpy as np
import pytest

from tensor_md import (
    extract_cnn_feature_maps,
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
    PatchExtractionConfig,
    make_cnn_feature_extractor,
)
from tensor_md.Data_Loading import resolve_data_root
from tensor_md.patch_estimators import (
    _fit_tensor_separable_model_from_centered,
    _score_tensor_separable_model,
)


def test_separable_tensor_fit_and_score_are_finite():
    patches = np.random.default_rng(0).normal(
        size=(6, 2, 2, 3, 2),
    ).astype(np.float32)
    mean = patches.mean(axis=0)
    state = _fit_tensor_separable_model_from_centered(
        patches,
        mean,
        iterations=2,
    )
    scores = _score_tensor_separable_model(state, patches)
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def _small_patch_dataset():
    return np.random.default_rng(12).normal(size=(12, 1, 1, 2, 2)).astype(np.float32)


def test_detector_fit_score_and_save_load_round_trip(tmp_path):
    patches = _small_patch_dataset()
    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        iterations=2,
        location_fit_workers=1,
    ).fit(patches)
    scores = detector.score(patches)
    model_path = detector.save(tmp_path / "detector.pkl")
    restored = LocationAwareTensorMahalanobisDetector.load(model_path)
    np.testing.assert_allclose(restored.score(patches), scores)
    assert restored.location_means.shape == detector.location_means.shape


def test_neighborhood_detector_round_trip(tmp_path):
    patches = _small_patch_dataset()
    detector = NeighborhoodScoreLocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        grid_shape=(2, 2),
        score_neighbor_radius=1,
        iterations=2,
        location_fit_workers=1,
    ).fit(patches)
    model_path = detector.save(tmp_path / "neighborhood.pkl")
    restored = NeighborhoodScoreLocationAwareTensorMahalanobisDetector.load(model_path)
    np.testing.assert_allclose(restored.score(patches), detector.score(patches))


def test_save_requires_fitted_model(tmp_path):
    detector = LocationAwareTensorMahalanobisDetector(patches_per_image=1)
    with pytest.raises(RuntimeError, match="fit"):
        detector.save(tmp_path / "unfitted.pkl")


def test_mvtec_data_root_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MVTEC_DATA_ROOT", str(tmp_path))
    config = PatchExtractionConfig(category="dummy")
    assert resolve_data_root(config) == tmp_path.resolve()


def test_mvtec_data_root_environment_override_rejects_missing_path(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("MVTEC_DATA_ROOT", str(missing))
    with pytest.raises(FileNotFoundError, match="MVTEC_DATA_ROOT"):
        resolve_data_root(PatchExtractionConfig(category="dummy"))


def test_custom_cnn_extractor_is_used_without_framework_specific_backbone():
    images = np.random.default_rng(4).random((3, 8, 8, 3), dtype=np.float32)

    def extractor(batch):
        # A stand-in for any CNN; real models may return one map or a list.
        return batch[:, ::2, ::2, :1]

    config = PatchExtractionConfig(
        category="dummy",
        input_representation="cnn_features",
        cnn_backbone="my_custom_model",
        cnn_feature_extractor=extractor,
    )
    maps = extract_cnn_feature_maps(images, config)
    assert maps.shape == (3, 4, 4, 1)
    np.testing.assert_allclose(maps, images[:, ::2, ::2, :1])


def test_keras_convenience_adapter_accepts_direct_model():
    class FakeKerasModel:
        def __call__(self, batch, training=False):
            assert training is False
            return batch[:, ::2, ::2, :1]

    images = np.random.default_rng(7).random((2, 8, 8, 3), dtype=np.float32)
    extractor = make_cnn_feature_extractor(FakeKerasModel(), framework="keras")
    maps = extractor(images)
    assert maps.shape == (2, 4, 4, 1)
