import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from tensor_md import (
    extract_cnn_feature_maps,
    LocationAwareTensorMahalanobisDetector,
    NeighborhoodScoreLocationAwareTensorMahalanobisDetector,
    PatchExtractionConfig,
    make_cnn_feature_extractor,
    save_score_diagnostics,
    load_patch_datasets,
    load_normal_patches,
    load_image_patches,
    light_background_orientation_context,
)
from tensor_md.Data_Loading import (
    _align_and_stack_feature_map_batches,
    resolve_data_root,
)
from tensor_md.location_aware_tensor_mahalanobis_detector import (
    aggregate_regular_grid_scores,
)
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


def _score_calibration_dataset():
    rng = np.random.default_rng(7)
    patches = rng.normal(size=(24, 4, 1, 1, 3)).astype(np.float32)
    patches[:, 1] += 0.4
    patches[:, 2] *= 1.3
    return patches


def test_score_calibration_scopes_and_legacy_alias():
    patches_by_image = _score_calibration_dataset()
    patches = patches_by_image.reshape(-1, 1, 1, 3)
    scores = {}

    for mode in ("none", "zscore", "location_zscore", "global_zscore"):
        detector = LocationAwareTensorMahalanobisDetector(
            patches_per_image=4,
            iterations=2,
            mean_shrinkage=1.0,
            covariance_shrinkage=1.0,
            score_normalization=mode,
            location_fit_workers=1,
        ).fit(patches)
        scores[mode] = detector.score(patches).reshape(24, 4)

        if mode == "none":
            assert detector.score_statistics is None
        elif mode == "global_zscore":
            assert detector.score_statistics["mean"].shape == (1,)
            assert detector.fit_timing["score_normalization_scope"] == "global"
        else:
            assert detector.score_statistics["mean"].shape == (4,)
            assert detector.fit_timing["score_normalization_scope"] == "location"

    np.testing.assert_allclose(scores["zscore"], scores["location_zscore"])
    np.testing.assert_allclose(scores["location_zscore"].mean(axis=0), 0.0, atol=2e-5)
    assert abs(float(scores["global_zscore"].mean())) < 2e-5
    np.testing.assert_array_equal(
        np.argsort(scores["none"], axis=None),
        np.argsort(scores["global_zscore"], axis=None),
    )


def test_global_score_calibration_supports_streaming_fit():
    patches_by_image = _score_calibration_dataset()

    def batches():
        yield patches_by_image[:9]
        yield patches_by_image[9:17]
        yield patches_by_image[17:]

    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        iterations=2,
        mean_shrinkage=1.0,
        covariance_shrinkage=1.0,
        score_normalization="global_zscore",
        location_fit_workers=1,
    ).fit_from_patch_batches(batches)

    scores = detector.score(patches_by_image.reshape(-1, 1, 1, 3))
    assert detector.score_statistics["mean"].shape == (1,)
    assert abs(float(np.mean(scores))) < 2e-5


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


def test_local_average_covariance_target_fits_and_scores():
    patches = _small_patch_dataset()
    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        iterations=2,
        covariance_shrinkage=0.5,
        covariance_shrinkage_target="local_average",
        location_fit_workers=1,
    ).fit(patches)
    scores = detector.score(patches)
    assert np.isfinite(scores).all()
    assert detector.fit_timing["covariance_shrinkage_target"] == "local_average"


def test_local_average_target_requires_intermediate_shrinkage():
    patches = _small_patch_dataset()
    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        covariance_shrinkage=1.0,
        covariance_shrinkage_target="local_average",
    )
    with pytest.raises(ValueError, match="intermediate covariance shrinkage"):
        detector.fit(patches)


def test_fourier_conditioned_mean_requires_and_uses_image_context():
    rng = np.random.default_rng(31)
    image_count = 24
    angles = np.linspace(-np.pi, np.pi, image_count, endpoint=False)
    patches = np.empty((image_count, 1, 1, 2), dtype=np.float32)
    patches[:, 0, 0, 0] = 4.0 * np.cos(angles)
    patches[:, 0, 0, 1] = 3.0 * np.sin(angles)
    patches += rng.normal(0.0, 0.03, size=patches.shape).astype(np.float32)

    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=1,
        iterations=2,
        conditioning="fourier_mean",
        conditioning_order=2,
        conditioning_ridge=1e-3,
        location_fit_workers=1,
    )
    with pytest.raises(ValueError, match="requires context"):
        detector.fit(patches)

    dataset = SimpleNamespace(patches=patches, image_context=angles)
    detector.fit_dataset(dataset)
    scores = detector.score_dataset(dataset)
    assert scores.shape == (image_count,)
    assert np.isfinite(scores).all()
    assert detector.conditioning_mean_coefficients is not None
    assert detector.fit_timing["conditioning"] == "fourier_mean"


def test_light_background_orientation_context_tracks_rotation(tmp_path):
    horizontal = np.full((80, 80, 3), 245, dtype=np.uint8)
    horizontal[32:48, 14:66] = 35
    horizontal[26:54, 10:24] = 35
    first = tmp_path / "horizontal.png"
    second = tmp_path / "vertical.png"
    Image.fromarray(horizontal).save(first)
    Image.fromarray(np.rot90(horizontal)).save(second)

    first_angle = light_background_orientation_context(first)
    second_angle = light_background_orientation_context(second)
    # np.rot90 is counter-clockwise in array coordinates, whose row axis points
    # downward; the returned Cartesian-style image angle therefore changes by -pi/2.
    error = np.angle(np.exp(1j * (second_angle - first_angle + np.pi / 2)))
    assert abs(error) < 0.15


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


def test_regular_grid_median_pooling_rejects_isolated_peak():
    scores = np.zeros((1, 9), dtype=np.float32)
    scores[0, 4] = 100.0
    pooled = aggregate_regular_grid_scores(
        scores_by_image=scores,
        grid_shape=(3, 3),
        radius=1,
        pooling="median",
    )
    np.testing.assert_array_equal(pooled, np.zeros_like(scores))

    supported = scores.reshape(1, 3, 3).copy()
    supported[:, 0:2, 0:2] = 10.0
    pooled = aggregate_regular_grid_scores(
        supported.reshape(1, -1),
        (3, 3),
        1,
        "median",
    )
    assert pooled.reshape(1, 3, 3)[0, 0, 0] == 10.0


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


def test_multilayer_fusion_requires_equal_channels_without_reduction():
    feature_maps = [
        np.zeros((2, 4, 4, 3), dtype=np.float32),
        np.zeros((2, 4, 4, 5), dtype=np.float32),
    ]
    config = PatchExtractionConfig(cnn_layer_names=("layer_a", "layer_b"))

    with pytest.raises(ValueError, match="different channel counts"):
        _align_and_stack_feature_map_batches(feature_maps, config)


def test_random_reduction_rejects_dimension_larger_than_any_layer():
    feature_maps = [
        np.zeros((2, 4, 4, 3), dtype=np.float32),
        np.zeros((2, 4, 4, 5), dtype=np.float32),
    ]
    config = PatchExtractionConfig(
        cnn_layer_names=("layer_a", "layer_b"),
        cnn_dimensionality_reduction="random",
        cnn_reduction_dimensions=4,
    )

    with pytest.raises(ValueError, match="has only 3 channels"):
        _align_and_stack_feature_map_batches(feature_maps, config)


def test_pca_reduction_rejects_dimension_larger_than_layer(tmp_path):
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8)).save(normal_dir / "one.png")

    def extractor(batch):
        return [batch[:, ::2, ::2, :3], np.concatenate((batch, batch[..., :2]), axis=-1)]

    config = PatchExtractionConfig(
        train_image_dir=normal_dir,
        image_size=(8, 8),
        patch_size=(8, 8),
        input_representation="cnn_features",
        cnn_feature_extractor=extractor,
        cnn_layer_names=("layer_a", "layer_b"),
        cnn_dimensionality_reduction="pca",
        cnn_reduction_dimensions=4,
    )

    with pytest.raises(ValueError, match="has only 3 channels"):
        load_normal_patches(config)


def test_keras_convenience_adapter_accepts_direct_model():
    class FakeKerasModel:
        def __call__(self, batch, training=False):
            assert training is False
            return batch[:, ::2, ::2, :1]

    images = np.random.default_rng(7).random((2, 8, 8, 3), dtype=np.float32)
    extractor = make_cnn_feature_extractor(FakeKerasModel(), framework="keras")
    maps = extractor(images)
    assert maps.shape == (2, 4, 4, 1)


def test_generic_train_and_test_directories_are_supported(tmp_path):
    train_dir = tmp_path / "normal_train"
    test_dir = tmp_path / "evaluation"
    (test_dir / "normal").mkdir(parents=True)
    (test_dir / "defect").mkdir(parents=True)
    train_dir.mkdir()
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    Image.fromarray(image).save(train_dir / "train.png")
    Image.fromarray(image).save(test_dir / "normal" / "good.png")
    Image.fromarray(image).save(test_dir / "defect" / "bad.png")
    config = PatchExtractionConfig(
        category="custom",
        train_image_dir=train_dir,
        test_image_dir=test_dir,
        image_size=(8, 8),
        patch_size=(4, 4),
        stride=4,
    )
    datasets = load_patch_datasets(config)
    assert len(datasets.train.image_paths) == 1
    assert len(datasets.test.image_paths) == 2
    assert set(datasets.test.labels.tolist()) == {0, 1}


def test_normal_only_loader_needs_no_test_directory(tmp_path):
    train_dir = tmp_path / "normal_images"
    train_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8)).save(train_dir / "one.png")
    config = PatchExtractionConfig(
        category="custom",
        train_image_dir=train_dir,
        image_size=(8, 8),
        patch_size=(4, 4),
        stride=4,
    )
    dataset = load_normal_patches(config)
    assert dataset.patches.shape == (4, 4, 4, 3)
    assert np.all(dataset.labels == 0)


def test_generic_api_needs_only_image_directories(tmp_path):
    normal_dir = tmp_path / "normal"
    new_dir = tmp_path / "new"
    normal_dir.mkdir()
    new_dir.mkdir()
    image = np.full((8, 8, 3), 120, dtype=np.uint8)
    Image.fromarray(image).save(normal_dir / "normal.png")
    Image.fromarray(image).save(new_dir / "new.png")
    config = PatchExtractionConfig(
        train_image_dir=normal_dir,
        image_size=(8, 8),
        patch_size=(4, 4),
        stride=4,
    )
    normal = load_normal_patches(config)
    other = load_image_patches(new_dir, config)
    assert normal.patches.shape == other.patches.shape


def test_score_diagnostics_formats_are_selectable(tmp_path):
    result = save_score_diagnostics(
        np.arange(8, dtype=np.float32),
        tmp_path,
        patches_per_image=4,
        grid_shape=(2, 2),
        formats=("npy", "json"),
    )
    assert set(result) == {"scores", "manifest"}
    assert (tmp_path / "scores.npy").exists()
    assert (tmp_path / "scores.json").exists()
    assert not (tmp_path / "scores.csv").exists()


def test_score_diagnostics_can_write_tiff_score_grids(tmp_path):
    from PIL import Image

    result = save_score_diagnostics(
        np.arange(8, dtype=np.float32),
        tmp_path,
        patches_per_image=4,
        grid_shape=(2, 2),
        formats=("tiff",),
    )
    tiff_path = tmp_path / "scores_tiff" / "000000.tiff"
    assert result["tiff_directory"] == str(tmp_path / "scores_tiff")
    assert np.asarray(Image.open(tiff_path)).shape == (2, 2)


def test_detector_accepts_patch_dataset_objects():
    from types import SimpleNamespace

    patches = _small_patch_dataset()
    dataset = SimpleNamespace(patches=patches)
    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        iterations=2,
        location_fit_workers=1,
    ).fit(dataset)
    scores = detector.score(dataset)
    assert scores.shape == (len(patches),)


def test_fit_and_save_diagnostics_handles_train_and_test_together(tmp_path):
    from types import SimpleNamespace

    patches = _small_patch_dataset()
    train = SimpleNamespace(
        patches=patches,
        image_paths=["train-0", "train-1", "train-2"],
    )
    test = SimpleNamespace(
        patches=patches,
        image_paths=["test-0", "test-1", "test-2"],
    )
    detector = LocationAwareTensorMahalanobisDetector(
        patches_per_image=4,
        iterations=2,
        location_fit_workers=1,
    )
    result = detector.fit_and_save_diagnostics(
        train,
        test,
        tmp_path,
        formats=("npy", "json"),
    )
    assert set(result) == {"train", "test"}
    assert (tmp_path / "train" / "scores.npy").exists()
    assert (tmp_path / "test" / "scores.json").exists()
