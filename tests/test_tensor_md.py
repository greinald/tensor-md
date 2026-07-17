import numpy as np

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
