import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from nuisance_forward_residual_ood import (
    ForwardResidualModel,
    ResidualScorer,
    ViewSpec,
    central_difference_jacobian,
    frozen_convex_blend,
    nuisance_hardened_quadratic,
    validate,
)


def test_nuisance_hardened_quadratic_ignores_tangent_displacements() -> None:
    """A wrong nuisance estimate must not look like an OoD residual."""
    precision = np.eye(3, dtype=np.float64)
    jacobians = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    residuals = np.asarray(
        [
            [0.0, 0.0, 4.0],
            [7.0, -3.0, 4.0],
        ],
        dtype=np.float64,
    )

    scores = nuisance_hardened_quadratic(
        residuals,
        jacobians,
        precision,
        ridge=0.0,
    )

    np.testing.assert_allclose(scores, [16.0, 16.0], atol=1.0e-12)


def test_central_difference_jacobian_recovers_local_parameter_tangents() -> None:
    """A swapped or missing derivative would project out the wrong physics."""
    parameters = np.asarray([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float64)

    def predictor(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                values[:, 0] ** 2 + 2.0 * values[:, 1],
                values[:, 0] * values[:, 1],
            ]
        )

    jacobians = central_difference_jacobian(predictor, parameters, step=1.0e-4)

    expected = np.asarray(
        [
            [[4.0, 2.0], [3.0, 2.0]],
            [[-2.0, 2.0], [4.0, -1.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(jacobians, expected, rtol=1.0e-8, atol=1.0e-8)


def test_residual_scorer_exposes_a_tangent_invariant_calibrated_energy() -> None:
    """The production scorer must actually use the nuisance projection."""
    rng = np.random.default_rng(1701)
    calibration = rng.normal(size=(256, 3))
    scorer = ResidualScorer().fit(calibration)
    jacobians = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    residuals = np.asarray(
        [[0.0, 0.0, 4.0], [7.0, -3.0, 4.0]],
        dtype=np.float64,
    )

    scores = scorer.score(residuals, jacobians)["nuisance_hardened_chi2"]

    np.testing.assert_allclose(scores[0], scores[1], rtol=1.0e-10, atol=1.0e-10)


def test_forward_model_returns_jacobian_at_its_final_profiled_labels() -> None:
    """The scorer needs tangents at the label point used for its residual."""
    rng = np.random.default_rng(812)
    labels = rng.normal(size=(128, 5)).astype(np.float32)
    features = np.column_stack(
        [
            labels,
            labels[:, 0] ** 2,
            labels[:, 0] * labels[:, 1],
            labels[:, 3] ** 2,
        ]
    ).astype(np.float32)
    model = ForwardResidualModel(
        components=6,
        trees=8,
        seed=91,
        ridge_alpha=1.0e-6,
        polynomial_degree=2,
        refine_neighbors=0,
    ).fit(features, labels)

    residuals, predicted_labels, jacobians = model.residual_labels_and_jacobian(
        features[:4]
    )

    assert residuals.shape == (4, 6)
    assert predicted_labels.shape == (4, 5)
    assert jacobians.shape == (4, 6, 5)
    normalized = np.asarray(
        model.label_scaler.transform(predicted_labels), dtype=np.float64
    )
    offset = np.zeros_like(normalized)
    offset[:, 0] = 1.0e-3
    upper = model.forward.predict(model.polynomial.transform(normalized + offset))
    lower = model.forward.predict(model.polynomial.transform(normalized - offset))
    expected_first_tangent = (upper - lower) / (2.0e-3)
    np.testing.assert_allclose(
        jacobians[:, :, 0], expected_first_tangent, rtol=1.0e-4, atol=1.0e-5
    )


def test_fold_validation_reports_the_nuisance_hardened_candidate(
    tmp_path: Path,
) -> None:
    """A correct projector is useless unless the frozen audit evaluates it."""
    rng = np.random.default_rng(662)
    groups = np.repeat(np.arange(8), 8)
    labels = rng.normal(size=(len(groups), 5)).astype(np.float32)
    x_a = np.column_stack(
        [labels, labels[:, 0] ** 2, labels[:, 1] * labels[:, 4], labels[:, 3] ** 2]
    ).astype(np.float32)
    x_b = x_a + rng.normal(0.0, 0.03, size=x_a.shape).astype(np.float32)
    proxy_features = x_b + rng.normal(0.0, 0.03, size=x_b.shape).astype(np.float32)
    proxy_features[:, -1] += 1.5
    source_index = np.concatenate(
        [group * 256 + np.arange(8) for group in range(8)]
    )
    proxy = {
        "features": proxy_features,
        "source_index": source_index,
        "proxy_type": np.repeat("orthogonal_shift", len(groups)),
    }
    args = SimpleNamespace(
        folds=2,
        max_folds=1,
        trees=8,
        seed=111,
        ridge_alpha=1.0e-3,
        polynomial_degree=2,
        refine_neighbors=0,
    )
    view = ViewSpec(
        "synthetic",
        np.ones(x_a.shape[1], dtype=np.bool_),
        components=6,
    )

    scores_output = tmp_path / "scores.npz"
    report = validate(
        args,
        x_a,
        x_b,
        labels,
        groups,
        [view],
        proxy,
        scores_output=scores_output,
    )

    assert "synthetic_nuisance_hardened_chi2" in report["summary"]
    saved = np.load(scores_output)
    assert saved["fold0_ind_synthetic_nuisance_hardened_chi2"].shape == (32,)
    assert saved["fold0_proxy_synthetic_nuisance_hardened_chi2"].shape == (32,)


def test_frozen_convex_blend_uses_the_predeclared_candidate_weight() -> None:
    """A reversed weight would silently replace the proven incumbent."""
    incumbent = np.asarray([1.0, 2.0], dtype=np.float64)
    candidate = np.asarray([3.0, 6.0], dtype=np.float64)

    blended = frozen_convex_blend(incumbent, candidate, candidate_weight=0.25)

    np.testing.assert_allclose(blended, [1.5, 3.0], atol=0.0)
    with pytest.raises(ValueError, match="between zero and one"):
        frozen_convex_blend(incumbent, candidate, candidate_weight=1.1)
