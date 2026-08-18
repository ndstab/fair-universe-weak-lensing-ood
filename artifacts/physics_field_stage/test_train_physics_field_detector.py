from __future__ import annotations

import numpy as np
import torch

from scripts.train_physics_field_detector import (
    aggregate_paired_fields,
    paired_d4_batch,
    physics_training_loss,
    promotion_decision,
)


def test_aggregate_paired_fields_keeps_dmo_and_hydro_fields_aligned() -> None:
    """A grouping bug would invalidate the low-FPR cross-simulation audit."""

    field_id = np.asarray([10, 10, 30, 30], dtype=np.int64)
    dmo = np.asarray([0.0, 2.0, 4.0, 6.0])
    hydro = np.asarray([4.0, 6.0, 8.0, 10.0])

    result = aggregate_paired_fields(field_id, dmo, hydro)

    np.testing.assert_array_equal(result["field_id"], np.asarray([10, 30]))
    np.testing.assert_allclose(result["dmo_mean"], np.asarray([1.0, 5.0]))
    np.testing.assert_allclose(result["hydro_mean"], np.asarray([5.0, 9.0]))
    np.testing.assert_allclose(result["dmo_tail"], np.asarray([2.0, 6.0]))
    np.testing.assert_allclose(result["hydro_tail"], np.asarray([6.0, 10.0]))


def test_promotion_requires_every_cross_suite_direction_and_noise_stability() -> None:
    """A high mean must not hide a failed physics suite or unstable noise view."""

    passing = {
        "kappatng_to_flamingo": {"low_fpr": 0.66},
        "flamingo_to_kappatng": {"low_fpr": 0.63},
    }
    failed_family = {
        "kappatng_to_flamingo": {"low_fpr": 0.82},
        "flamingo_to_kappatng": {"low_fpr": 0.39},
    }

    accepted = promotion_decision(passing, cross_noise_correlation=0.72)
    rejected_family = promotion_decision(failed_family, cross_noise_correlation=0.72)
    rejected_noise = promotion_decision(passing, cross_noise_correlation=0.20)

    assert accepted["passed"] is True
    assert rejected_family["passed"] is False
    assert "minimum_cross_suite_low_fpr" in rejected_family["failed_gates"]
    assert rejected_noise["passed"] is False
    assert "cross_noise_correlation" in rejected_noise["failed_gates"]


def test_physics_training_loss_rewards_paired_hydro_ordering_and_domain_alignment() -> None:
    """The optimization objective must prefer hydro above paired DMO evidence."""

    aligned = torch.zeros((2, 4))
    shifted = torch.ones((2, 4))
    good = physics_training_loss(
        dmo_logit=torch.asarray([-2.0, -1.5]),
        hydro_logit=torch.asarray([2.0, 1.5]),
        nominal_logit=torch.asarray([-1.0, -0.5]),
        dmo_embedding=aligned,
        nominal_embedding=aligned,
    )
    bad = physics_training_loss(
        dmo_logit=torch.asarray([2.0, 1.5]),
        hydro_logit=torch.asarray([-2.0, -1.5]),
        nominal_logit=torch.asarray([1.0, 0.5]),
        dmo_embedding=aligned,
        nominal_embedding=shifted,
    )

    assert good["loss"] < bad["loss"]
    assert good["domain_alignment"] == 0.0
    assert bad["domain_alignment"] > 0.0


def test_paired_d4_batch_preserves_pixel_aligned_physics_residual() -> None:
    """External DMO/hydro members must receive exactly the same augmentation."""

    base = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    changed = base + 17.0
    mask = torch.ones_like(base)
    generator = torch.Generator().manual_seed(22)

    transformed_base, transformed_changed, transformed_mask = paired_d4_batch(
        base, changed, mask, generator
    )

    torch.testing.assert_close(transformed_changed - transformed_base, torch.full_like(base, 17.0))
    torch.testing.assert_close(transformed_mask, mask)
