from __future__ import annotations

import numpy as np
import pytest

from scripts.build_physics_field_bank import paired_fields, rescale_paired_fields
from scripts.wl_physics_field import radial_power


def test_paired_fields_apply_identical_crops_and_d4_transforms() -> None:
    """Misaligned pair augmentation would turn cosmic variance into a fake signal."""

    base = np.arange(2 * 12 * 12, dtype=np.float32).reshape(2, 12, 12)
    changed = base + 17.0

    bank = paired_fields(
        base,
        changed,
        source_ids=np.asarray([10, 20]),
        fields=3,
        tiles_per_field=4,
        patch_size=6,
        seed=123,
    )

    assert bank["dmo"].shape == (12, 6, 6)
    assert bank["hydro"].shape == (12, 6, 6)
    np.testing.assert_allclose(bank["hydro"] - bank["dmo"], 17.0)
    np.testing.assert_array_equal(bank["field_id"], np.repeat(np.arange(3), 4))
    assert set(bank["source_id"].tolist()).issubset({10, 20})


def test_rescale_paired_fields_uses_one_dmo_derived_scale() -> None:
    """Independent hydro normalization would erase the physical response."""

    dmo = np.asarray([[[-2.0, 0.0], [0.0, 2.0]]], dtype=np.float32)
    hydro = dmo + np.asarray([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)

    scaled_dmo, scaled_hydro, metadata = rescale_paired_fields(
        dmo, hydro, target_std=0.5
    )

    assert float(np.std(scaled_dmo)) == pytest.approx(0.5, abs=1.0e-6)
    expected_scale = 0.5 / np.std(dmo)
    np.testing.assert_allclose(
        scaled_hydro - scaled_dmo, (hydro - dmo) * expected_scale, rtol=1.0e-6
    )
    assert metadata["scale"] == pytest.approx(float(expected_scale))


def test_paired_fields_can_power_match_at_the_final_crop_boundary() -> None:
    """Matching before a crop would reintroduce an exploitable two-point residual."""

    rng = np.random.default_rng(52)
    dmo = rng.normal(size=(2, 16, 16)).astype(np.float32)
    hydro = np.square(dmo) - 0.3 * dmo

    bank = paired_fields(
        dmo,
        hydro,
        source_ids=np.asarray([1, 2]),
        fields=2,
        tiles_per_field=3,
        patch_size=8,
        seed=4,
        power_match_after_crop=True,
        radial_bins=4,
    )

    for base, changed in zip(bank["dmo"], bank["hydro"]):
        np.testing.assert_allclose(
            radial_power(changed, bins=4),
            radial_power(base, bins=4),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
