from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.wl_physics_field import (
    ObservedPatchAccessor,
    PhysicsFieldDetector,
    aggregate_patch_scores,
    deterministic_tiles,
    phase_morphology_channels,
    radial_power,
    radial_power_match,
)


def test_radial_power_match_removes_two_point_difference() -> None:
    """A broken amplitude matcher would leave hydro power distinguishable."""

    rng = np.random.default_rng(41)
    reference = rng.normal(size=(64, 64)).astype(np.float32)
    hydro_fft = np.fft.rfft2(reference)
    ky = np.fft.fftfreq(64)[:, None]
    kx = np.fft.rfftfreq(64)[None, :]
    radius = np.sqrt(ky * ky + kx * kx)
    hydro = np.fft.irfft2(hydro_fft * (1.0 + 2.0 * radius), s=(64, 64)).real.astype(
        np.float32
    )

    matched = radial_power_match(hydro, reference, bins=16)

    matched_power = radial_power(matched, bins=16)
    reference_power = radial_power(reference, bins=16)
    populated = reference_power > 1.0e-8
    np.testing.assert_allclose(
        matched_power[populated], reference_power[populated], rtol=2.0e-5, atol=2.0e-5
    )
    assert not np.allclose(radial_power(hydro, bins=16), reference_power, rtol=1.0e-2)


def test_phase_channel_is_invariant_to_positive_amplitude_rescaling() -> None:
    """The phase branch must not relearn the power-spectrum shortcut."""

    generator = torch.Generator().manual_seed(7)
    field = torch.randn((2, 1, 64, 64), generator=generator)
    mask = torch.ones_like(field)

    original = phase_morphology_channels(field, mask)
    rescaled = phase_morphology_channels(3.5 * field, mask)

    assert original.shape == (2, 4, 64, 64)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[:, 2], rescaled[:, 2], rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(original[:, 3], mask[:, 0])


def test_deterministic_tiles_are_stable_cover_boundaries_and_respect_mask() -> None:
    """A tiling regression could silently discard the map-area advantage."""

    mask = np.ones((128, 96), dtype=np.bool_)
    mask[:40, :40] = False

    first = deterministic_tiles(mask, patch_size=64, stride=32, minimum_fraction=0.75)
    second = deterministic_tiles(mask, patch_size=64, stride=32, minimum_fraction=0.75)

    np.testing.assert_array_equal(first, second)
    assert tuple(first[-1, :2].astype(int)) == (64, 32)
    assert not np.any(np.all(first[:, :2] == np.asarray([0.0, 0.0]), axis=1))
    assert np.all(first[:, 2] >= 0.75)


def test_aggregate_patch_scores_uses_each_map_independently() -> None:
    """Scores must aggregate within one map and never use test-batch statistics."""

    map_ids = np.asarray([2, 1, 2, 1, 2, 1], dtype=np.int64)
    scores = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    result = aggregate_patch_scores(map_ids, scores, map_count=3)

    assert np.isnan(result["mean"][0])
    assert result["count"].tolist() == [0, 3, 3]
    assert result["mean"][1] == pytest.approx(3.0)
    assert result["mean"][2] == pytest.approx(2.0)
    assert result["upper_quartile"][1] == pytest.approx(4.0)
    assert result["upper_quartile"][2] == pytest.approx(3.0)
    assert result["top_fraction_mean"][1] == pytest.approx(5.0)
    assert result["top_fraction_mean"][2] == pytest.approx(4.0)


def test_observed_patch_accessor_reconstructs_only_the_requested_tile() -> None:
    """A lookup bug would scramble compressed challenge pixels during training."""

    mask = np.asarray(
        [
            [True, True, False, True],
            [True, False, True, True],
            [False, True, True, True],
            [True, True, True, False],
        ],
        dtype=np.bool_,
    )
    tiles = np.asarray([[0.0, 0.0, 0.75], [2.0, 2.0, 0.75]])
    accessor = ObservedPatchAccessor(mask, tiles, patch_size=2)
    compressed = np.arange(1, int(mask.sum()) + 1, dtype=np.float32)

    first, first_mask = accessor.extract(compressed, 0)
    second, second_mask = accessor.extract(compressed, 1)

    np.testing.assert_array_equal(first, np.asarray([[1.0, 2.0], [4.0, 0.0]]))
    np.testing.assert_array_equal(first_mask, np.asarray([[True, True], [True, False]]))
    np.testing.assert_array_equal(second, np.asarray([[8.0, 9.0], [12.0, 0.0]]))
    np.testing.assert_array_equal(second_mask, np.asarray([[True, True], [True, False]]))

    all_patches, all_masks = accessor.extract_all(compressed)
    np.testing.assert_array_equal(all_patches[0], first)
    np.testing.assert_array_equal(all_patches[1], second)
    np.testing.assert_array_equal(all_masks[0], first_mask)
    np.testing.assert_array_equal(all_masks[1], second_mask)


def test_field_detector_returns_one_logit_and_embedding_per_patch() -> None:
    """The training and inference contracts require aligned patch outputs."""

    model = PhysicsFieldDetector(base_channels=16, embedding_dim=64)
    generator = torch.Generator().manual_seed(91)
    field = torch.randn((3, 1, 64, 64), generator=generator)
    mask = torch.ones_like(field)

    output = model(field, mask)

    assert model.embedding_dim == 64
    assert output["logit"].shape == (3,)
    assert output["embedding"].shape == (3, 64)
    assert torch.isfinite(output["logit"]).all()
    assert torch.isfinite(output["embedding"]).all()
