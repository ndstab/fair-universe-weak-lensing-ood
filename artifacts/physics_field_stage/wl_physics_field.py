#!/usr/bin/env python3
"""Field-level utilities for paired hydrodynamic weak-lensing detection."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SHAPE_NOISE_STD = 0.4 / math.sqrt(2.0 * 30.0 * 2.0**2)
TRAIN_SIGNAL_STD = 0.01135636
NOISY_DATA_STD = math.sqrt(SHAPE_NOISE_STD**2 + TRAIN_SIGNAL_STD**2)


def _radial_index(shape: tuple[int, int], bins: int) -> tuple[np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    radius = np.sqrt(ky * ky + kx * kx)
    edges = np.linspace(0.0, float(radius.max()) + 1.0e-12, bins + 1)
    index = np.clip(np.digitize(radius, edges) - 1, 0, bins - 1)
    count = np.bincount(index.ravel(), minlength=bins).astype(np.float64)
    return index, count


def radial_power(values: np.ndarray, *, bins: int = 16) -> np.ndarray:
    """Return the mean Fourier power in fixed radial-frequency bins."""

    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"expected a two-dimensional field, got {field.shape}")
    index, count = _radial_index(field.shape, bins)
    transformed = np.fft.rfft2(field - field.mean())
    summed = np.bincount(
        index.ravel(), weights=np.square(np.abs(transformed)).ravel(), minlength=bins
    )
    return summed / np.maximum(count, 1.0)


def radial_power_match(
    values: np.ndarray, reference: np.ndarray, *, bins: int = 16
) -> np.ndarray:
    """Match radial Fourier power while retaining the input Fourier phases."""

    field = np.asarray(values, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if field.shape != target.shape or field.ndim != 2:
        raise ValueError(f"field/reference shape mismatch: {field.shape} != {target.shape}")
    index, count = _radial_index(field.shape, bins)
    field_fft = np.fft.rfft2(field - field.mean())
    target_fft = np.fft.rfft2(target - target.mean())
    field_power = np.bincount(
        index.ravel(), weights=np.square(np.abs(field_fft)).ravel(), minlength=bins
    ) / np.maximum(count, 1.0)
    target_power = np.bincount(
        index.ravel(), weights=np.square(np.abs(target_fft)).ravel(), minlength=bins
    ) / np.maximum(count, 1.0)
    scale = np.sqrt((target_power + 1.0e-30) / (field_power + 1.0e-30))
    matched = np.fft.irfft2(field_fft * scale[index], s=field.shape).real
    return (matched + target.mean()).astype(np.float32)


def _masked_blur(field: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
    kernel = torch.ones((1, 1, size, size), device=field.device, dtype=field.dtype)
    kernel /= float(size * size)
    padding = size // 2
    numerator = F.conv2d(field * mask, kernel, padding=padding)
    denominator = F.conv2d(mask, kernel, padding=padding).clamp_min(1.0e-4)
    return numerator / denominator


def phase_morphology_channels(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build amplitude, band-pass, phase-only, and validity channels."""

    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError(f"expected field shape (B,1,H,W), got {tuple(field.shape)}")
    if mask.shape != field.shape:
        raise ValueError(f"field/mask shape mismatch: {tuple(field.shape)} != {tuple(mask.shape)}")
    weight = mask.to(field.dtype)
    count = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (field * weight).sum(dim=(2, 3), keepdim=True) / count
    centered = (field - mean) * weight
    fine = _masked_blur(centered, weight, 3)
    coarse = _masked_blur(centered, weight, 9)
    band = (fine - coarse) * weight

    transformed = torch.fft.rfft2(centered)
    phase_spectrum = transformed / transformed.abs().clamp_min(1.0e-7)
    phase = torch.fft.irfft2(phase_spectrum, s=centered.shape[-2:]).real * weight
    phase_mean = phase.sum(dim=(2, 3), keepdim=True) / count
    phase_var = ((phase - phase_mean).square() * weight).sum(
        dim=(2, 3), keepdim=True
    ) / count
    phase = (phase - phase_mean) / phase_var.sqrt().clamp_min(1.0e-6)
    phase *= weight
    return torch.cat(
        (fine / NOISY_DATA_STD, band / NOISY_DATA_STD, phase, weight), dim=1
    )


def _axis_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size > length:
        raise ValueError(f"patch size {patch_size} exceeds axis length {length}")
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def deterministic_tiles(
    mask: np.ndarray,
    *,
    patch_size: int = 64,
    stride: int = 32,
    minimum_fraction: float = 0.75,
) -> np.ndarray:
    """Return stable row-major ``(y, x, observed_fraction)`` map tiles."""

    observed = np.asarray(mask, dtype=np.bool_)
    if observed.ndim != 2:
        raise ValueError(f"expected a two-dimensional mask, got {observed.shape}")
    if stride < 1 or not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("invalid stride or minimum fraction")
    rows = []
    for y in _axis_starts(observed.shape[0], patch_size, stride):
        for x in _axis_starts(observed.shape[1], patch_size, stride):
            fraction = float(observed[y : y + patch_size, x : x + patch_size].mean())
            if fraction >= minimum_fraction:
                rows.append((float(y), float(x), fraction))
    if not rows:
        raise ValueError("no tiles satisfy the observed-fraction threshold")
    return np.asarray(rows, dtype=np.float64)


def aggregate_patch_scores(
    map_ids: np.ndarray,
    scores: np.ndarray,
    *,
    map_count: int,
    top_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    """Aggregate patch evidence independently within each input map."""

    identifiers = np.asarray(map_ids, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if identifiers.shape != values.shape or identifiers.ndim != 1:
        raise ValueError("map identifiers and scores must be aligned vectors")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top fraction must lie in (0, 1]")
    output = {
        "count": np.zeros(map_count, dtype=np.int64),
        "mean": np.full(map_count, np.nan, dtype=np.float64),
        "upper_quartile": np.full(map_count, np.nan, dtype=np.float64),
        "top_fraction_mean": np.full(map_count, np.nan, dtype=np.float64),
    }
    for map_id in range(map_count):
        selected = values[identifiers == map_id]
        output["count"][map_id] = len(selected)
        if not len(selected):
            continue
        output["mean"][map_id] = float(selected.mean())
        output["upper_quartile"][map_id] = float(np.quantile(selected, 0.75))
        count = max(1, int(math.ceil(len(selected) * top_fraction)))
        output["top_fraction_mean"][map_id] = float(np.partition(selected, -count)[-count:].mean())
    return output


class ObservedPatchAccessor:
    """Extract dense patches directly from challenge-compressed map rows."""

    def __init__(
        self,
        mask: np.ndarray,
        tiles: np.ndarray,
        *,
        patch_size: int = 64,
    ) -> None:
        observed = np.asarray(mask, dtype=np.bool_)
        positions = np.asarray(tiles, dtype=np.float64)
        if observed.ndim != 2 or positions.ndim != 2 or positions.shape[1] < 2:
            raise ValueError("invalid mask or tile table")
        lookup = np.full(observed.shape, -1, dtype=np.int64)
        lookup[observed] = np.arange(int(observed.sum()), dtype=np.int64)
        self.n_observed = int(observed.sum())
        self.patch_size = int(patch_size)
        self.lookup = []
        self.patch_masks = []
        for y_value, x_value in positions[:, :2]:
            y, x = int(y_value), int(x_value)
            patch_lookup = lookup[y : y + patch_size, x : x + patch_size]
            if patch_lookup.shape != (patch_size, patch_size):
                raise ValueError(f"tile {(y, x)} exceeds mask boundary")
            self.lookup.append(patch_lookup.copy())
            self.patch_masks.append(patch_lookup >= 0)
        self.lookup = np.asarray(self.lookup, dtype=np.int64)
        self.patch_masks = np.asarray(self.patch_masks, dtype=np.bool_)

    def extract(self, compressed: np.ndarray, tile_id: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(compressed)
        if values.ndim != 1 or len(values) != self.n_observed:
            raise ValueError(
                f"expected one compressed map with {self.n_observed} pixels, got {values.shape}"
            )
        indices = self.lookup[int(tile_id)]
        valid = self.patch_masks[int(tile_id)]
        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        patch[valid] = values[indices[valid]].astype(np.float32, copy=False)
        return patch, valid.copy()

    def extract_all(self, compressed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized extraction of every frozen tile from one compressed map."""

        values = np.asarray(compressed)
        if values.ndim != 1 or len(values) != self.n_observed:
            raise ValueError(
                f"expected one compressed map with {self.n_observed} pixels, got {values.shape}"
            )
        safe = np.maximum(self.lookup, 0)
        patches = values[safe].astype(np.float32, copy=False)
        patches = np.where(self.patch_masks, patches, 0.0).astype(np.float32, copy=False)
        return patches, self.patch_masks.copy()


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.layers = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class _DownsampleStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 4, stride=2, padding=1),
            _ResidualBlock(output_channels),
            _ResidualBlock(output_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class PhysicsFieldDetector(nn.Module):
    """Compact multiscale CNN over power, band-pass, and phase-only fields."""

    def __init__(self, *, base_channels: int = 32, embedding_dim: int = 192) -> None:
        super().__init__()
        if base_channels % 8:
            raise ValueError("base channels must be divisible by eight")
        self.embedding_dim = int(embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(4, base_channels, 5, padding=2),
            _ResidualBlock(base_channels),
        )
        self.stages = nn.Sequential(
            _DownsampleStage(base_channels, 2 * base_channels),
            _DownsampleStage(2 * base_channels, 4 * base_channels),
            _DownsampleStage(4 * base_channels, 8 * base_channels),
        )
        pooled = 16 * base_channels
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Linear(pooled, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, field: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        channels = phase_morphology_channels(field, mask)
        encoded = self.stages(self.stem(channels))
        pooled = torch.cat(
            (encoded.mean(dim=(2, 3)), encoded.amax(dim=(2, 3))), dim=1
        )
        embedding = self.embedding(pooled)
        return {
            "logit": self.classifier(embedding).squeeze(1),
            "embedding": embedding,
        }
