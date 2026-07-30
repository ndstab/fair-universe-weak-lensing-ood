"""Reconstruct and render individual masked convergence maps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wl_ood.data import CompetitionDataset


def reconstruct_map(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Restore one flattened valid-pixel vector to the survey mask layout."""
    vector = np.asarray(values)
    mask_array = np.asarray(mask, dtype=bool)
    expected = int(np.count_nonzero(mask_array))
    if vector.ndim != 1 or vector.size != expected:
        raise ValueError(
            f"values must be one-dimensional with {expected} valid pixels"
        )
    dtype = np.result_type(vector.dtype, np.float32)
    full = np.full(mask_array.shape, fill_value, dtype=dtype)
    full[mask_array] = vector
    return full


def save_map_preview(
    values: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    """Render one map using colour limits calculated from that map alone."""
    full = reconstruct_map(values, mask)
    valid = full[np.asarray(mask, dtype=bool)]
    vmin, vmax = np.percentile(valid, [1.0, 99.0])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 10))
    image = axis.imshow(full, cmap="magma", vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02, label="kappa")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def save_mask_preview(mask: np.ndarray, output_path: str | Path) -> Path:
    """Render the fixed survey mask."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4, 10))
    axis.imshow(mask, cmap="gray", vmin=0, vmax=1)
    axis.set_title("WIDE12H survey mask")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_fixed_previews(
    dataset: CompetitionDataset,
    output_dir: str | Path,
    *,
    training_indices: tuple[tuple[int, int], ...] = ((0, 0), (100, 255)),
    test_indices: tuple[int, ...] = (0, 9999),
) -> list[Path]:
    """Render predeclared individual maps and the survey mask."""
    root = Path(output_dir)
    outputs: list[Path] = []
    for cosmology, realization in training_indices:
        outputs.append(
            save_map_preview(
                dataset.training[cosmology, realization],
                dataset.mask,
                root / f"train-c{cosmology:03d}-r{realization:03d}.png",
                title=f"Training cosmology {cosmology}, realization {realization}",
            )
        )
    for index in test_indices:
        outputs.append(
            save_map_preview(
                dataset.test[index],
                dataset.mask,
                root / f"test-{index:05d}.png",
                title=f"Test map {index}",
            )
        )
    outputs.append(save_mask_preview(dataset.mask, root / "mask.png"))
    return outputs
