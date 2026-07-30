"""Memory-safe access to the FAIR Universe Phase-2 NumPy arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAINING_FILENAME = "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"
TEST_FILENAME = "WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy"
MASK_FILENAME = "WIDE12H_bin2_2arcmin_mask.npy"
LABELS_FILENAME = "label_newrealization.npy"


@dataclass(frozen=True)
class DatasetSchema:
    training_shape: tuple[int, int, int]
    test_shape: tuple[int, int]
    mask_shape: tuple[int, int]
    labels_shape: tuple[int, int, int]
    training_dtype: np.dtype
    test_dtype: np.dtype
    mask_dtype: np.dtype
    labels_dtype: np.dtype
    valid_pixels: int
    unique_cosmologies: int


PHASE2_SCHEMA = DatasetSchema(
    training_shape=(101, 256, 132_019),
    test_shape=(10_000, 132_019),
    mask_shape=(1424, 176),
    labels_shape=(101, 256, 5),
    training_dtype=np.dtype("float16"),
    test_dtype=np.dtype("float64"),
    mask_dtype=np.dtype("bool"),
    labels_dtype=np.dtype("float64"),
    valid_pixels=132_019,
    unique_cosmologies=101,
)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    training: Path
    test: Path
    mask: Path
    labels: Path

    @classmethod
    def from_dir(cls, data_dir: str | Path) -> "DatasetPaths":
        root = Path(data_dir)
        return cls(
            root=root,
            training=root / TRAINING_FILENAME,
            test=root / TEST_FILENAME,
            mask=root / MASK_FILENAME,
            labels=root / LABELS_FILENAME,
        )

    def require_all(self) -> None:
        missing = [
            path.name
            for path in (self.training, self.test, self.mask, self.labels)
            if not path.is_file()
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"Missing competition data: {joined}")


@dataclass(frozen=True)
class CompetitionDataset:
    paths: DatasetPaths
    training: np.memmap
    test: np.memmap
    mask: np.memmap
    labels: np.memmap


def load_dataset(data_dir: str | Path) -> CompetitionDataset:
    """Open all competition arrays read-only without loading them into RAM."""
    paths = DatasetPaths.from_dir(data_dir)
    paths.require_all()
    options = {"mmap_mode": "r", "allow_pickle": False}
    return CompetitionDataset(
        paths=paths,
        training=np.load(paths.training, **options),
        test=np.load(paths.test, **options),
        mask=np.load(paths.mask, **options),
        labels=np.load(paths.labels, **options),
    )
