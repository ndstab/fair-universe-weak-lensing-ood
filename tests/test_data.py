from pathlib import Path

import numpy as np
import pytest

from wl_ood.data import DatasetPaths, load_dataset


FILENAMES = {
    "training": "WIDE12H_bin2_2arcmin_kappa_newrealization.npy",
    "test": "WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy",
    "mask": "WIDE12H_bin2_2arcmin_mask.npy",
    "labels": "label_newrealization.npy",
}


def write_tiny_dataset(root: Path) -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    training = np.arange(2 * 3 * 3, dtype=np.float16).reshape(2, 3, 3)
    test = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
    labels = np.zeros((2, 3, 5), dtype=np.float64)
    labels[0, :, :2] = [0.2, 0.7]
    labels[1, :, :2] = [0.3, 0.8]
    np.save(root / FILENAMES["training"], training)
    np.save(root / FILENAMES["test"], test)
    np.save(root / FILENAMES["mask"], mask)
    np.save(root / FILENAMES["labels"], labels)


def test_paths_report_missing_files(tmp_path: Path) -> None:
    paths = DatasetPaths.from_dir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing competition data"):
        paths.require_all()


def test_loader_opens_every_array_as_read_only_memmap(tmp_path: Path) -> None:
    write_tiny_dataset(tmp_path)
    dataset = load_dataset(tmp_path)

    assert isinstance(dataset.training, np.memmap)
    assert isinstance(dataset.test, np.memmap)
    assert isinstance(dataset.mask, np.memmap)
    assert isinstance(dataset.labels, np.memmap)
    assert dataset.training.shape == (2, 3, 3)
    assert dataset.test.shape == (4, 3)
    assert dataset.mask.shape == (2, 3)
    assert dataset.labels.shape == (2, 3, 5)
    assert not dataset.training.flags.writeable
    assert not dataset.test.flags.writeable
    assert not dataset.mask.flags.writeable
    assert not dataset.labels.flags.writeable
