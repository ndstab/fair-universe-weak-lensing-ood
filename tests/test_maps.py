from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from tests.test_data import write_tiny_dataset
from wl_ood.data import load_dataset
from wl_ood.maps import (
    generate_fixed_previews,
    reconstruct_map,
    save_map_preview,
)


def test_reconstruct_map_preserves_boolean_index_order() -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    result = reconstruct_map(np.array([10.0, 20.0, 30.0]), mask)
    np.testing.assert_array_equal(result[mask], [10.0, 20.0, 30.0])
    assert np.all(np.isnan(result[~mask]))


def test_reconstruct_map_rejects_wrong_value_count() -> None:
    mask = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError, match="2 valid pixels"):
        reconstruct_map(np.array([1.0]), mask)


def test_save_map_preview_creates_png(tmp_path: Path) -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    output = save_map_preview(
        np.array([1.0, 2.0, 3.0]),
        mask,
        tmp_path / "map.png",
        title="Tiny map",
    )
    image = mpimg.imread(output)
    assert output.is_file()
    assert image.ndim == 3


def test_generate_fixed_previews_with_explicit_indices(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_tiny_dataset(data_dir)
    outputs = generate_fixed_previews(
        load_dataset(data_dir),
        tmp_path / "previews",
        training_indices=((0, 0), (1, 2)),
        test_indices=(0, 3),
    )
    assert [path.name for path in outputs] == [
        "train-c000-r000.png",
        "train-c001-r002.png",
        "test-00000.png",
        "test-00003.png",
        "mask.png",
    ]
    assert all(path.is_file() for path in outputs)
