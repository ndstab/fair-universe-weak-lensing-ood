"""Crash-resumable feature arrays with compatibility metadata."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib.format import open_memmap


def require_run_config(run_dir: str | Path, payload: dict) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    normalized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != normalized:
        raise ValueError("run directory contains an incompatible configuration")
    if not path.exists():
        path.write_text(normalized, encoding="utf-8")
    return path


@dataclass
class FeatureCache:
    values: np.memmap
    complete: np.memmap

    @classmethod
    def open(
        cls,
        run_dir: str | Path,
        role: str,
        shape: tuple[int, ...],
    ) -> "FeatureCache":
        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        values_path = root / f"{role}-log-power.npy"
        complete_path = root / f"{role}-complete.npy"
        leading_shape = shape[:-1]

        if values_path.exists():
            if not complete_path.exists():
                raise ValueError("existing feature cache is missing completion data")
            values = np.load(values_path, mmap_mode="r+")
            complete = np.load(complete_path, mmap_mode="r+")
            if values.shape != shape or complete.shape != leading_shape:
                raise ValueError("existing feature cache shape is incompatible")
            if values.dtype != np.dtype("float64") or complete.dtype != np.dtype(
                "bool"
            ):
                raise ValueError("existing feature cache dtype is incompatible")
        else:
            if complete_path.exists():
                raise ValueError("completion data exists without a feature cache")
            values = open_memmap(
                values_path,
                mode="w+",
                dtype="float64",
                shape=shape,
            )
            complete = open_memmap(
                complete_path,
                mode="w+",
                dtype="bool",
                shape=leading_shape,
            )
            complete[...] = False
            values.flush()
            complete.flush()

        return cls(values=values, complete=complete)

    def pending_indices(self) -> Iterator[tuple[int, ...]]:
        for index in np.argwhere(~self.complete):
            yield tuple(int(value) for value in index)

    def write(self, index: tuple[int, ...], features: np.ndarray) -> None:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (self.values.shape[-1],):
            raise ValueError("feature row has the wrong shape")
        if not np.all(np.isfinite(vector)):
            raise ValueError("feature row must be finite")
        self.values[index] = vector
        self.values.flush()
        self.complete[index] = True
        self.complete.flush()
