# Competition Data Audit Design

## Objective

Build a reproducible, memory-safe audit for the four Phase-2 NumPy files in
`public_data/`. The audit must confirm that the local dataset is structurally
usable, reconstruct flattened valid-pixel maps into the documented sky layout,
and create fixed individual-map previews without using aggregate public-test
information for model development.

## Observed Local Dataset

The local files total approximately 16 GB:

- `WIDE12H_bin2_2arcmin_kappa_newrealization.npy`
  - Shape: `(101, 256, 132019)`
  - Dtype: `float16`
- `WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy`
  - Shape: `(10000, 132019)`
  - Dtype: `float64`
- `WIDE12H_bin2_2arcmin_mask.npy`
  - Shape: `(1424, 176)`
  - Dtype: `bool`
  - Valid pixels: `132019`
- `label_newrealization.npy`
  - Shape: `(101, 256, 5)`
  - Dtype: `float64`

The training and test maps store only values corresponding to `True` positions
in the two-dimensional mask. A full map is reconstructed by filling those mask
positions in NumPy boolean-index order.

## Fixed Safety Rules

- Add `public_data/` to `.gitignore`; raw competition data must never be
  committed.
- Open the large training and test arrays with `numpy.load(...,
  mmap_mode="r", allow_pickle=False)`.
- Do not convert an entire large array to another dtype or load it into RAM.
- Test-set checks may verify filename, header, shape, dtype, readable chunks,
  and finiteness.
- Do not compute or record public-test means, standard deviations, quantiles,
  histograms, ranks, clusters, or fitted normalization.
- Preview only fixed, predeclared individual indices. Do not select examples
  after surveying the test population.
- Store generated reports and images under ignored `artifacts/`.
- Keep all operations read-only with respect to raw `.npy` files.

## Architecture

### Dataset registry

A `DatasetPaths` value object resolves the four required filenames under a
caller-provided directory. It reports missing files before any array is opened
and provides stable names to the rest of the package.

### Memory-mapped loader

`load_dataset()` opens all arrays with pickling disabled and memory mapping
enabled. It returns a `CompetitionDataset` object containing the four arrays.
The object does not reshape or cast the full training or test maps.

### Structural validation

The validator checks:

- exact shapes and dtypes;
- the mask's valid-pixel count against the stored feature dimension;
- all label values are finite;
- cosmological labels are constant across the 256 realizations for each
  cosmology;
- exactly 101 unique `(\Omega_m, S_8)` pairs exist;
- every training and test value is finite, scanned in bounded chunks.

The test finiteness scan returns only a pass/fail count of non-finite values. It
does not retain distributional summaries.

### File fingerprints

SHA-256 is calculated by streaming each file in fixed-size byte blocks. The
hashes identify the exact local dataset used by an experiment without reading
files into memory. The audit report records hashes, file sizes, shapes, dtypes,
and structural validation results.

### Map reconstruction

`reconstruct_map(values, mask, fill_value=np.nan)` accepts exactly one
132,019-value map and returns a `(1424, 176)` array. Valid positions contain
the original values in their original order; excluded positions contain the
fill value.

This function is independently tested with a small synthetic mask to prove the
ordering convention.

### Fixed previews

The preview command renders:

- training map `(cosmology=0, realization=0)`;
- training map `(cosmology=100, realization=255)`;
- test map `0`;
- test map `9999`;
- the binary survey mask.

Indices are fixed before looking at image contents. Each map is reconstructed
independently. Training and test colour limits are determined per displayed
map rather than from the test population.

### Command-line interface

Two commands extend the existing `wl-ood` CLI:

- `wl-ood audit-data --data-dir public_data --output artifacts/data-audit.json`
- `wl-ood preview-data --data-dir public_data --output-dir artifacts/previews`

`audit-data` performs the structural scan and fingerprints. `preview-data`
produces PNGs for the fixed indices and mask.

## Testing Strategy

Tests create tiny temporary `.npy` datasets with the same logical relationships
as the competition files. Production validation accepts an explicit schema so
tests do not need enormous arrays.

Test coverage includes:

- required-path resolution and missing-file errors;
- memory-mapped loading with pickling disabled;
- shape, dtype, mask-count, label, and finite-value validation;
- bounded chunk iteration;
- detection of a non-finite value in a later chunk;
- reconstruction ordering and fill values;
- deterministic SHA-256 streaming;
- JSON audit schema;
- fixed preview filenames and dimensions;
- CLI success and failure behavior.

The actual 16 GB files are then audited as a separate integration run after all
synthetic tests pass.

## Outputs

The milestone produces:

- tested data-loading and validation modules;
- updated CLI commands;
- `artifacts/data-audit.json`;
- five fixed PNG previews in `artifacts/previews/`;
- dataset fingerprints for future experiment records;
- documentation explaining the flattened storage layout;
- automated tests proving behavior without depending on private raw data.

## Success Criteria

- All automated tests pass from the isolated environment.
- `public_data/` is ignored and absent from Git status.
- The real-data audit completes without loading full arrays into RAM.
- The audit confirms the observed four-file schema and zero non-finite values.
- SHA-256 fingerprints and file sizes are recorded.
- All fixed previews are created and visually inspected.
- No public-test population statistic is computed or saved.

