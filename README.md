# FAIR Universe Weak-Lensing OoD Research

> **PRIVATE COMPETITION WORKSPACE:** Do not publish code, methodology, data,
> trained artifacts, scores, or submissions during the competition.

This repository develops independently evaluated OoD scores for Phase 2 of the
FAIR Universe Weak Lensing ML Uncertainty Challenge.

## Current milestone

The package reproduces the public scoring formula, creates deterministic
result-submission archives, and audits the local competition arrays without
loading them fully into memory. It does not yet train an OoD model.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Codabench uses `codalab/codalab-legacy:py39` to score uploaded JSON results.
Our model code runs locally; only `result.json` is submitted.

## Verify

```bash
.venv/bin/python -m pytest
```

## Create a plumbing-only dummy submission

```bash
.venv/bin/wl-ood make-dummy \
  --output submissions/dummy-seed-20260727.zip \
  --seed 20260727
```

Dummy scores test file formatting only and are not a scientific method.

## Competition data

Place the four organizer-provided `.npy` files in `public_data/`. This directory
is ignored by Git.

Although the documented sky grid is `1424 × 176`, each stored map contains only
the 132,019 valid mask pixels. The loader reconstructs individual full maps
using the boolean survey mask and never loads the complete training or test
arrays into RAM.

Run the structural audit:

```bash
.venv/bin/wl-ood audit-data \
  --data-dir public_data \
  --output artifacts/data-audit.json
```

Create the fixed individual-map previews:

```bash
.venv/bin/wl-ood preview-data \
  --data-dir public_data \
  --output-dir artifacts/previews
```

The audit verifies test structure and finiteness but deliberately does not
calculate test-wide distribution statistics.

## Power-spectrum baseline

The first scientific baseline is CPU-only. It reproduces the organizer's
10-bin angular power-spectrum and best-fit chi-squared method with deterministic
training noise, resumable feature caches, and a training-only validation split.

Run the fixed benchmark:

```bash
.venv/bin/wl-ood benchmark-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

Extract training features and validate the InD tail:

```bash
.venv/bin/wl-ood validate-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

After validation succeeds, fit all training simulations and create the
submission:

```bash
.venv/bin/wl-ood run-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730 \
  --submission submissions/ps-baseline-seed-20260730.zip
```

The public-test command scores every map independently. Cached test features
only make interrupted batch execution resumable; no test-population statistic
is fitted or recorded. Generated artifacts are ignored by Git. Do not interpret
or submit a result unless `validation-report.json` and `run-manifest.json`
exist.

## Research rules

- One test map is scored without access to other test maps.
- Test-set statistics never fit preprocessing, calibration, or ensembles.
- Every experiment records its source revision, configuration, seed, local
  evidence, artifacts, and conclusion.
- AI assistance is disclosed in `docs/AI_USAGE.md`.
