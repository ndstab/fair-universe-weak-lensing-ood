# FAIR Universe Weak-Lensing OoD Research

> **PRIVATE COMPETITION WORKSPACE:** Do not publish code, methodology, data,
> trained artifacts, scores, or submissions during the competition.

This repository develops independently evaluated OoD scores for Phase 2 of the
FAIR Universe Weak Lensing ML Uncertainty Challenge.

## Current milestone

The initial package reproduces the public scoring formula and creates clean,
deterministic result-submission archives. It does not yet train an OoD model.

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

## Research rules

- One test map is scored without access to other test maps.
- Test-set statistics never fit preprocessing, calibration, or ensembles.
- Every experiment records its source revision, configuration, seed, local
  evidence, artifacts, and conclusion.
- AI assistance is disclosed in `docs/AI_USAGE.md`.
