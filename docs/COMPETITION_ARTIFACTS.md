# Competition artifact inventory

This private repository contains the reproducible source, tests, research
notes, and selected evaluation artifacts for the FAIR Universe Weak Lensing ML
Uncertainty Challenge.

## Archived research stages

- `artifacts/physics_field_stage/`: physics-field detector implementation,
  diagnostics, training/scoring entry points, and focused tests.
- `artifacts/nuisance_hardened_stage/`: nuisance-forward residual detector,
  fusion audit/bootstrap utilities, and its focused test.
- `artifacts/power-spectrum/seed-20260730/*.json`: configuration, validation,
  benchmark, and run provenance for the power-spectrum baseline.
- `artifacts/data-audit.json`: bounded structural audit of the competition
  arrays.

## Preserved submission

`submissions/ready_20260809/sub.zip` is the final noise-debiased power-profile
candidate produced on 2026-08-09. Its SHA-256 digest is:

```text
f2a5329ad7f9d5d9d5ab1987a8357bbe04a3fd8815e0ad6bccd7eca1a0fa7846
```

The archive contains one root-level `result.json` with 10,000 finite scores.

## Deliberately excluded

Raw organizer datasets, derived feature arrays, trained model binaries, draft
and dummy submissions, local environments, credentials, caches, and unrelated
personal documents are intentionally excluded from version control. They are
either reproducible, externally supplied, unsuitable for Git, or unrelated to
the competition source archive.
