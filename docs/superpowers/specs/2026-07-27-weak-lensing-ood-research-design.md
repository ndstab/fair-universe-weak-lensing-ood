# Weak-Lensing OoD Competition Research Design

## Objective

Build a reproducible method that assigns an out-of-distribution (OoD) score to each Phase-2 weak-lensing test map. A larger score must mean that the map is less compatible with the physical simulation family represented by the training data.

The practical goal is to perform strongly on the public leaderboard and generalize to the organizers' final holdout set. The scientific goal is to detect when a simulated cosmological model may be unsafe to use for interpreting an observation.

## Fixed Competition Constraints

- Process each test map independently. A score may depend on that map and artifacts fitted from training data, but not on other test maps.
- Preserve the original ordering of the 10,000 test samples.
- Output one finite floating-point OoD score per test map in `result.json`.
- Treat allowed cosmological, nuisance, random-realization, mask, and noise variation as in-distribution.
- Do not use public-test population statistics, clustering, rank normalization, or test-time fitting.
- Keep code and methodology private during the competition.
- Maintain a record of how AI tools contribute to the work.
- Make every final result reproducible from source code, configuration, fixed seeds, and documented data.

## Research Principles

### Baseline before novelty

Reproduce the official power-spectrum baseline before evaluating more complicated ideas. This validates data loading, noise handling, feature extraction, fitting, calibration, scoring, and submission generation.

### One controlled change at a time

Every experiment changes one clearly stated component. Each run records its hypothesis, configuration, local metrics, runtime, and Codabench result when submitted.

### Generalization before leaderboard optimization

The public test set is used for occasional external checks, not as training data. Method selection primarily uses held-out InD data and a diverse suite of controlled pseudo-OoD shifts.

### Physics and machine learning are complementary

Physics-derived summaries provide interpretable and resource-efficient signals. Learned representations are added where they capture spatial information that those summaries miss.

### The plan is intentionally adaptable

The named candidate methods are not obligations. New approaches may be added, and weak approaches may be removed, when repeatable evidence supports the change. Competition rules, honest validation, independent inference, and reproducibility remain fixed.

## Research Workflow

### Stage 1: Project and environment setup

Create a private, version-controlled project with pinned dependencies, data-path configuration, experiment records, an AI-usage log, and clear separation between source data and generated artifacts.

Success means that another authorized researcher can recreate the environment and understand where data, code, configurations, tests, models, and submissions belong.

### Stage 2: Data audit

Load the full arrays using memory mapping so the 16 GB development machine does not load unnecessary copies. Verify shapes, dtypes, label ordering, mask values, finite-value ranges, disk requirements, and the association between 101 cosmologies and 256 realizations.

Visualize representative maps, masked regions, label distributions, and noise realizations. Confirm the organizer's exact noise and mask convention before training models.

Success means that the dataset structure and preprocessing assumptions are tested rather than inferred.

### Stage 3: Local scorer and submission path

Implement the official mean-TPR metric at 100 logarithmically spaced FPR values from 0.001 to 0.05. Validate submission length, ordering, numeric finiteness, JSON structure, and ZIP contents.

Submit a dummy result to confirm Codabench registration and the complete upload path.

Success means that a deterministic command can turn 10,000 scores into a valid submission archive.

### Stage 4: Official power-spectrum baseline

Reproduce the organizer workflow:

1. Add realistic observation noise to training maps.
2. Calculate masked power-spectrum summaries.
3. Estimate expected summaries and covariance across allowed realizations.
4. Find the best-fitting allowed cosmology for each map.
5. Measure residual disagreement with a chi-squared statistic.
6. Calibrate that statistic using held-out InD examples.
7. Generate independent scores for all test maps.

The target is to understand and approach the published public score of 0.2143. Differences from the organizer result must be investigated and documented rather than hidden.

### Stage 5: Honest validation

Because no labeled OoD training examples exist, validation has two parts.

First, held-out genuine InD data measures false alarms. Splits include unseen realizations and held-out cosmological regions. Training-derived normalization and calibration are fitted separately within every split.

Second, a pseudo-OoD stress suite measures sensitivity to several distinct violations, including controlled noise mismatch, smoothing, scale-dependent power changes, spatial rearrangement, and held-out parameter regions.

No single pseudo-OoD family determines the winning method. A candidate must maintain a clean InD tail and improve across multiple shift families and random seeds.

### Stage 6: Additional physical summaries

Evaluate complementary features one family at a time:

- one-point pixel distributions and moments;
- peak counts and peak heights;
- multiscale filtered statistics;
- morphology of dense and empty regions;
- wavelet or scattering summaries;
- other higher-order statistics justified by a specific hypothesis.

Each feature is tested alone and in combination with the power spectrum. Retain it only when ablations show stable complementary value.

### Stage 7: Learned representations

After the statistical pipeline is trustworthy, evaluate the organizer autoencoder and selected learned alternatives such as a cosmology-aware CNN or self-supervised encoder.

Neural methods must use the same split discipline, independent inference rule, fixed preprocessing, and multiple-seed evaluation. Reconstruction error is treated as one candidate score rather than assumed to be an OoD probability.

### Stage 8: Calibration and ensemble

Convert heterogeneous component scores onto training-calibrated scales. Combine only components whose errors are demonstrably complementary.

All ensemble components and weights are fixed before public-test inference. Test-set ranks, mean scores, quantiles, or mixture estimates are never used.

### Stage 9: Disciplined public submissions

Every Codabench submission corresponds to a registered experiment with:

- source revision;
- configuration;
- random seed;
- model and feature artifacts;
- local InD-tail results;
- pseudo-OoD results;
- generated submission checksum;
- overall public score and the 100 reported TPR values;
- an interpretation and next decision.

Submit only hypotheses that local evidence supports. Avoid probing the hidden labels through numerous arbitrary variations.

### Stage 10: Finalization

Freeze the selected method early enough to rerun it from raw data. Audit single-instance compliance, deterministic behavior, memory usage, runtime, dependency versions, and submission order.

Prepare the complete private source package, trained artifacts or deterministic training instructions, concise technical write-up, experiment summary, resource statement, and AI-use disclosure required for organizer verification.

## Data Flow

The intended pipeline is:

`raw arrays -> audited memory-mapped loader -> fixed mask/noise preprocessing -> feature extractors -> InD model and calibration -> independent per-map component scores -> fixed ensemble -> validated result.json -> clean ZIP submission`

Training and calibration artifacts flow forward. Information calculated from the public test population never flows backward into preprocessing, model fitting, calibration, or ensemble selection.

## Decision Gates

A stage advances only after its deliverable is verifiable:

1. Data audit passes shape, dtype, mask, label, and finite-value checks.
2. Submission tooling reproduces the official scorer behavior on synthetic labels and creates a valid dummy archive.
3. The baseline runs end to end and its deviations from the organizer reference are understood.
4. A new feature or model improves multiple validation settings without damaging the low-FPR InD tail.
5. An ensemble beats its strongest component across repeated local splits before receiving a public submission.
6. The final pipeline reproduces its submission from a clean environment.

## Initial Schedule

- July 27–August 3: setup, data audit, and dummy submission.
- August 4–16: power-spectrum baseline.
- August 17–September 7: validation suite and physical summaries.
- September 8–21: learned representations and calibrated combinations.
- September 22–October 1: method selection and freeze.
- October 2–10: full rerun, verification, final submissions, and write-up.
- October 11: UTC deadline safety buffer.

Schedule and candidate methods may change as evidence accumulates. Any change must preserve enough time for full reproduction and final verification.

