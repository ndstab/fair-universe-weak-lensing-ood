# FAIR Universe Weak-Lensing OoD Challenge — New Chat Handoff

> **PRIVATE COMPETITION CONTEXT:** This document contains our private
> competition methodology, code status, and results. Do not publish or share it
> with other participants while the competition is active.

Copy this entire document into the new chat. The new assistant should treat it
as authoritative context, inspect the repository before changing anything, and
continue from the pending tasks rather than restarting the project.

---

## Prompt for the new chat

You are continuing an active research and coding project for the **FAIR
Universe — Weak Lensing ML Uncertainty Challenge, Phase 2**, also described by
the user as the **NeurIPS 2026 Weak Lensing Uncertainty Challenge**.

Work collaboratively with the user until the competition goal is genuinely
handled. This is intended to be the user's central workspace for competition
physics explanations, research, experiments, code, submissions, and final
write-up.

### Immediate instruction

Do not restart the project or regenerate completed work. First inspect the
repository and the artifacts listed below. The immediate unfinished task is to
submit the already-generated power-spectrum baseline ZIP to Codabench, capture
its score and 100 logged TPR values, and record the result. As far as the prior
chat knows, the ZIP has **not yet been confirmed as uploaded**. Verify that with
the user or Codabench before making a duplicate submission.

After the first score is recorded, use it as the reference baseline and design
the next controlled research milestone. Do not make arbitrary leaderboard
submissions.

## User preferences and authority

- The user is highly motivated to achieve a meaningful result and ideally rank
  strongly enough for NeurIPS recognition.
- Explain unfamiliar physics and statistics simply and one concept at a time
  when the user asks. The user previously found dense explanations difficult
  to follow.
- The user explicitly approved working directly on the repository's `main`
  branch. Do not create a worktree unless the user changes that instruction.
- Preserve unrelated user changes and private artifacts.
- The user has GCloud credentials and access to useful GPUs. **No GPU has been
  used so far. Before any GPU training or GCloud configuration, explicitly tell
  the user and coordinate access with them. Never ask them to paste secrets into
  chat or commit credentials.**
- Current work should remain private during the competition.
- AI use must be disclosed. Continue updating `docs/AI_USAGE.md` for material
  AI-assisted decisions.

## Competition links and source material

- Codabench competition:
  <https://www.codabench.org/competitions/10902/#/pages-tab>
- Organizer GitHub repository:
  <https://github.com/FAIR-Universe/Cosmology_Challenge>
- Official Phase-2 power-spectrum notebook:
  <https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Phase_2_Startingkit_WL_PSAnalysis.ipynb>
- Challenge white paper:
  <https://arxiv.org/abs/2604.14451>
- Competition deadline: **11 October 2026 at 23:59 UTC**, equivalent to
  **12 October 2026 at 05:29 IST**.
- Public test data and training data are organizer-provided simulations.

Original user-provided competition text remains available at:

- `/Users/ndstab/.codex/attachments/f0e15975-adcc-4500-8e92-aaca94d90231/pasted-text.txt`
- `/Users/ndstab/.codex/attachments/9020b2e3-4dcd-4de9-80ae-2b96a735ef98/pasted-text.txt`
- `/Users/ndstab/.codex/attachments/5fbe36a4-76cb-4743-bb80-7c784c8f6837/pasted-text.txt`

These contain the competition overview, starting-kit explanation, and terms
and conditions respectively.

## Problem statement in plain language

Weak gravitational lensing measures tiny, coherent distortions in the shapes
of distant galaxies caused by matter—including invisible dark matter—between
those galaxies and us. Those distortions can be reconstructed into a
two-dimensional **convergence map**. A convergence value, usually denoted
`kappa`, summarizes the projected matter focusing along that sky direction.

Important clarification from the previous teaching discussion: a map pixel
does **not** contain one numerical matter value plus a separate brightness
value. It contains one numerical convergence value. Brightness/color is only
how visualization software displays that number.

The training and test data are all simulations. Training maps are noiseless;
the public-test maps already include observational noise to resemble real
survey measurements. Some test maps follow the same simulation family as the
training maps (InD), while others were generated under undisclosed alternative
physical assumptions (OoD).

Here, “normal” means the ordinary English word—compatible with the training
simulation family—not “Gaussian normal distribution.” A normal map need not
look identical to a training map. It only needs to be a plausible random
realization of the same allowed physical rules and parameter distributions.

The scientific purpose is to detect **simulation-to-reality mismatch**: if an
observed weak-lensing map cannot be adequately explained by the trusted
simulation family, cosmological parameter inference from that simulation may
be biased or unsafe.

## Exact competition task and restrictions

For each test map `x_i`, output one continuous score `t(x_i)`. Larger scores
must mean greater confidence that the map is OoD.

The official metric is:

- construct the ROC curve from the hidden InD/OoD labels;
- evaluate TPR at 100 logarithmically spaced FPR values from `0.001` to `0.05`;
- return the mean of those 100 TPR values.

This strongly rewards anomaly detection in the very-low-false-positive tail.

Critical rule: each test instance must be scoreable independently. A method
may use artifacts learned from training data and the current map, but it must
not use other public-test maps. Therefore:

- no test-set normalization;
- no test-set means, standard deviations, quantiles, histograms, or ranks;
- no clustering or mixture estimation across test samples;
- no test-time fitting across the 10,000 maps;
- no ensemble calibration using test-score distributions.

Operational batching and caching of independently computed test features are
allowed; cross-sample statistical dependence is not.

Submission format: a ZIP containing only root-level `result.json` with:

```json
{"ood_scores": [10000 finite floating-point scores in original test order]}
```

The Codabench scoring image `codalab/codalab-legacy:py39` is the server image
used to score the uploaded JSON. Our model runs locally; we submit only scores.

Prize eligibility requires reproducible source code and a concise technical
write-up. Organizers can verify single-instance compliance and reproduce the
method. Competition code and methodology must not be shared publicly while the
competition is active. AI-agent usage is allowed but must be disclosed.

## Dataset facts already verified

Repository data directory:

`/Users/ndstab/Desktop/unc/public_data/`

It is ignored by Git.

Files:

1. `WIDE12H_bin2_2arcmin_kappa_newrealization.npy`
   - training data;
   - shape `(101, 256, 132019)`;
   - dtype `float16`;
   - 25,856 noiseless simulations;
   - SHA-256 `f3f9f75e5aaf72e70ff9a486e165d57969e1fe2095bfe90dc4983879110bdcbd`;
   - size `6,826,966,656` bytes.
2. `WIDE12H_bin2_2arcmin_kappa_test_phase2_new.npy`
   - public test data;
   - shape `(10000, 132019)`;
   - dtype `float64`;
   - already noisy;
   - hidden mix of InD and OoD;
   - SHA-256 `d8d91680709ce8a9abd907e4b2cfdb752f55cf5acd13f819386458d4f34bb82f`;
   - size `10,561,520,128` bytes.
3. `WIDE12H_bin2_2arcmin_mask.npy`
   - shape `(1424, 176)`;
   - dtype `bool`;
   - exactly 132,019 valid pixels and 118,605 excluded pixels;
   - SHA-256 `ffe44ff93ad3122488fb18882dd457f337dc90c4e31306f50c4bb77aa8822da6`;
   - size `250,752` bytes.
4. `label_newrealization.npy`
   - shape `(101, 256, 5)`;
   - dtype `float64`;
   - SHA-256 `fd144017be3d50adc29a8665ef59341802f7acad58387f8ae68e30ad29fae8ff`;
   - size `1,034,368` bytes.

The maps are stored as flattened values only at `True` mask positions, not as
full `1424 x 176` arrays. Reconstruction fills NumPy boolean-mask positions in
their original order. The audit found zero non-finite values in training and
test maps and exactly 101 unique `(Omega_m, S8)` cosmologies.

Label columns:

1. `Omega_m`: matter density fraction;
2. `S8`: amplitude of present-day matter fluctuations;
3. `T_AGN`: AGN-feedback baryonic nuisance parameter;
4. `f0`: baryonic/star-formation nuisance parameter;
5. `delta_z`: photometric-redshift bias nuisance parameter.

The first two are constant across the 256 realizations within a cosmology. The
nuisance parameters are meant to be marginalized rather than treated as OoD.

The new Phase-2 training realization file has independent simulation seeds,
unlike the optional Phase-1 data with 256 shared seeds. Phase-1 data has not
been used in the current pipeline.

## Local repository and environment

Workspace:

`/Users/ndstab/Desktop/unc`

Git state at handoff creation:

- normal Git repository, not a worktree;
- branch `main`;
- last completed scientific commit before this handoff:
  `4938e4b8282a7a7db2a911622616a65eccad6f4b`;
- working tree was clean before creating this document.

Python environment:

- virtual environment: `/Users/ndstab/Desktop/unc/.venv`;
- Python `3.12.8`;
- NumPy `2.3.3`;
- SciPy `1.18.0`;
- scikit-learn `1.8.0`;
- Matplotlib `3.11.1`;
- pytest `8.4.2`;
- package installed editable as `fair-universe-wl-ood`;
- CLI executable: `.venv/bin/wl-ood`.

Install or refresh with:

```bash
cd /Users/ndstab/Desktop/unc
.venv/bin/python -m pip install -e ".[dev]"
```

Run tests with:

```bash
.venv/bin/python -m pytest -v
```

The latest verified suite contained **69 passing tests** after the optimizer
regression fix.

## Completed project foundations

The repository already provides:

- exact implementation of the official low-FPR score;
- validation and deterministic packaging of `result.json` submissions;
- deterministic dummy submission command;
- read-only memory-mapped competition-data loader;
- exact local dataset schema;
- chunked finiteness audit;
- streaming SHA-256 fingerprints;
- flattened-map reconstruction and fixed previews;
- private research and AI-usage documentation;
- pinned dependencies and automated tests.

Important files:

- `README.md`: current commands and research rules.
- `docs/AI_USAGE.md`: AI disclosure policy.
- `docs/superpowers/specs/2026-07-27-weak-lensing-ood-research-design.md`:
  overall research roadmap.
- `docs/superpowers/specs/2026-07-30-data-audit-design.md`: data audit design.
- `docs/superpowers/specs/2026-07-30-power-spectrum-baseline-design.md`:
  approved baseline specification.
- `docs/superpowers/plans/2026-07-30-power-spectrum-baseline.md`:
  completed implementation plan.
- `artifacts/data-audit.json`: verified real-data audit.
- `artifacts/previews/`: five visually inspected fixed map previews.

Primary modules:

- `src/wl_ood/data.py`: paths, schema, read-only memory maps.
- `src/wl_ood/audit.py`: structural validation and hashes.
- `src/wl_ood/maps.py`: mask reconstruction and previews.
- `src/wl_ood/metric.py`: official competition metric.
- `src/wl_ood/submission.py`: score validation and deterministic ZIP.
- `src/wl_ood/ps_config.py`: baseline constants and seed.
- `src/wl_ood/noise.py`: deterministic organizer-compatible training noise.
- `src/wl_ood/power_spectrum.py`: one-map FFT power-spectrum extractor.
- `src/wl_ood/feature_cache.py`: resumable memory-mapped feature caches.
- `src/wl_ood/ps_model.py`: cosmology-conditioned Gaussian emulator,
  likelihood, MAP fitting, and chi-squared score.
- `src/wl_ood/ps_validation.py`: leakage-safe training split and tail checks.
- `src/wl_ood/ps_pipeline.py`: benchmark, extraction, validation, full fit,
  independent test scoring, manifest, and ZIP.
- `src/wl_ood/cli.py`: all public commands.

## Approaches researched and implemented

### 1. Official power-spectrum baseline — implemented

The organizer's classical baseline was selected before neural models because it
is interpretable, resource-efficient, and validates the complete scientific
pipeline.

For each map it:

1. reconstructs the full masked field with zeros outside the survey mask;
2. uses the organizer training-noise scale
   `0.4 / sqrt(2 * 30 * 2^2)`;
3. seeds training noise independently with
   `SeedSequence([20260730, 1, cosmology, realization])`;
4. performs `numpy.fft.rfft2` on shape `(1424, 176)` without transposing;
5. bins power into 10 logarithmic multipole bins from 100 to 10,000;
6. uses `log10(power)` as the 10-dimensional summary;
7. estimates a mean and covariance at each of 101 cosmologies;
8. uses organizer covariance denominator `N - n_features - 2`;
9. interpolates mean and covariance with SciPy `LinearNDInterpolator`;
10. finds the best-fit `(Omega_m, S8)` using Nelder–Mead from `(0.3, 0.8)`;
11. outputs raw best-fit chi-squared as the OoD score.

Raw chi-squared was chosen instead of negative empirical p-values because it is
a permitted monotonic score and avoids ties. Higher chi-squared means the map
is harder to explain using any allowed training cosmology.

### 2. Leakage-safe local validation — implemented

Realization indices are split identically at every cosmology:

- `index % 5 in {2,3,4}`: fit;
- `index % 5 == 1`: calibration;
- `index % 5 == 0`: validation.

This gives:

- 15,453 fit maps;
- 5,151 calibration maps;
- 5,252 validation maps.

Thresholds come only from calibration scores at FPR targets `0.05`, `0.01`,
`0.005`, and `0.001`, using NumPy quantiles with `method="higher"`.

No public-test value participates in this validation.

### 3. Resumable feature computation — implemented

Training and test log-power features are stored as `float64` memory-mapped
`.npy` arrays with boolean completion maps. A row is flushed before its
completion flag, making interrupted jobs safely resumable. Cache configuration
includes algorithm version `ps-baseline-v1`, seed, physics constants, shapes,
valid-pixel count, and all raw-file SHA-256 hashes. Incompatible reuse fails
explicitly.

### 4. Neural autoencoder/CNN — researched but not attempted

The organizer also provides an autoencoder reconstruction-error baseline and a
CNN/MCMC baseline. They have not been implemented or trained yet. They are
deferred until the classical baseline is submitted and evaluated. These will
likely be the first tasks that benefit from the user's GCloud GPUs.

### 5. Higher-order physical summaries — planned, not attempted

The overall roadmap proposes evaluating, one controlled family at a time:

- pixel-distribution moments;
- peak counts and peak heights;
- multiscale filtered features;
- morphology/topology of dense and empty regions;
- wavelet or scattering summaries;
- learned representations;
- training-calibrated ensembles of genuinely complementary components.

No pseudo-OoD stress suite or higher-order feature family has been implemented
yet. These are logical next research stages after the first leaderboard score.

## Important debugging history

The first real validation attempt failed at flattened calibration row 273,
corresponding to cosmology index 5 and realization index 91. SciPy's default
two-dimensional Nelder–Mead budget stopped after 400 function evaluations.

Root-cause investigation found:

- the optimum was near the interpolation hull boundary;
- the unfinished default solution was finite but not converged;
- the same objective converged successfully after 754 evaluations;
- the organizer notebook does not check optimizer success and would silently
  use the unfinished point;
- our implementation deliberately keeps the strict success check.

The tested fix in commit `4938e4b` sets both `maxfev` and `maxiter` to `4,000`.
After this change, all real validation and test fits completed with zero
optimizer failures. Do not remove this limit increase without a controlled
experiment and regression test.

## Real baseline run and results

Canonical run directory:

`artifacts/power-spectrum/seed-20260730/`

### CPU benchmark

File: `benchmark.json`

- fixed maps benchmarked: 64;
- repeat features bitwise identical: true;
- throughput: approximately `284.94 maps/second`;
- peak resident memory: `152,551,424` bytes;
- projected full feature extraction: approximately `125.84 seconds`.

This established that the baseline is comfortably CPU-feasible. No GPU was
used.

### Training-only validation

File: `validation-report.json`

- optimizer failures: `0`;
- finite held-out scores: `10,403`;
- covariance maximum condition number: `4070.125669390005`;
- minimum Cholesky diagonal: `0.002857217990527163`;
- scoring runtime after cached extraction: approximately `17.14 seconds`.

Calibration thresholds and observed held-out InD false-positive rates:

| Target FPR | Threshold | Observed validation FPR |
|---:|---:|---:|
| 0.001 | 55.35374997527806 | 0.000952018278750952 |
| 0.005 | 44.78539246801814 | 0.003808073115003808 |
| 0.01 | 36.57631536468169 | 0.010472201066260471 |
| 0.05 | 20.546121854133837 | 0.0594059405940594 |

The low-FPR tail is well behaved enough for a reference submission.

### Full fit and test inference

File: `run-manifest.json`

- all 25,856 training maps used in the final fit;
- exactly 10,000 test scores generated in original order;
- all scores finite: true;
- all scores non-negative: true;
- scores non-constant: true;
- optimizer failures: `0`;
- full-fit/test-scoring runtime after cached extraction: approximately
  `16.78 seconds`;
- source revision: `4938e4b8282a7a7db2a911622616a65eccad6f4b`;
- source was clean during the run.

Submission:

`/Users/ndstab/Desktop/unc/submissions/ps-baseline-seed-20260730.zip`

- size: `88,919` bytes;
- SHA-256:
  `edaf44e1b7cc033bbb7b62573e24d9741352823d199d516e2e51e59c26a91f76`;
- archive contains only root-level `result.json`;
- `result.json` contains exactly 10,000 scores.

The organizer-compatible published reference mentioned in the research design
is roughly `0.2143`. Do not assume our score until Codabench actually reports
it.

## Commands already available

Audit the data:

```bash
.venv/bin/wl-ood audit-data \
  --data-dir public_data \
  --output artifacts/data-audit.json
```

Create fixed previews:

```bash
.venv/bin/wl-ood preview-data \
  --data-dir public_data \
  --output-dir artifacts/previews
```

Benchmark the baseline:

```bash
.venv/bin/wl-ood benchmark-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

Re-run or resume training-only validation:

```bash
.venv/bin/wl-ood validate-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730
```

Re-run or resume full inference and recreate the ZIP:

```bash
.venv/bin/wl-ood run-ps-baseline \
  --data-dir public_data \
  --run-dir artifacts/power-spectrum/seed-20260730 \
  --seed 20260730 \
  --submission submissions/ps-baseline-seed-20260730.zip
```

The feature commands re-hash raw inputs and refuse incompatible run
directories. Completed rows are skipped.

## Immediate pending work

1. Confirm whether the user has already uploaded
   `submissions/ps-baseline-seed-20260730.zip` manually.
2. If not, use the authenticated Codabench competition page to upload exactly
   that archive. The previous chat opened the Codabench login page, but no
   authenticated upload was confirmed.
3. Wait for terminal scoring status. Do not treat “submitted” or “running” as a
   final result.
4. Capture:
   - submission timestamp;
   - public leaderboard score;
   - all 100 logged TPR values;
   - submission ID/status;
   - archive SHA-256;
   - source revision.
5. Write the browser result to:
   `artifacts/power-spectrum/seed-20260730/leaderboard-result.json`.
6. Create the private experiment record:
   `experiments/20260730-ps-baseline/record.md`, following
   `experiments/README.md`. Use the actual submission date if the organizer
   submission occurs later.
7. Commit the experiment record, rerun tests, and confirm the repository is
   clean. Do not commit raw data, cached features, the score vector, or the ZIP.

If Codabench requires login, ask the user to sign in themselves. Do not request
their password, OTP, API token, or GCloud credentials in chat.

## Recommended next research milestone

After the first baseline score is recorded, do not jump directly into an
expensive neural model or tune blindly against the public leaderboard.

Recommended next order:

1. build a fixed pseudo-OoD stress suite using training-derived validation maps
   only;
2. include controlled extra/insufficient noise, smoothing, scale-dependent
   power changes, spatial rearrangement, and held-out parameter regions;
3. evaluate the power-spectrum baseline against each shift while preserving
   the InD low-FPR tail;
4. add one interpretable higher-order feature family, likely multiscale peak
   counts or wavelet/scattering summaries;
5. keep a feature only if it improves several stress families across fixed
   seeds without damaging the InD tail;
6. only then plan GPU-based learned representations or autoencoders, after
   notifying the user and coordinating GCloud access.

Use one controlled change per experiment. Every Codabench submission must map
to a source revision, configuration, seed, validation result, artifact hashes,
and a written hypothesis. Avoid using frequent arbitrary submissions to infer
hidden labels.

## Non-negotiable checks for the new assistant

- Inspect `git status`, recent commits, and existing artifacts before acting.
- Do not alter or delete the current reference run.
- Do not recompute or print public-test score distributions.
- Do not use the public test set to calibrate or choose model settings.
- Do not reorder test samples.
- Do not share the private repository or methodology publicly.
- Do not begin GPU work without notifying the user first.
- Preserve deterministic seed `20260730` for the reference baseline.
- Record all future approaches as separate, reproducible experiments.
- Keep explanations simple when discussing cosmology with the user; introduce
  one new concept at a time.

## Quick start for the new chat

Run these read-only checks first:

```bash
cd /Users/ndstab/Desktop/unc
git status --short --branch
git log -8 --oneline
.venv/bin/python -m pytest -q
.venv/bin/python -c 'import json; from pathlib import Path; r=json.loads(Path("artifacts/power-spectrum/seed-20260730/run-manifest.json").read_text()); print(r["score_count"], r["optimizer_failures"], r["submission"])'
```

Then ask one concise question only if needed:

> Have you already uploaded `ps-baseline-seed-20260730.zip` to Codabench, or
> should I upload it now from the signed-in competition page?

If the user confirms it is not uploaded and the browser is authenticated,
continue the submission workflow immediately.
