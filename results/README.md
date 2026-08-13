# Reference results

This directory contains a compact set of machine-readable outputs from the
runs used to produce the figures and numerical results in this repository.

These files are included for verification and comparison. They are not a dump
of all intermediate compute products; full intermediate outputs are regenerated
by the analysis scripts and are intentionally not tracked in Git.

## Contents

### `stage3/`

Reference outputs for the selected Stage-3 synthetic baseline.

- `summary.json` — compact broadening, shape, S/N, and diagnostic results.
- `checkpoint.sha256` — SHA-256 identifier for the selected Stage-3 model.

The model weights themselves are not tracked in ordinary Git.

### `test_a/`

Frozen machine-readable outputs for Synthetic Test A, the labelled
detected-conditioned Stage-4 evaluation.

- `stage4_evaluation.json` — pooled and cellwise Stage-4 evaluation results.
- `stage4_width_results.csv` — results by width, signal shape, and population.
- `operating_point.json` — validation-selected operating point.
- `test_a_stage4_minus_filter.json` — paired Stage-4 versus transparent-filter comparison.
- `checkpoint_hashes.txt` — identifiers for the Stage-3 and Stage-4 checkpoints.

Test A evaluates downstream coincidence after a usable candidate is available.
It is not an end-to-end measurement of BLISS detection completeness.

### `real_lofts0050/`

Curated outputs from the unlabeled LOFTS0050 Ireland-Sweden real-data pilot.

- `observations_summary.json` — observation geometry and metadata summary.
- `candidate_union.summary.json` — candidate-union summary.
- `extraction_summary.json` — high-resolution pair-extraction summary.
- `stage4_bliss_inference_summary.json` — Stage-4 inference summary.
- `real_pair_analysis.json` — label-free real-pair analysis.
- `artifact_checksums.txt` — SHA-256 hashes of the curated public artifacts.
- `full_run_manifest.sha256` — identifier for the complete frozen run manifest.

Because the LOFTS0050 pilot is unlabeled, these outputs support descriptive
candidate ranking and shifted-control comparisons rather than classification
metrics.

### `test_b_smoke/`

Curated outputs from the completed one-event paired BLISS engineering smoke on
separate injected copies of the simultaneous LOFTS0050 Ireland and Sweden
backgrounds.

- `truth.csv` — injection truth for the two station instances of the smoke event.
- `recovery.csv` — post-search association summary for the independent station searches.
- `conventions.json` — field, frequency-epoch, and association conventions used by the smoke.
- `provenance.sha256` — checksums and source identifiers for the curated smoke bundle.

The 30 Hz signal at 140.3 MHz with +0.06 Hz/s drift was recovered by the
independent BLISS searches at both stations. This result validates the paired
injection and blind-search integration path; the full locked multi-event
Synthetic Test B remains the statistical end-to-end evaluation.

Associated presentation and diagnostic figures are stored under
`figures/test_b_smoke/`.

## Large artifacts

Raw filterbanks, extracted pair arrays, prediction catalogs, model weights,
training runs, and other large/intermediate compute products are intentionally
excluded from this directory.

Checkpoint paths referenced by these results correspond to the local/external
locations documented under `checkpoints/` and in `REPRODUCIBILITY.md`.
