# Reproducibility guide

This document describes the minimum record needed to reproduce a scientific
result. Machine-specific absolute paths may differ, but inputs, code revision,
configuration, seeds, checkpoints, and output hashes must be preserved.

## 1. Capture the environment

From the repository root:

```bash
bash scripts/capture_environment.sh results/environment
bash scripts/check_repository.sh
```

Commit the generated text files with the result they describe. Record the job
scheduler request separately when GPU model, memory, or wall time affects the
run.

## 2. Record immutable inputs

For every experiment, retain:

* repository commit and dirty/clean status;
* exact command line and relevant environment variables;
* SHA-256 for each filterbank, candidate table, policy, preregistration, and
  checkpoint;
* filterbank header fields (`fch1`, signed `foff`, `tsamp`, dimensions, bit
  depth, start time, and barycentric status);
* random seed and bootstrap seed;
* software environment and compute device;
* explicit statement of whether station B is a real second station or a
  background proxy.

Do not identify files only by basename when two storage trees may contain the
same observation name.

## 3. Stage 3

Stage 3 requires the fine-frequency `0000` product for the broadened-signal
analysis. Screen candidate backgrounds first:

```bash
cd stages/stage3
python check_background_quality.py \
  --glob '/datax2/projects/LOFTS/**/*.rawspec.0000.fil' \
  --out_json ../../results/stage3/background_quality_scan.json
```

Pass the selected station-A background explicitly:

```bash
python train.py \
  --mode high_freq \
  --filterbank /absolute/path/station_a.rawspec.0000.fil \
  --include_broadened \
  --station_b_filterbank /absolute/path/station_b.rawspec.0000.fil \
  --run_id stage3 \
  --num_workers 0
```

Copy the frozen checkpoint to
`checkpoints/stage3/model_high_freq_broadened.pth` outside ordinary Git and
record its SHA-256. Evaluate the Stage-2 and Stage-3 checkpoints on the same
controlled grid:

```bash
python evaluate_broadening.py \
  --mode high_freq \
  --checkpoint_a training_runs/stage3_high_freq/model_high_freq.pth \
  --checkpoint_a_label stage2 \
  --checkpoint_b training_runs/stage3_high_freq/model_high_freq_broadened.pth \
  --checkpoint_b_label stage3 \
  --station_b_filterbank /absolute/path/station_b.rawspec.0000.fil \
  --margin VALUE_FROM_recommended_margin_stage3.json \
  --run_snr_grid \
  --out_dir ../../results/stage3/evaluation
```

## 4. Stage 4 / Synthetic Test A

The runner resolves the repository layout automatically. Export the real data
and checkpoint paths, then use a compute node:

```bash
export STATION_A=/absolute/path/station_a.rawspec.0000.fil
export STATION_B=/absolute/path/station_b.rawspec.0000.fil
export STAGE3_CHECKPOINT=$PWD/checkpoints/stage3/model_high_freq_broadened.pth

bash stages/stage4/run_stage4_experiment.sh preflight
bash stages/stage4/run_stage4_experiment.sh all
```

The full evaluation uses disjoint frequency partitions, fixed width/shape/S/N
cells, and paired bootstrap comparisons across methods. Preserve the generated
manifest before any reanalysis. Copy the Stage-4 checkpoint to
`checkpoints/stage4/model_stage4_candidate_conditioned.pt` outside ordinary
Git and record its hash.

## 5. Real LOFTS0050 pilot

The real-pair runner is intentionally separate from labelled evaluation. Its
default device is CPU and its outputs are descriptive only.

```bash
export IRL_H5=/absolute/path/LOFTS0050.bary.0000_part000.h5
export SWE_H5=/absolute/path/LOFTS0050.bary.0000_part000.h5
export NAOISE_REPO=/absolute/path/bliss-broadened-signals
export BLISS_REPO=/absolute/path/bliss
export BLISS_PYTHON=/absolute/path/bliss-environment/bin/python

bash stages/real_pair/run_lofts0050_real_pair.sh preflight
bash stages/real_pair/run_lofts0050_real_pair.sh help
```

Before processing, verify the pinned upstream search checkout and script hash,
the BLISS backend revision, HDF5 headers, barycentric provenance, and the
frozen association policy. The real-pair pilot is unlabelled, so reproduce its
candidate rankings and shifted-control score comparisons rather than
classification metrics.

## 6. Test-B engineering smoke and locked evaluation

The Test-B framework contains two distinct activities: a completed one-event engineering smoke and the larger locked statistical evaluation.

### 6.1 Completed paired engineering smoke

A one-event engineering smoke was completed using the simultaneous LOFTS0050 Ireland and Sweden backgrounds.

The injected signal was a 30 Hz Lorentzian at 140.3 MHz with +0.06 Hz/s drift, requested strength 30, and seed 2026081201. The initial working frequency of 140.0 MHz was moved to 140.3 MHz before the final searches after the pinned geometry showed that 140.0 MHz lay in the Sweden coarse-channel rolloff. No search or association setting was changed after the recovery result was inspected.

Separate injected copies of the two observations were searched independently with the pinned broadened-signal BLISS implementation. The search script SHA-256 was:

```text
50d3c512f68946fd1786ecc122441341382d46ec163ef4c009c40d8692b26c6a
```

The final searches used 262144 fine channels per coarse channel and a +/-0.2 Hz/s drift range.

Recovery association used the raw uncollapsed candidate catalogues. Candidate frequency was `FREQ_MHZ`, candidate drift was `DR_HZ_S`, and truth frequency was evaluated at the first observation row. The pre-existing association gates were +/-0.3 kHz in frequency and +/-0.007 Hz/s in drift.

The injected event produced one association at each station:

| Station | Recovered frequency (MHz) | Recovered drift (Hz/s) | BLISS bank S/N |
|---|---:|---:|---:|
| Ireland | 140.299999 | +0.05921 | 266.913 |
| Sweden | 140.300000 | +0.05907 | 178.229 |

Both station searches completed successfully, and the original HDF5 inputs remained unchanged. Curated smoke outputs belong under `results/test_b_smoke/` and `figures/test_b_smoke/`.

This smoke validates the paired injection and independent BLISS-recovery path. It is not the statistical Synthetic Test-B endpoint.

### 6.2 Locked Synthetic Test B

The full locked Test B remains the end-to-end statistical evaluation. Follow `docs/TEST_B_HANDOFF.md`.

The required order is:

1. estimate association tolerances on a separate calibration population;
2. freeze the association policy;
3. freeze the Test-B design, seed, checkpoint hashes, and preregistration;
4. run BLISS independently at both stations and adapt the blinded candidate exports;
5. construct the two-station union, including one-sided detections;
6. extract paired station views and run frozen Stage-4 inference without truth access;
7. reveal or link truth only after predictions have been written;
8. evaluate the registered endpoints and freeze the complete result bundle.

Any model, threshold, union policy, population, or association change after truth reveal creates a new analysis and must not replace the preregistered result.

The complete multi-event locked Test B has not been run for the current repository release.

## 7. Archiving

For every reported run, create a manifest of relative paths and SHA-256 values.
The repository-level `MANIFEST.sha256` verifies the distributed source bundle;
run-specific manifests verify large external inputs and generated artifacts.
