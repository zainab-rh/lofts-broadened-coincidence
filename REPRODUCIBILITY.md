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
frozen association policy. Do not derive accuracy or AUC from this unlabelled
run. Use shifted controls only as empirical score-distribution diagnostics.

## 6. Synthetic Test B

Test B remains pending until the definitive upstream blind-injection search and
export contract are available. Follow `docs/TEST_B_HANDOFF.md`. The required
order is:

1. estimate association tolerances on the calibration set;
2. freeze the association policy;
3. freeze the Test-B design, seed, checkpoint hashes, and preregistration;
4. run upstream BLISS and adapt the blinded candidate exports;
5. construct the two-station union and run inference without truth access;
6. reveal/link truth only after predictions exist;
7. evaluate and freeze all artifacts.

Any model, threshold, union policy, or association change after label reveal
creates a new analysis and must not replace the preregistered result.

## 7. Archiving

For every reported run, create a manifest of relative paths and SHA-256 values.
The repository-level `MANIFEST.sha256` verifies the distributed source bundle;
run-specific manifests verify large external inputs and generated artifacts.
