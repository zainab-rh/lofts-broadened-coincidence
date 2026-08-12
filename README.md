# LOFTS broadened-signal coincidence

Research code for testing whether spectrally broadened narrowband candidates
are consistent between simultaneous LOFAR stations. The repository follows the
internship work from synthetic signal modelling through candidate-conditioned
learning and an operational Ireland--Sweden pilot.

The central scientific question is whether a candidate reported independently
at two stations remains recognisably coincident after accounting for its drift
rate and spectral width. The code therefore keeps three distinct evidence
levels separate:

| Component | Question | Evidence status |
|---|---|---|
| Stage 3 | Can a contrastive U-Net learn coincidence for synthetic broadened signals injected into real backgrounds? | Controlled synthetic evaluation |
| Stage 4 / Test A | Does candidate-informed de-chirping and frequency integration improve comparison at 10--100 Hz? | Held-out, detected-conditioned synthetic test |
| Real-pair pilot | Can the frozen pipeline process an operational union of Ireland and Sweden candidates? | Unlabelled engineering/scientific pilot |
| Synthetic Test B | Does the complete BLISS-to-Stage-4 pipeline recover blind injections? | Harness prepared; results pending the upstream BLISS script/export |

The unlabelled real-pair pilot is not a measurement of accuracy, completeness,
or astrophysical probability. Those claims require the labelled, blind
Synthetic Test B.

## Repository map

```text
lofts-broadened-coincidence/
├── README.md
├── REPRODUCIBILITY.md
├── stages/
│   ├── stage3/        # physical injections, training, diagnostics, width sweeps
│   ├── stage4/        # candidate-conditioned model and frozen Test-A evaluator
│   └── real_pair/     # BLISS adapters, union, extraction, controls, Test-B harness
├── tests/             # cross-stage smoke and unit-test entry points
├── requirements/      # environment requirements by stage
├── checkpoints/       # local checkpoint locations; model binaries are not committed
├── results/           # machine-readable, curated scientific outputs
├── figures/           # publication/presentation figures
├── docs/              # status, provenance, source selection, and result checklist
└── scripts/           # repository validation and environment capture
```

Each stage has a short README containing its own inputs and commands. The root
README remains the authoritative overview; the stage files do not duplicate the
project narrative.

## Installation

Python 3.10 or newer is recommended. On the Sweden compute environment, retain
the tested PyTorch/CUDA build and install only missing packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/check_repository.sh
```

The code does not distribute LOFAR filterbanks, BLISS source, trained weights,
or private candidate tables. Place local model files at the paths documented in
`checkpoints/`, and pass data paths through command-line arguments or runner
environment variables.

## Typical workflow

1. Screen candidate background observations with
   `stages/stage3/check_background_quality.py` and retain the scan JSON.
2. Train Stage 3 on fine-frequency (`*.0000`) backgrounds and calibrate its
   distance threshold on held-out pairs.
3. Run the fixed-width and width-by-S/N evaluation; archive the numerical
   tables together with the plots.
4. Train and evaluate Stage 4 through
   `stages/stage4/run_stage4_experiment.sh` using the frozen Stage-3
   checkpoint.
5. Run the unlabelled LOFTS0050 real-pair workflow only as a processing and
   distributional pilot.
6. When the upstream blind-injection exports become available, freeze the
   Test-B preregistration before inference, run the retained labelled harness,
   and add the locked outputs without retuning the model or association policy.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for reproduction commands and
[results/README.md](results/README.md) for the curated machine-readable
reference artifacts.

## Scientific conventions

- Spectral width is treated as FWHM in Hz unless an input contract explicitly
  states otherwise.
- Drift is in physical Hz/s. Signed channel spacing is retained during
  de-chirping, including descending-frequency filterbanks.
- Candidate union means candidates reported by either station are considered;
  missing counterparts are explicit one-sided cases, not silently discarded.
- AUC-ROC and paired bootstrap intervals are primary rank-based summaries.
  Threshold-dependent metrics must identify the calibration set and threshold.
- Synthetic and real-data results are stored in separate directories and must
  not be pooled into one performance estimate.

## Results policy

Commit curated tables, JSON/JSONL manifests, configuration, checksums, and a
small set of interpretable figures. Do not commit raw filterbanks, extracted
pair arrays, caches, duplicate plots, or exploratory outputs with no role in a
reported result. Large checkpoints should be distributed through an approved
artifact store or Git LFS and identified by SHA-256 in the result manifest.

