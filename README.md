# LOFTS broadened-signal coincidence

Research code for testing whether spectrally broadened narrowband candidates are consistent between simultaneous LOFAR stations. The repository follows the research from simulated broadened-signal modelling through candidate-informed learning, a real Ireland--Sweden LOFTS0050 pilot, and blind BLISS recovery validation on paired real station backgrounds.

The central scientific question is whether a broadened candidate found at one or both stations is consistent with the same physical event in the simultaneous observation at the other station. The operational candidate set is therefore a union rather than an intersection: one-station BLISS detections are retained and projected to the corresponding physical location in the other observation.

## Technical report

A detailed description of the scientific motivation, Stage-3 and Stage-4 methods, controlled Test-A evaluation, real LOFTS0050 pilot, blind BLISS recovery validation, and planned multi-event Test B is provided in the final technical report:

[**Zainab Rehman - LOFTS Stage-4 Technical Report**](docs/Zainab_Rehman_LOFTS_Stage4_Technical_Report.pdf)

The README provides the repository-level overview and execution structure; the technical report provides the fuller scientific interpretation and context.

## Evidence overview

| Component | Question | Evidence status |
| --- | --- | --- |
| Stage 3 | Can the inherited dual-station comparison framework learn coincidence for simulated broadened signals injected into real backgrounds? | Controlled simulated evaluation |
| Stage 4 / Test A | Does candidate-informed preparation plus the learned Stage-4 comparison improve 10--100 Hz cross-station discrimination? | Frozen labelled simulated evaluation |
| Real-pair pilot | Can the frozen pipeline process an operational union of Ireland and Sweden candidates? | Completed unlabelled real-data pilot |
| Blind BLISS recovery | Can a known shared broadened event be independently recovered by BLISS in both real station backgrounds? | Completed; recovered independently at both stations |
| Multi-event Test B | What are BLISS completeness, Stage-4 conditional coincidence performance, and overall end-to-end performance on a blind multi-event evaluation? | Evaluation design defined; full campaign not yet run |

## Current results

On the frozen detected-conditioned Test-A population, Raw Stage 3 reached an AUC of 0.818, the transparent candidate-informed baseline reached 0.9747, and Stage 4 reached 0.9932. The transparent baseline captures most of the improvement relative to Raw Stage 3, while Stage 4 provides a smaller additional learned improvement on the same frozen pairs.

The LOFTS0050 real-pair workflow has also been run on simultaneous Ireland--Sweden data. Because this pilot is unlabelled, it is used for descriptive ranking and control comparisons rather than classification metrics.

For the blind BLISS recovery validation, a 30 Hz Lorentzian signal at 140.3 MHz with +0.06 Hz/s drift was injected into separate copies of the real Ireland and Sweden backgrounds. Independent pinned BLISS searches recovered one associated candidate at each station at the expected frequency and nearest searched drift. This validates the paired injection, independent search, and recovery-association chain on real station backgrounds; the larger multi-event Test B will provide the quantitative end-to-end evaluation.

## Repository map

```text
lofts-broadened-coincidence/
├── README.md
├── REPRODUCIBILITY.md
├── docs/
│   └── Zainab_Rehman_LOFTS_Stage4_Technical_Report.pdf
├── stages/
│   ├── stage3/        # physical injections, training, diagnostics, width sweeps
│   ├── stage4/        # candidate-conditioned model and frozen Test-A evaluator
│   └── real_pair/     # BLISS integration, union, extraction, controls and Test-B framework
├── tests/             # cross-stage integration and unit-test entry points
├── requirements/      # environment requirements by stage
├── checkpoints/       # local checkpoint locations; model binaries are not committed
├── results/           # machine-readable, curated scientific outputs
├── figures/           # publication/presentation figures
└── scripts/           # repository validation and environment capture
```

Each stage has a short README containing its own inputs and commands. The root README remains the authoritative repository overview; the stage files provide stage-specific execution details.

## Installation

Python 3.10 or newer is recommended. On the Sweden compute environment, retain the tested PyTorch/CUDA build and install only missing packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/check_repository.sh
```

The code does not distribute LOFAR filterbanks, BLISS source, trained weights, or private candidate tables. Place local model files at the paths documented in `checkpoints/`, and pass data paths through command-line arguments or runner environment variables.

## Typical workflow

1. Screen candidate background observations with `stages/stage3/check_background_quality.py` and retain the scan JSON.
2. Train and evaluate Stage 3 on fine-frequency backgrounds, preserving the selected checkpoint and threshold-calibration artifacts.
3. Run the Stage-4 experiment using the frozen Stage-3 checkpoint. Preserve the Test-A manifest, pair-level results, checkpoint hashes, and bootstrap outputs before any post-hoc comparison.
4. Run the LOFTS0050 real-pair workflow using independently produced Ireland and Sweden BLISS candidate catalogues. Construct the candidate union, extract the corresponding views from both stations, and apply the frozen Stage-4 comparison.
5. Use shifted, candidate-excluded locations as real-data controls for descriptive score-distribution comparisons.
6. For blind BLISS recovery validation, use isolated paired injections and independent pinned BLISS searches. Keep injection truth separate from the search and retain the recovery table and search provenance.
7. For the multi-event Test B, derive and freeze the recovery-association policy on separate calibration data, freeze the evaluation design and checkpoint hashes, perform truth-blind inference, and link truth only after predictions have been written.

For the full scientific description and interpretation, see the [technical report](docs/Zainab_Rehman_LOFTS_Stage4_Technical_Report.pdf). See `REPRODUCIBILITY.md` for reproduction commands and `results/README.md` for the curated machine-readable reference artifacts.

## Scientific conventions

- Spectral width is treated as FWHM in Hz unless an input contract explicitly states otherwise.
- Drift is in physical Hz/s. Signed channel spacing is retained during de-chirping, including descending-frequency filterbanks.
- Candidate union means candidates reported by either station are considered; missing counterparts are explicit one-sided cases, not silently discarded.
- AUC-ROC and paired bootstrap intervals are primary rank-based summaries. Threshold-dependent metrics must identify the calibration set and threshold.
- Simulated and real-data results are stored in separate directories and must not be pooled into one performance estimate.

## Results policy

Commit curated tables, JSON/JSONL manifests, configuration, checksums, and a small set of interpretable figures. Do not commit raw filterbanks, extracted pair arrays, caches, duplicate plots, or exploratory outputs with no role in a reported result. Large checkpoints should be distributed through an approved artifact store or Git LFS and identified by SHA-256 in the result manifest.
