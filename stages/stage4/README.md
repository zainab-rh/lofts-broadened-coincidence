# Stage 4: candidate-informed coincidence

Stage 4 evaluates whether the same broadened candidate is mutually consistent
between two station views after candidate-informed signal preparation.

The method builds on the inherited Stage-3 dual-station encoder. This project
adds the candidate-informed preparation used for broadened signals and a
symmetric learned pair-comparison head. For each station, the candidate view is
de-chirped using the reported signed drift, recentered in frequency, and
integrated according to the reported spectral width before comparison.

The implementation retains signed channel spacing, tracks shifted-pixel
validity, and uses L2-normalised frequency filters. These details prevent a
descending-frequency axis or edge padding from creating an artificial signal.

## Architecture

The Stage-4 path is:

```text
BLISS candidate metadata
        |
        v
signed de-chirp -> recenter -> width-dependent frequency integration
        |
        v
prepared station A and station B views
        |
        v
shared inherited Stage-3 encoder
        |
        v
symmetric pair features + candidate-centred statistics
        |
        v
learned Stage-4 pair score
```

BLISS remains the upstream search: it identifies where to look and provides the
candidate metadata. Stage 4 performs the downstream cross-station consistency
comparison.

## Four reported comparison methods

Synthetic Test A evaluates four methods on the same frozen pairs. They are
comparison methods, not four sequential stages of one pipeline.

| Method | Definition |
|---|---|
| Raw Stage 3 | The inherited Stage-3 encoder applied directly to the raw station cutouts. |
| Corrected Stage 3 | The same inherited Stage-3 model applied after candidate-informed preparation. |
| Transparent filter | A model-free candidate-centred statistic computed from the prepared views, without a learned pair head. |
| Stage 4 | Candidate-informed preparation, the inherited encoder, and the symmetric learned pair-comparison head. |

This comparison separates the contribution of the candidate-informed
representation from the additional contribution of the learned Stage-4 head.

## Files

| File | Role |
|---|---|
| `candidate_preprocessing.py` | Signed de-chirp, validity masks, recentering, and width integration |
| `channelized_injection.py` | Channel-integrated synthetic profiles and S/N calibration |
| `stage4_data.py` | Deterministic pair generation with disjoint frequency splits |
| `stage4_model.py` | Symmetric candidate-coincidence head using the inherited Stage-3 encoder |
| `train_stage4.py` | Stage-4 fitting and operating-point calibration |
| `evaluate_stage4.py` | Same-pair evaluation of Raw Stage 3, Corrected Stage 3, transparent filter, and Stage 4 |
| `run_stage4_experiment.sh` | Preflight, smoke training, full training, and Test-A evaluation |
| `test_stage4_core.py` | Geometry, injection, split, checkpoint, and invariance tests |

## Inputs

The Stage-4 experiment requires:

- two screened fine-frequency backgrounds;
- the frozen broadened-aware Stage-3 checkpoint;
- explicit acknowledgement when station B is a proxy rather than a different
  telescope;
- sufficient compute for the frozen pair-level evaluation and bootstrap
  analysis.

The Stage-3 checkpoint is an inherited component of the Stage-4 model and
should be identified by its recorded SHA-256 when reproducing a reported run.

## Commands

From the repository root:

```bash
export STATION_A=/absolute/path/station_a.rawspec.0000.fil
export STATION_B=/absolute/path/station_b.rawspec.0000.fil
export STAGE3_CHECKPOINT=$PWD/checkpoints/stage3/model_high_freq_broadened.pth

bash stages/stage4/run_stage4_experiment.sh preflight
bash stages/stage4/run_stage4_experiment.sh smoke-train
bash stages/stage4/run_stage4_experiment.sh all
```

The full evaluation covers Lorentzian, box, and Gaussian profiles and includes
both detected-conditioned and fixed-power signal populations. Preserve the
evaluation manifest and pair-level export before any post-hoc comparison.

## Frozen Synthetic Test-A result

The primary frozen detected-conditioned Test-A population contains 10,800
labelled pairs in the 10--100 Hz broadening range.

| Method | AUC-ROC |
|---|---:|
| Raw Stage 3 | 0.818 |
| Corrected Stage 3 | 0.833 |
| Transparent filter | 0.9747 |
| Stage 4 | 0.9932 |

Relative to Raw Stage 3, Stage 4 improves AUC by approximately +0.1754, with a
paired 95% confidence interval of approximately [0.1679, 0.1833].

The transparent filter is already a strong candidate-informed baseline. On the
same frozen pairs, Stage 4 improves over that baseline by approximately +0.0185
AUC, with a paired 95% confidence interval of approximately
[0.0156, 0.0214]. This isolates an additional learned contribution beyond the
explicit candidate-centred signal-processing statistic.

At the frozen operating points, Raw Stage 3 has 44.7% recall at 3.20% false-
positive rate, while Stage 4 has 96.3% recall at 3.13% false-positive rate.

## Population interpretation

The detected-conditioned population asks how well the downstream comparison
works once a usable broadened candidate is already available. Across this
population, Stage 4 remains strong over the tested 10--100 Hz range.

The fixed-power population asks a different question: the same total signal
power is spread over increasing spectral width. It therefore becomes harder as
the signal broadens and is useful for studying the boundary between downstream
comparison and upstream candidate recovery.

Synthetic Test A is the labelled downstream evaluation. End-to-end BLISS
recovery is treated separately in the Test-B framework under
`stages/real_pair/`.
