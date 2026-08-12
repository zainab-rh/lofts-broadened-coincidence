# Stage 4: candidate-conditioned coincidence

Stage 4 evaluates a frozen candidate-informed comparison path in the 10--100 Hz
fine-frequency regime. For each station, the candidate view is de-chirped using
the reported drift, recentered, and integrated across the reported width before
being scored by a symmetric pair model.

The implementation retains signed channel spacing, tracks shifted-pixel
validity, and uses L2-normalised frequency filters. These details prevent a
descending-frequency axis or edge padding from producing an artificial signal.

## Files

| File | Role |
|---|---|
| `candidate_preprocessing.py` | Signed de-chirp, validity masks, recentering, and width integration |
| `channelized_injection.py` | Channel-integrated synthetic profiles and S/N calibration |
| `stage4_data.py` | Deterministic pair generation with disjoint frequency splits |
| `stage4_model.py` | Symmetric candidate-coincidence head on the Stage-3 encoder |
| `train_stage4.py` | Stage-4 fitting and operating-point calibration |
| `evaluate_stage4.py` | Same-pair comparison of raw Stage 3, corrected Stage 3, transparent filter, and Stage 4 |
| `run_stage4_experiment.sh` | Preflight, smoke, full training, and Test-A evaluation |
| `test_stage4_core.py` | Geometry, injection, split, checkpoint, and invariance tests |

## Inputs

- two screened fine-frequency backgrounds;
- the frozen Stage-3 broadened-aware checkpoint;
- explicit acknowledgement if station B is a proxy rather than a different
  telescope;
- sufficient compute for the full fixed-cell bootstrap evaluation.

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

The default full evaluation covers Lorentzian, box, and Gaussian profiles;
detected-conditioned and fixed-power S/N regimes; and a fixed mixture of
one-sided, independent, and noise negatives. Preserve the evaluation manifest
and pair-level export before any post-hoc comparison.

Test A is a synthetic held-out experiment. It supports claims about the stated
injection population, not end-to-end BLISS completeness on real observations.
