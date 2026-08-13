# Stage 3: broadened-signal training and controlled evaluation

Stage 3 adapts the inherited dual-station U-Net comparison framework to
physically broadened synthetic signals. The underlying dual-station
architecture is inherited; the work in this stage adds broadened-signal
injection, training, diagnostics, and controlled evaluation, and produces the
broadened-aware checkpoint later reused by Stage 4.

Real filterbank observations provide the backgrounds, while labels and injected
signal parameters remain synthetic and known.

## Stage-3 contribution

The Stage-3 work in this repository focuses on the broadened-signal regime:

- physically broadened synthetic injection into real filterbank backgrounds;
- width-, S/N-, and profile-stratified evaluation;
- background-quality and noise-mismatch diagnostics;
- threshold calibration for downstream threshold-dependent summaries; and
- a frozen broadened-aware checkpoint for subsequent Stage-4 experiments.

The controlled evaluation remains synthetic by construction, so injection truth
is available directly during training and evaluation.

## Files

| File | Role |
|---|---|
| `train.py` | Three-stage training, case-stratified evaluation, and threshold calibration |
| `lorentzian_signals.py` | Broadening model, population sampling, and profile injection |
| `evaluate_broadening.py` | Fixed-width, width-by-S/N, and shape-generalisation evaluation |
| `diagnostics.py` | Background comparison, case visualisation, and noise-mismatch tests |
| `check_background_quality.py` | Bounded-memory background screening |
| `threshold_utils.py` | Shared threshold metrics and best-F1 selection |
| `json_utils.py` | Standards-compliant serialization of NumPy and missing values |

## Run notes

- Use the fine-frequency `*.rawspec.0000.fil` product for broadening work.
- Screen and visually inspect both station backgrounds before training.
- Pass the selected station-A file through `--filterbank`.
- Pass a genuine second-station background with `--station_b_filterbank` when
  making cross-station claims. Reusing station A is only a fallback simulation.
- Use `recommended_margin_stage3.json` for downstream threshold-dependent
  summaries; AUC-ROC itself is threshold-free.

Minimal entry points from the repository root:

```bash
python stages/stage3/train.py --help
python stages/stage3/diagnostics.py --help
python stages/stage3/evaluate_broadening.py --help
```

Detailed reproduction commands and the required result inventory are documented in `REPRODUCIBILITY.md` and the repository root.
