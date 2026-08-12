# Stage 3: broadened-signal training and controlled evaluation

Stage 3 extends the original dual-station U-Net with physically broadened
synthetic injections. Real filterbank data provide the backgrounds; labels and
injected signal parameters remain synthetic and known.

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

Minimal entry points:

```bash
python train.py --help
python diagnostics.py --help
python evaluate_broadening.py --help
```

Detailed commands and the required result inventory are in the repository root.
