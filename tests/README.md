# Tests

`test_stage3_core.py` covers the shared Stage-3 threshold and preprocessing
helpers. Stage-specific regression suites remain beside the code they exercise:

- `stages/stage4/test_stage4_core.py`;
- `stages/real_pair/test_bliss_stage4_integration.py`;
- `stages/real_pair/test_real_pair_v3.py`;
- `stages/real_pair/test_cpu_and_plots_v3_2.py`.

Run all available repository checks with:

```bash
bash scripts/check_repository.sh
```
