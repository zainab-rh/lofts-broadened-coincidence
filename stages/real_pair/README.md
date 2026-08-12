# Operational BLISS integration and real-pair pilot

This directory contains the adapter and orchestration layer between an upstream
BLISS broadened-candidate search and the frozen Stage-4 model. It supports two
scientifically separate workflows:

1. an unlabelled, simultaneous Ireland--Sweden LOFTS0050 pilot;
2. a blinded, labelled Synthetic Test B once the definitive upstream exports
   are available.

## Processing chain

```text
station catalogs
    -> validated canonical records
    -> candidate union from either station
    -> association and one-sided cases
    -> paired waterfall extraction
    -> frozen Stage-3/Stage-4 inference
    -> controls and descriptive analysis
```

The union is deliberately score-blind: candidate association must not use the
downstream model score it is later evaluated against.

## Key modules

| Area | Files |
|---|---|
| Schema and adapters | `lofts_bliss_schema.py`, `bliss_candidate_adapter.py`, `adapt_naoise_blind_catalog.py`, `BLISS_EXPORT_CONTRACT.md` |
| Observation/provenance | `make_observation_manifest.py`, `verify_naoise_checkout.py`, `verify_bliss_backend.py`, `lofts_filterbank.py` |
| Union and geometry | `candidate_union.py`, `real_candidate_union.py`, `real_pair_geometry.py`, `derive_real_pair_policy.py` |
| Extraction/inference | `extract_candidate_pairs.py`, `extract_shifted_controls.py`, `run_stage4_on_bliss.py` |
| Analysis/figures | `analyze_real_pair.py`, `plot_real_pair_roadmap.py`, `plot_representative_pair.py`, `plot_stage4_frozen_results.py` |
| Test-B locking | `freeze_association_policy.py`, `freeze_test_b_preregistration.py`, `link_test_b_recoveries.py`, `evaluate_bliss_stage4.py`, `freeze_test_b_artifacts.py` |

## LOFTS0050 pilot

```bash
bash stages/real_pair/run_lofts0050_real_pair.sh preflight
bash stages/real_pair/run_lofts0050_real_pair.sh help
```

Export the site-specific paths listed in the root `.env.example`, then verify
the upstream commit, script SHA-256, station identifiers, and barycentric
declarations before processing. The default inference device is CPU. The
output is a ranked/descriptive candidate analysis, not a labelled performance
measurement.

## Synthetic Test B

```bash
bash stages/real_pair/run_bliss_stage4_pipeline.sh preflight
bash stages/real_pair/run_bliss_stage4_pipeline.sh help
```

The upstream BLISS search itself remains outside the wrapper because its final
script and export columns are project-specific. Complete and freeze
`BLISS_EXPORT_CONTRACT.md` and the mapping JSON files before adapting any
locked candidate table. Follow `docs/TEST_B_HANDOFF.md`; do not reveal truth or
retune the policy before inference is complete.
