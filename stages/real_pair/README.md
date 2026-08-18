# Real-pair and Test-B workflow

This directory connects upstream BLISS candidate searches to the frozen LOFTS coincidence analysis. It supports three distinct validation workflows:

1. the completed unlabelled LOFTS0050 Ireland--Sweden real-data pilot;
2. the completed paired blind-injection BLISS recovery validation; and
3. the multi-event Test-B evaluation framework for quantitative end-to-end validation.

These workflows share infrastructure but answer different scientific questions and should not be combined into a single performance estimate.

## Processing model

The operational real-pair path is:

```text
independent station catalogues
        |
        v
canonical schema validation
        |
        v
Ireland union Sweden candidates
        |
        v
cross-station geometry and paired extraction
        |
        v
frozen Stage-3 / Stage-4 inference
        |
        v
candidate rankings, controls, and descriptive analysis
```

The union is deliberately inclusive. A candidate reported at only one station is retained and projected to the corresponding physical location in the other observation. This allows the downstream comparison to retain shared events whose signal-to-noise ratio differs between stations.

When both stations report a candidate, their recovered parameters remain station-specific rather than being silently averaged. Candidate association is performed in physical frequency and signed-drift coordinates.

## Main entry points

| File | Role |
| --- | --- |
| `run_lofts0050_real_pair.sh` | Orchestrates the LOFTS0050 pilot, pinned searches, policy freeze, union, extraction, inference, controls, analysis, and result freezing. |
| `run_pinned_naoise_search.py` | Runs the pinned broadened-signal BLISS search independently per station and records runtime provenance. |
| `bliss_candidate_adapter.py` / `adapt_naoise_blind_catalog.py` | Validate and adapt upstream candidate exports to the canonical schema. |
| `lofts_bliss_schema.py` | Defines canonical candidate fields and validation rules. |
| `derive_real_pair_policy.py` / `freeze_association_policy.py` | Derive and freeze association and control policies before evaluation. |
| `candidate_union.py` / `real_candidate_union.py` | Construct the sparse physical-coordinate two-station candidate union, including one-sided detections. |
| `real_pair_geometry.py` | Handle cross-station time/frequency geometry for paired extraction. |
| `analyze_real_pair.py` | Produce descriptive rankings and candidate-versus-control diagnostics for the unlabelled real pilot. |
| `link_test_b_recoveries.py` | Associate BLISS search candidates with injection truth after the blind search stage. |
| `evaluate_bliss_stage4.py` | Evaluate the multi-event Test-B result after truth linkage. |
| `freeze_test_b_preregistration.py` / `freeze_test_b_artifacts.py` | Freeze the Test-B design and final evaluation artifacts. |
| `BLISS_EXPORT_CONTRACT.md` | Defines the external BLISS candidate and truth-export contract. |

Use `--help` for the Python entry points and:

```bash
bash stages/real_pair/run_lofts0050_real_pair.sh help
```

for the current shell-runner command list.

## LOFTS0050 real-pair pilot

The real pilot uses simultaneous barycentric LOFTS0050 observations from Ireland and Sweden. The runner expects local paths to the two HDF5 observations, the pinned broadened-signal search checkout, the BLISS checkout, and the BLISS Python environment.

A minimal preflight pattern is:

```bash
export IRL_H5=/absolute/path/LOFTS0050.bary.0000_part000.h5
export SWE_H5=/absolute/path/LOFTS0050.bary.0000_part000.h5
export NAOISE_REPO=/absolute/path/bliss-broadened-signals
export BLISS_REPO=/absolute/path/bliss
export BLISS_PYTHON=/absolute/path/bliss-environment/bin/python

bash stages/real_pair/run_lofts0050_real_pair.sh preflight
bash stages/real_pair/run_lofts0050_real_pair.sh help
```

The label-free processing order is:

```text
adapt -> derive-policy -> freeze-policy -> union -> extract -> infer
      -> controls -> infer-controls -> analyze -> freeze-results
```

If the upstream searches also need to be reproduced, the runner provides station-specific pinned-search commands as well as `search-both`. The pinned search source is verified before execution, and each station receives its own runtime provenance record.

The completed LOFTS0050 pilot is unlabelled. Its outputs therefore support candidate ranking, score-distribution inspection, and comparison with candidate-excluded shifted controls rather than labelled classification metrics.

The pilot establishes that the frozen union, paired-extraction, candidate-conditioned scoring, and control workflow can be applied to real simultaneous Ireland--Sweden data. Quantitative discrimination performance is measured separately on the labelled Test-A population.

## Paired blind-injection BLISS recovery validation

A known broadened event was injected into separate copies of the real Ireland and Sweden LOFTS0050 backgrounds and then searched independently by BLISS. Injection truth was kept separate from the search and used only after both searches completed for recovery association.

The final injected event used:

- Lorentzian profile;
- 140.3 MHz reference frequency;
- 30 Hz FWHM;
- +0.06 Hz/s signed drift;
- requested injection strength 30;
- seed 2026081201.

The working frequency was moved from 140.0 to 140.3 MHz before the final searches after the pinned geometry showed that 140.0 MHz lay in the Sweden coarse-channel rolloff. No search or recovery-association parameter was changed after the final recovery result was inspected.

Each injected observation was then searched independently with the pinned broadened-signal BLISS implementation.

Post-search recovery association used:

- the raw uncollapsed BLISS candidate catalogues;
- `FREQ_MHZ` evaluated at the first-row truth epoch;
- `DR_HZ_S`;
- a frequency tolerance of +/-0.3 kHz; and
- a signed-drift tolerance of +/-0.007 Hz/s.

These tolerances were applied only after the independent searches. They were not supplied to BLISS.

Exactly one candidate fell inside the recovery gate at each station:

| Station | Recovered frequency (MHz) | Recovered drift (Hz/s) | BLISS bank S/N |
| --- | ---: | ---: | ---: |
| Ireland | 140.299999 | +0.05921 | 266.913 |
| Sweden | 140.300000 | +0.05907 | 178.229 |

Both recovered drift estimates differ from the injected +0.0600 Hz/s track by approximately \(10^{-3}\) Hz/s, consistent with the nearest searched drift-grid value.

This result validates the paired upstream chain on real station backgrounds:

```text
paired injection -> independent BLISS search -> recovery association
```

It demonstrates that the injected 30 Hz broadened event is independently recovered by BLISS at both stations. It is not, by itself, a Stage-4 performance measurement or an estimate of BLISS completeness.

Some existing repository directories retain the earlier internal `test_b_smoke` name for compatibility:

```text
results/test_b_smoke/
figures/test_b_smoke/
```

The scientific interpretation of those products is the paired blind-injection BLISS recovery validation described above.

Raw injected HDF5 files, complete blind-search candidate catalogues, and full search logs remain outside ordinary Git.

## Multi-event Test B

The next quantitative phase is a blind multi-event evaluation of the complete chain:

```text
paired injections -> independent BLISS searches -> candidate union
                  -> paired extraction -> frozen Stage-4 inference
                  -> post-inference truth linkage -> evaluation
```

The purpose of Test B is to measure three quantities separately:

1. **BLISS completeness** for broadened signals in the paired real-background setting;
2. **Stage-4 coincidence performance conditional on candidate availability**; and
3. **overall end-to-end performance** of the search-to-coincidence pipeline.

Calibration and evaluation populations remain separate. Recovery-association tolerances are derived and frozen on calibration data before the blind evaluation population is scored. The evaluation design, random seeds, checkpoint hashes, association policy, and other frozen artifacts are recorded before truth is used for final evaluation.

The repository contains the relevant design and evaluation machinery, including:

```text
config/synthetic_test_b_design.example.json
config/synthetic_test_b_preregistration.json
stages/real_pair/build_synthetic_test_b_labels.py
stages/real_pair/freeze_association_policy.py
stages/real_pair/freeze_test_b_preregistration.py
stages/real_pair/link_test_b_recoveries.py
stages/real_pair/evaluate_bliss_stage4.py
stages/real_pair/freeze_test_b_artifacts.py
```

Some filenames retain the earlier `synthetic_test_b` identifier for compatibility with the implemented code. In the research documentation, this evaluation is referred to as the **multi-event Test B**.

The complete multi-event Test B has not yet been run for the current repository release. The completed one-event recovery result above validates the paired injection, independent BLISS search, and post-search recovery-association interface on which the larger evaluation is built.

## Export and provenance rules

Candidate exports must satisfy `BLISS_EXPORT_CONTRACT.md`. In particular, frequency epoch, signed-drift convention, width convention, station identity, and searched/unsearched regions must be explicit before candidate association.

Injection truth must remain separate from blind search candidates until the registered inference outputs have been written. Machine-specific absolute paths belong in local runtime provenance rather than curated public result tables.

Checkpoint, input, search-script, policy, and frozen-result identities should be recorded with SHA-256 whenever they affect a reported result.
