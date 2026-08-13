# Real-pair and Synthetic Test-B workflow

This directory connects upstream BLISS candidate searches to the frozen LOFTS
coincidence analysis. It supports three distinct workflows:

1. the completed unlabelled LOFTS0050 Ireland--Sweden real-data pilot;
2. the completed one-event paired BLISS engineering smoke; and
3. the locked multi-event Synthetic Test-B evaluation framework.

These workflows share infrastructure, but they answer different questions and
should not be combined into a single performance estimate.

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

The union is deliberately inclusive. A candidate reported at only one station
is retained and projected to the corresponding physical location in the other
observation. When both stations report a candidate, their recovered parameters
remain station-specific rather than being silently averaged.

## Main entry points

| File | Role |
|---|---|
| `run_lofts0050_real_pair.sh` | Orchestrates the LOFTS0050 pilot, pinned searches, policy freeze, union, extraction, inference, controls, analysis, and result freezing. |
| `run_pinned_naoise_search.py` | Runs the pinned broadened-signal BLISS search independently per station and records runtime provenance. |
| `bliss_candidate_adapter.py` / `adapt_naoise_blind_catalog.py` | Validate and adapt upstream candidate exports to the canonical schema. |
| `lofts_bliss_schema.py` | Canonical candidate-field definitions and validation. |
| `derive_real_pair_policy.py` / `freeze_association_policy.py` | Derive and freeze association/control policy before scoring. |
| `candidate_union.py` / `real_candidate_union.py` | Construct the sparse two-station candidate union, including one-sided detections. |
| `real_pair_geometry.py` | Handle the cross-station time/frequency geometry used for paired extraction. |
| `analyze_real_pair.py` | Produce descriptive rankings and candidate-versus-control diagnostics for the unlabelled real pilot. |
| `link_test_b_recoveries.py` | Associate blind BLISS candidates with injection truth after the search stage. |
| `evaluate_bliss_stage4.py` | Evaluate the locked end-to-end Synthetic Test-B result after truth linkage. |
| `freeze_test_b_preregistration.py` / `freeze_test_b_artifacts.py` | Freeze the Test-B design and final artifact bundle. |
| `BLISS_EXPORT_CONTRACT.md` | Defines the external BLISS candidate and truth-export contract. |

Use `--help` for the Python entry points and:

```bash
bash stages/real_pair/run_lofts0050_real_pair.sh help
```

for the current shell-runner command list.

## LOFTS0050 real-pair pilot

The real pilot uses simultaneous barycentric LOFTS0050 observations from
Ireland and Sweden. The runner expects local paths to the two HDF5 observations,
the pinned broadened-signal search checkout, the BLISS checkout, and the BLISS
Python environment.

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

If the upstream searches also need to be reproduced, the runner provides
station-specific pinned-search commands as well as `search-both`. The pinned
search source is verified before execution and each station receives its own
runtime provenance record.

The completed LOFTS0050 pilot is unlabelled. Its outputs therefore support
candidate ranking, score-distribution inspection, and comparisons with
candidate-excluded shifted controls rather than labelled classification
metrics.

## Completed paired BLISS engineering smoke

A one-event smoke was run through separate injected copies of the real Ireland
and Sweden LOFTS0050 backgrounds to test the upstream injection/search path.

The final injected event used:

- Lorentzian profile;
- 140.3 MHz reference frequency;
- 30 Hz FWHM;
- +0.06 Hz/s signed drift;
- requested injection strength 30;
- seed 2026081201.

The working frequency was moved from 140.0 to 140.3 MHz before the final
searches after the pinned geometry showed that 140.0 MHz lay in the Sweden
coarse-channel rolloff. No search or association parameter was changed after
the recovery result was inspected.

Each injected observation was then searched independently with the pinned
broadened-signal BLISS implementation. Recovery association used the raw
uncollapsed candidate catalogues, `FREQ_MHZ` at the first-row truth epoch,
`DR_HZ_S`, and the pre-existing gates of +/-0.3 kHz in frequency and
+/-0.007 Hz/s in drift.

The event was recovered independently at both stations:

| Station | Recovered frequency (MHz) | Recovered drift (Hz/s) | BLISS bank S/N |
|---|---:|---:|---:|
| Ireland | 140.299999 | +0.05921 | 266.913 |
| Sweden | 140.300000 | +0.05907 | 178.229 |

The smoke is an engineering validation of paired injection and independent
BLISS recovery. It is not a Stage-4 performance measurement and is not the
multi-event Synthetic Test-B endpoint.

Curated public products are kept under:

```text
results/test_b_smoke/
figures/test_b_smoke/
```

Raw injected HDF5 files, full blind-search candidate catalogues, and complete
search logs remain outside ordinary Git.

## Locked Synthetic Test B

The full Synthetic Test B is the planned blind end-to-end statistical
evaluation of:

```text
paired injections -> independent BLISS searches -> candidate union
                  -> paired extraction -> frozen Stage-4 inference
                  -> post-inference truth linkage -> evaluation
```

Calibration and evaluation populations must remain separate. Association
tolerances are derived and frozen on calibration data before the blind
evaluation population is scored. The Test-B design, seed, checkpoint hashes,
association policy, and preregistration are frozen before truth is used for
evaluation.

The repository contains the relevant design and evaluation machinery,
including:

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

The complete locked multi-event Test B has not been run for the current
repository release.

## Export and provenance rules

Candidate exports must satisfy `BLISS_EXPORT_CONTRACT.md`. In particular,
frequency epoch, drift convention, width convention, station identity, and
searched/unsearched regions must be explicit before candidate association.

Injection truth must remain separate from blinded search candidates until the
registered inference outputs have been written. Machine-specific absolute paths
belong in local runtime provenance, not in curated public result tables.

Checkpoint, input, search-script, policy, and frozen-result identities should
be recorded with SHA-256 whenever they affect a reported result.
