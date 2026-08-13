# BLISS export contract

This document defines the candidate and truth-table formats expected by the
Stage-4 integration.

BLISS is run independently for each station. Candidate tables are used to
construct the station union and run inference. Injection truth and recovery
links are stored separately and are joined only after predictions have been
written.

Column names may differ between BLISS versions, so each export requires an
explicit JSON mapping to the canonical fields below.

## Required recovered-candidate fields

| Canonical field         | Meaning                                   | Canonical unit or rule                                              |
| ----------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| `candidate_id`          | Stable hit identifier within the export   | Globally unique after station prefixing                             |
| `observation_id`        | Filterbank product identifier             | Matches the observation manifest                                    |
| `simultaneous_group_id` | Pairable observing interval               | Same value at both real stations                                    |
| `station_id`            | Station identity                          | Supplied explicitly                                                 |
| `frequency_hz`          | Recovered candidate frequency             | Hz after explicit conversion                                        |
| Frequency reference     | Epoch at which `frequency_hz` is defined  | Absolute MJD for real data; normalized seconds for a labelled proxy |
| `drift_hz_s`            | Recovered signed drift                    | Hz s⁻¹ using the documented BLISS sign convention                   |
| `width_hz`              | Recovered spectral FWHM                   | Hz; other width definitions require a documented conversion         |
| `snr`                   | Recovered search statistic                | Numeric, with its definition recorded                               |
| `resampling_block_id`   | Independent background or injection block | Recommended for clustered uncertainty intervals                     |

Before freezing the mapping, confirm with the upstream search maintainer:

* whether frequency is reported at the first integration, midpoint, or another
  explicit timestamp;
* whether the drift sign represents `df/dt` in physical frequency,
  independently of the sign of SIGPROC `foff`;
* whether width is Lorentzian FWHM, fitted FWHM, a boxcar length, or another
  scale;
* the definition and threshold of the reported S/N statistic;
* the BLISS repository commit, configuration, and command used.

## Synthetic truth fields

| Field                                                          | Meaning                                                                          |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `injection_id`                                                 | Station-specific injection identifier, made globally unique by station prefixing |
| `event_id`                                                     | Physical-event identifier shared across stations for a genuine match             |
| `frequency`, reference epoch, `drift`, `width`, `snr`, `shape` | Injected quantities from the independent injection harness                       |
| `pair_label`                                                   | `1` for a shared event and `0` for a noncoincident event                         |
| `case`                                                         | Registered event class, such as `genuine_match`, `onesided`, or `independent`    |
| `population`                                                   | `detected_conditioned` or `fixed_power`                                          |
| `evaluation_cell_id`                                           | Width × profile × S/N design-cell identifier                                     |
| `resampling_block_id`                                          | Independent paired background or injection block                                 |

For a genuine dual-station event, the station-specific `injection_id` values
differ, while `event_id`, injected width, shape, and `pair_label=1` agree.
Station-specific S/N values may differ. Independent local events use different
`event_id` values and `pair_label=0`.

The `detected_conditioned` and `fixed_power` populations represent different
evaluation conditions and are reported separately.

## Candidate-to-injection link sidecar

For synthetic calibration and Test B, the injection harness should provide
candidate-to-injection links in a separate JSONL file:

```json
{"candidate_id": "SE_A:hit_0042", "recovery_link_id": "SE_A:inj_0017"}
```

False BLISS hits have no recovery link. The adapter removes truth-related
columns from operational candidate records and writes valid links to a
separate sidecar. The sidecar is not used during candidate union, extraction,
or inference.

If exact recovery links are unavailable, candidate-to-injection matching may
use the calibration-derived tolerances stored under
`recovery_link_tolerances`. These tolerances are frozen before the locked
Test-B predictions are inspected.

`link_test_b_recoveries.py` performs this matching after the prediction file
has been written. Recovery tolerances are separate from
`association_tolerances`: the former link recovered candidates to injected
events, whereas the latter associate candidate estimates between stations.

## Audited engineering-smoke convention

The completed one-event paired BLISS engineering smoke used the raw,
uncollapsed candidate catalogues from the independent Ireland and Sweden
searches. For that audit:

- candidate frequency was read from BLISS `FREQ_MHZ` at the first observation
  row;
- candidate drift was read from `DR_HZ_S`;
- the corresponding truth frequency was `f_first_MHz`; and
- a recovery association required no more than 0.3 kHz frequency separation
  and 0.007 Hz/s drift separation.

These values document the completed engineering smoke only. They do not replace
the calibration-derived recovery and inter-station association policies that
must be reviewed and frozen for the full locked Synthetic Test B.

## Observation modes

### Real simultaneous data

Real simultaneous observations use:

* `time_alignment=absolute_mjd`;
* `barycentric_status=barycentric` for both stations;
* overlapping observing intervals;
* established barycentric filterbank products, such as `*.bary*.fil` or
  audited BLISS-compatible `*.bary*.h5` exports.

### Sweden proxy experiment

Proxy experiments use:

* `time_alignment=normalized_proxy`;
* `barycentric_status=synthetic_proxy`;
* the `--allow-normalized-proxy` command-line option;
* output metadata identifying station B as a proxy.

Proxy mode aligns synthetic event coordinates by normalized offset. It does
not constitute Ireland–Sweden coincidence or barycentric validation.

## Union semantics

The operational candidate set is the union (C_A \cup C_B), rather than the
intersection.

| Detection state | Station A extraction                    | Station B extraction                    |
| --------------- | --------------------------------------- | --------------------------------------- |
| Both stations   | Use station A’s recovered parameters    | Use station B’s recovered parameters    |
| Station A only  | Use station A’s recovered parameters    | Project station A’s anchor to station B |
| Station B only  | Project station B’s anchor to station A | Use station B’s recovered parameters    |

Association is performed at a declared common frequency epoch. Frequency,
drift, and log-width tolerances are derived from the calibration set, reviewed,
and frozen before the locked Test-B run.
