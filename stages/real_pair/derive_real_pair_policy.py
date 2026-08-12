#!/usr/bin/env python3
"""Derive a resolution-anchored policy for the unlabeled real-pair pilot.

This is not a substitute for injection-derived recovery errors.  It provides a
transparent, preregisterable engineering policy for LOFTS0050 before any
cross-station associations or Stage-4 scores are inspected.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from lofts_bliss_schema import load_observations, sha256_file, stable_id, write_json


def main(args: argparse.Namespace) -> None:
    observations = list(
        load_observations(args.observations, require_files=False).values()
    )
    if len(observations) != 2:
        raise ValueError("real-pair pilot policy requires exactly two observations")
    if args.frequency_base_channels <= 0 or args.drift_tolerance_bins <= 0:
        raise ValueError("frequency/drift scale factors must be positive")
    if args.frequency_width_sum_fraction < 0:
        raise ValueError("frequency_width_sum_fraction must be non-negative")
    control_shifts = [
        float(value)
        for value in str(args.control_shifts_hz).split(",")
        if value.strip()
    ]
    if (
        not control_shifts
        or any(value == 0.0 for value in control_shifts)
        or len(control_shifts) != len(set(control_shifts))
    ):
        raise ValueError("control_shifts_hz must contain unique, non-zero offsets")
    if args.control_minimum_per_pair <= 0:
        raise ValueError("control_minimum_per_pair must be positive")
    if args.control_minimum_per_pair > len(control_shifts):
        raise ValueError(
            "control_minimum_per_pair cannot exceed the number of frozen shifts"
        )
    if (
        min(
            args.control_candidate_exclusion_widths,
            args.control_candidate_exclusion_base_hz,
            args.control_edge_guard_widths,
        )
        < 0
    ):
        raise ValueError("control exclusion and edge-guard scales must be non-negative")
    banks = {tuple(item.search_bank_width_channels) for item in observations}
    if len(banks) != 1 or not next(iter(banks)):
        raise ValueError(
            "both observations must record the same non-empty template bank"
        )
    bank = next(iter(banks))
    if any(item.search_fine_channels_per_coarse <= 0 for item in observations):
        raise ValueError("both observations must record BLISS coarse-channel geometry")
    drift_steps = {
        item.station_id: abs(item.signed_foff_hz) / ((item.n_time - 1) * item.tsamp_s)
        for item in observations
    }
    maximum_adjacent_ratio = max(
        float(right) / float(left) for left, right in zip(bank[:-1], bank[1:])
    )
    station_coarse_bandwidths = {
        item.station_id: (
            item.search_fine_channels_per_coarse * abs(item.signed_foff_hz)
        )
        for item in observations
    }
    payload = {
        "format_version": 2,
        "status": "resolution_derived_draft",
        "dataset_role": "unlabeled_real_pair_engineering_pilot",
        "observation_manifest": str(Path(args.observations).resolve()),
        "observation_manifest_sha256": sha256_file(args.observations),
        "derivation": {
            "labels_used": False,
            "stage4_scores_used": False,
            "cross_station_association_counts_used": False,
            "drift_grid_definition": "abs(signed_foff_hz)/((n_time-1)*tsamp_s)",
            "drift_grid_steps_hz_s": drift_steps,
            "bank_width_channels": list(bank),
            "maximum_adjacent_template_ratio": maximum_adjacent_ratio,
            "limitation": (
                "Resolution-derived gates are for the real observational pilot only. "
                "A locked labeled Test B must replace them with blind-injection "
                "recovery errors."
            ),
        },
        "association": {
            "frequency_base_hz": args.frequency_base_channels
            * max(abs(item.signed_foff_hz) for item in observations),
            "frequency_width_sum_fraction": args.frequency_width_sum_fraction,
            "drift_hz_s": args.drift_tolerance_bins * max(drift_steps.values()),
            "log_width": math.log(maximum_adjacent_ratio) + args.log_width_epsilon,
            "component_max_cost": 3.0,
        },
        "deduplication": {
            "mode": "disabled_input_is_naoise_frequency_primary_nms",
            "reason": (
                "Naoise raw output has already undergone frequency-primary NMS. "
                "Additional deduplication could discard valid union members."
            ),
        },
        "computation": {
            "association_algorithm": "frequency_indexed_edges_component_hungarian",
            "maximum_component_nodes": args.maximum_component_nodes,
            "reason": (
                "Fail closed on an unexpectedly dense RFI component instead of "
                "allocating an unbounded dense assignment matrix."
            ),
        },
        "coverage": {
            "guard_fwhm_fraction": args.coverage_guard_fwhm_fraction,
            "counterpart_not_searched_is_not_a_non_detection": True,
        },
        "resampling": {
            "mode": "common_physical_frequency_blocks",
            "block_width_hz": max(station_coarse_bandwidths.values()),
            "station_native_coarse_bandwidth_hz": station_coarse_bandwidths,
            "reason": (
                "Candidate/RFI scores within one coarse-frequency region are "
                "correlated; real score-control uncertainty is resampled by "
                "frequency block rather than by individual candidate."
            ),
        },
        "controls": {
            "kind": "distant_frequency_shift",
            "shifts_hz": control_shifts,
            "minimum_controls_per_pair": int(args.control_minimum_per_pair),
            "candidate_exclusion_width_sum_fraction": float(
                args.control_candidate_exclusion_widths
            ),
            "candidate_exclusion_base_hz": float(
                args.control_candidate_exclusion_base_hz
            ),
            "edge_guard_widths": float(args.control_edge_guard_widths),
            "station_rule": (
                "shift_non_reporting_counterpart_for_one_station_entries_and_"
                "each_station_separately_for_two_station_entries"
            ),
            "labels_used": False,
            "frozen_before_union_and_stage4_scores": True,
            "interpretation": (
                "Shifted views are paired empirical score controls, not "
                "astrophysical negative labels."
            ),
        },
        "routing": {
            "width_mode": args.width_mode,
            "stage4_width_min_hz": args.stage4_width_min_hz,
            "stage4_width_max_hz": args.stage4_width_max_hz,
        },
    }
    payload["policy_id"] = stable_id("real_policy_draft", payload)
    write_json(args.output, payload)
    print("Wrote resolution-derived real-pair policy draft to %s" % args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a preregisterable unlabeled real-pair pilot policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frequency-base-channels", type=float, default=2.0)
    parser.add_argument("--frequency-width-sum-fraction", type=float, default=1.0)
    parser.add_argument("--drift-tolerance-bins", type=float, default=2.0)
    parser.add_argument("--log-width-epsilon", type=float, default=1e-6)
    parser.add_argument("--coverage-guard-fwhm-fraction", type=float, default=0.5)
    parser.add_argument("--maximum-component-nodes", type=int, default=512)
    parser.add_argument(
        "--control-shifts-hz",
        default="-300000,-100000,100000,300000",
    )
    parser.add_argument("--control-minimum-per-pair", type=int, default=2)
    parser.add_argument("--control-candidate-exclusion-widths", type=float, default=4.0)
    parser.add_argument(
        "--control-candidate-exclusion-base-hz", type=float, default=6.0
    )
    parser.add_argument("--control-edge-guard-widths", type=float, default=4.0)
    parser.add_argument(
        "--width-mode", choices=("native", "restricted_subbank"), default="native"
    )
    parser.add_argument("--stage4-width-min-hz", type=float, default=10.0)
    parser.add_argument("--stage4-width-max-hz", type=float, default=100.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
