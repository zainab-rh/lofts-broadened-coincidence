#!/usr/bin/env python3
"""Scalable union construction for real Naoise BLISS catalogs.

Unlike the original Synthetic-Test-B implementation, this program is designed
for thousands of already-NMS candidates.  It constructs only frequency-local
compatibility edges, solves one-to-one assignment within small connected
components, retains every unmatched candidate, and audits whether a projected
counterpart was actually inside the other station's BLISS clean search band.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lofts_bliss_schema import (
    CandidateRecord,
    ObservationRecord,
    candidate_frequency_at_reference,
    group_reference_time,
    load_candidates,
    load_observations,
    sha256_file,
    stable_id,
    validate_group_observations,
    write_json,
    write_jsonl,
)
from real_pair_geometry import (
    common_frequency_bounds_hz,
    search_coverage_audit,
    stage4_width_from_candidate,
)
from scipy.optimize import linear_sum_assignment


def load_policy(path: str) -> Dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_real_pair_pilot":
        raise ValueError("real-pair policy must have status=frozen_real_pair_pilot")
    association = policy.get("association", {})
    required = {
        "frequency_base_hz",
        "frequency_width_sum_fraction",
        "drift_hz_s",
        "log_width",
        "component_max_cost",
    }
    if set(association) != required:
        raise ValueError(
            "policy.association must contain exactly %s" % sorted(required)
        )
    if any(
        float(association[key]) <= 0
        for key in required
        if key != "frequency_width_sum_fraction"
    ):
        raise ValueError("association scales must be positive")
    if float(association["frequency_width_sum_fraction"]) < 0:
        raise ValueError("frequency_width_sum_fraction must be non-negative")
    if (
        policy.get("deduplication", {}).get("mode")
        != "disabled_input_is_naoise_frequency_primary_nms"
    ):
        raise ValueError("real raw catalog must retain Naoise's already-NMS candidates")
    maximum_component = int(
        policy.get("computation", {}).get("maximum_component_nodes", 512)
    )
    if maximum_component < 2:
        raise ValueError("computation.maximum_component_nodes must be at least two")
    resampling = policy.get("resampling", {})
    if resampling.get("mode") != "common_physical_frequency_blocks":
        raise ValueError(
            "policy.resampling.mode must be common_physical_frequency_blocks"
        )
    if float(resampling.get("block_width_hz", 0.0)) <= 0:
        raise ValueError("policy.resampling.block_width_hz must be positive")
    return policy


def compatibility(
    left: CandidateRecord,
    right: CandidateRecord,
    reference_kind: str,
    reference_value: float,
    policy: Mapping[str, Any],
) -> Optional[Dict[str, float]]:
    config = policy["association"]
    left_frequency = candidate_frequency_at_reference(
        left, reference_kind, reference_value
    )
    right_frequency = candidate_frequency_at_reference(
        right, reference_kind, reference_value
    )
    frequency_delta = abs(left_frequency - right_frequency)
    frequency_gate = float(config["frequency_base_hz"]) + float(
        config["frequency_width_sum_fraction"]
    ) * 0.5 * (float(left.width_hz) + float(right.width_hz))
    drift_delta = abs(float(left.drift_hz_s) - float(right.drift_hz_s))
    drift_gate = float(config["drift_hz_s"])
    log_width_delta = abs(math.log(float(left.width_hz) / float(right.width_hz)))
    log_width_gate = float(config["log_width"])
    normalized = (
        frequency_delta / frequency_gate,
        drift_delta / drift_gate,
        log_width_delta / log_width_gate,
    )
    if any(value > 1.0 for value in normalized):
        return None
    cost = float(sum(value * value for value in normalized))
    if cost > float(config["component_max_cost"]):
        return None
    return {
        "cost": cost,
        "frequency_delta_hz": float(frequency_delta),
        "frequency_gate_hz": float(frequency_gate),
        "drift_delta_hz_s": float(drift_delta),
        "drift_gate_hz_s": float(drift_gate),
        "log_width_delta": float(log_width_delta),
        "log_width_gate": float(log_width_gate),
    }


def sparse_edges(
    left: Sequence[CandidateRecord],
    right: Sequence[CandidateRecord],
    reference_kind: str,
    reference_value: float,
    policy: Mapping[str, Any],
) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], int]:
    right_frequencies = [
        candidate_frequency_at_reference(item, reference_kind, reference_value)
        for item in right
    ]
    if right_frequencies != sorted(right_frequencies):
        raise ValueError("right candidates must be frequency sorted")
    config = policy["association"]
    maximum_right_width = max((item.width_hz for item in right), default=0.0)
    maximum_left_width = max((item.width_hz for item in left), default=0.0)
    global_radius = float(config["frequency_base_hz"]) + float(
        config["frequency_width_sum_fraction"]
    ) * 0.5 * (maximum_left_width + maximum_right_width)
    edges: Dict[Tuple[int, int], Dict[str, float]] = {}
    comparisons = 0
    for left_index, left_item in enumerate(left):
        frequency = candidate_frequency_at_reference(
            left_item, reference_kind, reference_value
        )
        start = bisect.bisect_left(right_frequencies, frequency - global_radius)
        stop = bisect.bisect_right(right_frequencies, frequency + global_radius)
        for right_index in range(start, stop):
            comparisons += 1
            audit = compatibility(
                left_item,
                right[right_index],
                reference_kind,
                reference_value,
                policy,
            )
            if audit is not None:
                edges[(left_index, right_index)] = audit
    return edges, comparisons


def component_assignment(
    n_left: int,
    n_right: int,
    edges: Mapping[Tuple[int, int], Mapping[str, float]],
    maximum_component_nodes: int = 512,
) -> List[Tuple[int, int, Mapping[str, float]]]:
    left_to_right: Dict[int, List[int]] = defaultdict(list)
    right_to_left: Dict[int, List[int]] = defaultdict(list)
    for left_index, right_index in edges:
        left_to_right[left_index].append(right_index)
        right_to_left[right_index].append(left_index)
    visited_left, visited_right = set(), set()
    matches: List[Tuple[int, int, Mapping[str, float]]] = []
    for seed in sorted(left_to_right):
        if seed in visited_left:
            continue
        component_left, component_right = set(), set()
        queue = deque([("left", seed)])
        while queue:
            side, index = queue.popleft()
            if side == "left":
                if index in component_left:
                    continue
                component_left.add(index)
                visited_left.add(index)
                queue.extend(("right", value) for value in left_to_right[index])
            else:
                if index in component_right:
                    continue
                component_right.add(index)
                visited_right.add(index)
                queue.extend(("left", value) for value in right_to_left[index])
        left_values = sorted(component_left)
        right_values = sorted(component_right)
        if len(left_values) + len(right_values) > int(maximum_component_nodes):
            raise RuntimeError(
                "association component has %d candidates, exceeding the frozen "
                "safety limit of %d; inspect this frequency region rather than "
                "allocating an unbounded dense assignment matrix"
                % (len(left_values) + len(right_values), maximum_component_nodes)
            )
        matrix = np.full((len(left_values), len(right_values)), np.inf, dtype=float)
        left_position = {value: index for index, value in enumerate(left_values)}
        right_position = {value: index for index, value in enumerate(right_values)}
        # Traverse only edges adjacent to this component.  Scanning the complete
        # edge dictionary for every component would quietly become O(E^2) on the
        # thousands-candidate LOFTS0050 catalogs.
        for left_index in left_values:
            for right_index in left_to_right[left_index]:
                if right_index in component_right:
                    matrix[left_position[left_index], right_position[right_index]] = (
                        float(edges[(left_index, right_index)]["cost"])
                    )
        rows, columns = linear_sum_assignment(
            np.where(np.isfinite(matrix), matrix, 1e12)
        )
        for row, column in zip(rows, columns):
            if not np.isfinite(matrix[row, column]):
                continue
            left_index = left_values[row]
            right_index = right_values[column]
            matches.append((left_index, right_index, edges[(left_index, right_index)]))
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches


def route_for_widths(
    candidates: Sequence[CandidateRecord], policy: Mapping[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    routing = policy["routing"]
    width_mode = str(routing["width_mode"])
    widths, sources, errors = [], [], []
    for candidate in candidates:
        try:
            width, source = stage4_width_from_candidate(candidate, mode=width_mode)
            widths.append(float(width))
            sources.append(source)
        except Exception as exc:
            errors.append("%s: %s" % (candidate.candidate_id, exc))
    audit = {
        "width_mode": width_mode,
        "widths_hz": widths,
        "width_sources": sources,
        "errors": errors,
    }
    if errors or not widths:
        return "no_stage4_eligible_width", audit
    low = float(routing["stage4_width_min_hz"])
    high = float(routing["stage4_width_max_hz"])
    if all(value < low for value in widths):
        return "narrowband_or_stage3", audit
    if all(value > high for value in widths):
        return "mid_resolution_handoff", audit
    if any(value < low or value > high for value in widths):
        return "manual_width_disagreement_review", audit
    return "high_resolution_stage4", audit


def build_entry(
    group_id: str,
    observations: Sequence[ObservationRecord],
    candidates: Sequence[CandidateRecord],
    reference_kind: str,
    reference_value: float,
    policy: Mapping[str, Any],
    association_audit: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    station_ids = sorted(item.station_id for item in observations)
    observation_by_station = {item.station_id: item for item in observations}
    candidate_by_station = {item.station_id: item for item in candidates}
    anchor = sorted(candidates, key=lambda item: (-item.snr, item.candidate_id))[0]
    coverage_guard = float(policy["coverage"]["guard_fwhm_fraction"])
    stations: Dict[str, Any] = {}
    for station_id in station_ids:
        detected_candidate = candidate_by_station.get(station_id)
        metadata_candidate = detected_candidate or anchor
        coverage = search_coverage_audit(
            metadata_candidate,
            observation_by_station[station_id],
            guard_fwhm_fraction=coverage_guard,
        )
        try:
            preprocessing_width_hz, preprocessing_width_source = (
                stage4_width_from_candidate(
                    metadata_candidate, mode=str(policy["routing"]["width_mode"])
                )
            )
        except Exception as exc:
            preprocessing_width_hz = None
            preprocessing_width_source = "unavailable: %s" % exc
        stations[station_id] = {
            "detected": detected_candidate is not None,
            "candidate_id": (
                None if detected_candidate is None else detected_candidate.candidate_id
            ),
            "candidate": (
                None
                if detected_candidate is None
                else detected_candidate.to_dict(include_truth=False)
            ),
            "metadata_anchor_candidate_id": metadata_candidate.candidate_id,
            "metadata_anchor_station_id": metadata_candidate.station_id,
            "metadata_anchor": metadata_candidate.to_dict(include_truth=False),
            "preprocessing_width_hz": preprocessing_width_hz,
            "preprocessing_width_source": preprocessing_width_source,
            "coverage": coverage,
        }
    if len(candidates) == 2:
        detection_state = "two_station"
        operational_eligibility = "two_station_detected"
    else:
        detection_state = "%s_only" % anchor.station_id
        counterpart = next(
            value for key, value in stations.items() if key != anchor.station_id
        )
        operational_eligibility = (
            "eligible_one_station_non_detection"
            if counterpart["coverage"]["searched_clean_band_covered"]
            else "counterpart_not_searched"
        )
    route, width_audit = route_for_widths(candidates, policy)
    if (
        route == "high_resolution_stage4"
        and operational_eligibility == "counterpart_not_searched"
    ):
        route = "high_resolution_stage4_counterpart_not_searched"
    if any(not value["coverage"]["full_data_covered"] for value in stations.values()):
        route = "outside_common_filterbank_coverage"
    policy_id = str(policy["policy_id"])
    common_low_hz, _ = common_frequency_bounds_hz(observations)
    block_width_hz = float(policy["resampling"]["block_width_hz"])
    anchor_frequency_at_reference = candidate_frequency_at_reference(
        anchor, reference_kind, reference_value
    )
    frequency_block_index = int(
        math.floor((anchor_frequency_at_reference - common_low_hz) / block_width_hz)
    )
    resampling_block_id = "%s:frequency_block:%d" % (
        group_id,
        frequency_block_index,
    )
    union_id = stable_id(
        "real_union",
        group_id,
        sorted(item.candidate_id for item in candidates),
        reference_kind,
        reference_value,
        policy_id,
    )
    return {
        "format_version": 2,
        "union_id": union_id,
        "simultaneous_group_id": group_id,
        "station_ids": station_ids,
        "detection_state": detection_state,
        "operational_eligibility": operational_eligibility,
        "reference_kind": reference_kind,
        "reference_value": float(reference_value),
        "anchor_station_id": anchor.station_id,
        "anchor_candidate_id": anchor.candidate_id,
        "anchor_frequency_hz_at_reference": anchor_frequency_at_reference,
        "resampling_block_id": resampling_block_id,
        "resampling_block_width_hz": block_width_hz,
        "resampling_frequency_block_index": frequency_block_index,
        "association": None if association_audit is None else dict(association_audit),
        "association_policy_id": policy_id,
        "route": route,
        "width_routing": width_audit,
        "stations": stations,
        "contains_broadband_rfi_like": any(
            bool(item.extras.get("broadband_rfi_like", False)) for item in candidates
        ),
        "truth": None,
    }


def build_group(
    group_id: str,
    observations: Sequence[ObservationRecord],
    candidates: Sequence[CandidateRecord],
    policy: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    validate_group_observations(observations, allow_normalized_proxy=False)
    station_ids = sorted(item.station_id for item in observations)
    if len(station_ids) != 2:
        raise ValueError("real union requires exactly two stations")
    reference_kind, reference_value = group_reference_time(observations)
    common_low, common_high = common_frequency_bounds_hz(observations)
    by_station = {
        station_id: sorted(
            [item for item in candidates if item.station_id == station_id],
            key=lambda item: (
                candidate_frequency_at_reference(item, reference_kind, reference_value),
                item.candidate_id,
            ),
        )
        for station_id in station_ids
    }
    left, right = by_station[station_ids[0]], by_station[station_ids[1]]
    edges, comparisons = sparse_edges(
        left, right, reference_kind, reference_value, policy
    )
    matched = component_assignment(
        len(left),
        len(right),
        edges,
        maximum_component_nodes=int(
            policy.get("computation", {}).get("maximum_component_nodes", 512)
        ),
    )
    matched_left = {item[0] for item in matched}
    matched_right = {item[1] for item in matched}
    entries = [
        build_entry(
            group_id,
            observations,
            (left[left_index], right[right_index]),
            reference_kind,
            reference_value,
            policy,
            audit,
        )
        for left_index, right_index, audit in matched
    ]
    entries.extend(
        build_entry(
            group_id,
            observations,
            (candidate,),
            reference_kind,
            reference_value,
            policy,
            None,
        )
        for index, candidate in enumerate(left)
        if index not in matched_left
    )
    entries.extend(
        build_entry(
            group_id,
            observations,
            (candidate,),
            reference_kind,
            reference_value,
            policy,
            None,
        )
        for index, candidate in enumerate(right)
        if index not in matched_right
    )
    entries.sort(
        key=lambda item: (item["anchor_frequency_hz_at_reference"], item["union_id"])
    )
    audit = {
        "simultaneous_group_id": group_id,
        "station_ids": station_ids,
        "input_candidates_by_station": {
            station_id: len(by_station[station_id]) for station_id in station_ids
        },
        "common_frequency_low_hz": common_low,
        "common_frequency_high_hz": common_high,
        "candidate_pair_comparisons": comparisons,
        "compatible_sparse_edges": len(edges),
        "two_station_assignments": len(matched),
        "quadratic_pair_count_avoided": len(left) * len(right),
    }
    return entries, audit


def main(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    observations_by_key = load_observations(args.observations, require_files=False)
    candidates: List[CandidateRecord] = []
    for path in args.candidate_files:
        candidates.extend(load_candidates(path))
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique across stations")
    observations_by_group: Dict[str, List[ObservationRecord]] = defaultdict(list)
    for observation in observations_by_key.values():
        observations_by_group[observation.simultaneous_group_id].append(observation)
    candidates_by_group: Dict[str, List[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.simultaneous_group_id, candidate.station_id)
        if key not in observations_by_key:
            raise ValueError(
                "candidate %s references an unknown observation"
                % candidate.candidate_id
            )
        if candidate.observation_id != observations_by_key[key].observation_id:
            raise ValueError(
                "candidate %s observation ID mismatch" % candidate.candidate_id
            )
        candidates_by_group[candidate.simultaneous_group_id].append(candidate)

    entries, group_audits = [], []
    for group_id, observations in sorted(observations_by_group.items()):
        group_entries, group_audit = build_group(
            group_id,
            observations,
            candidates_by_group.get(group_id, []),
            policy,
        )
        entries.extend(group_entries)
        group_audits.append(group_audit)
    write_jsonl(args.output, entries)
    summary = {
        "format_version": 2,
        "dataset_role": "unlabeled_real_pair_candidate_union",
        "n_input_candidates": len(candidates),
        "n_union_entries": len(entries),
        "detection_states": dict(
            sorted(Counter(item["detection_state"] for item in entries).items())
        ),
        "operational_eligibility": dict(
            sorted(Counter(item["operational_eligibility"] for item in entries).items())
        ),
        "routes": dict(sorted(Counter(item["route"] for item in entries).items())),
        "n_resampling_frequency_blocks": len(
            {item["resampling_block_id"] for item in entries}
        ),
        "resampling_block_sizes": dict(
            sorted(Counter(item["resampling_block_id"] for item in entries).items())
        ),
        "n_contains_broadband_rfi_like": sum(
            bool(item["contains_broadband_rfi_like"]) for item in entries
        ),
        "policy": str(Path(args.policy).resolve()),
        "policy_sha256": sha256_file(args.policy),
        "policy_id": policy["policy_id"],
        "policy_status": policy["status"],
        "groups": group_audits,
        "labels_used": False,
        "scientific_boundary": (
            "This is a union of real unlabeled BLISS candidates. A one-station "
            "non-detection is operationally eligible only when the counterpart "
            "track was inside the other station's recorded BLISS clean search band."
        ),
    }
    write_json(str(Path(args.output).with_suffix(".summary.json")), summary)
    print(
        "Wrote %d real union entries (%d two-station) to %s"
        % (
            len(entries),
            sum(item["detection_state"] == "two_station" for item in entries),
            args.output,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sparse, rolloff-aware real Ireland-Sweden candidate union",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
