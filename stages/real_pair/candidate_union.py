#!/usr/bin/env python3
"""Build the union of independently detected station candidates.

This is the operational interpretation of the candidate-union requirement:

* run BLISS independently at each station;
* deduplicate hits within each station;
* associate compatible two-station hits one-to-one;
* retain every unmatched one-station hit in the union; and
* preserve station-specific recovered parameters.

The program never reads candidate ``truth`` mappings and emits no match label.
Association is made at a common frequency reference epoch with an explicitly
frozen empirical policy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
from scipy.optimize import linear_sum_assignment

TOLERANCE_KEYS = {"frequency_hz", "drift_hz_s", "log_width"}


def load_policy(path: str, allow_draft: bool) -> Dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    status = str(policy.get("status", ""))
    if status != "frozen" and not (allow_draft and status == "empirical_draft"):
        raise ValueError(
            "association policy must be frozen; use --allow-unfrozen-policy only "
            "for development, never for locked Test B"
        )
    for section in ("association_tolerances", "deduplication_tolerances"):
        values = policy.get(section, {})
        if set(values) != TOLERANCE_KEYS:
            raise ValueError(
                "policy.%s must contain exactly %s" % (section, sorted(TOLERANCE_KEYS))
            )
        if any(float(values[key]) <= 0 for key in TOLERANCE_KEYS):
            raise ValueError("policy.%s tolerances must be positive" % section)
    return policy


def _distance_components(
    left: CandidateRecord,
    right: CandidateRecord,
    reference_kind: str,
    reference_value: float,
    tolerances: Mapping[str, float],
) -> Tuple[float, float, float]:
    frequency_delta = abs(
        candidate_frequency_at_reference(left, reference_kind, reference_value)
        - candidate_frequency_at_reference(right, reference_kind, reference_value)
    )
    drift_delta = abs(float(left.drift_hz_s) - float(right.drift_hz_s))
    log_width_delta = abs(math.log(float(left.width_hz) / float(right.width_hz)))
    return (
        frequency_delta / float(tolerances["frequency_hz"]),
        drift_delta / float(tolerances["drift_hz_s"]),
        log_width_delta / float(tolerances["log_width"]),
    )


def _compatible_cost(
    left: CandidateRecord,
    right: CandidateRecord,
    reference_kind: str,
    reference_value: float,
    tolerances: Mapping[str, float],
) -> float:
    components = _distance_components(
        left, right, reference_kind, reference_value, tolerances
    )
    if any(value > 1.0 for value in components):
        return float("inf")
    return float(sum(value * value for value in components))


def _deduplicate(
    candidates: Sequence[CandidateRecord],
    reference_kind: str,
    reference_value: float,
    tolerances: Mapping[str, float],
) -> Tuple[List[CandidateRecord], Dict[str, List[str]]]:
    """Deterministic non-maximum suppression for duplicate station hits.

    A graph-connected-components rule can over-merge a chain of hits whose
    endpoints are mutually incompatible.  Here the highest-S/N remaining hit
    is the representative and only hits directly compatible with that
    representative are absorbed.
    """

    remaining = sorted(
        candidates,
        key=lambda item: (-float(item.snr), item.candidate_id),
    )
    groups: List[List[CandidateRecord]] = []
    while remaining:
        representative = remaining[0]
        component = [representative]
        survivors = []
        for candidate in remaining[1:]:
            if np.isfinite(
                _compatible_cost(
                    representative,
                    candidate,
                    reference_kind,
                    reference_value,
                    tolerances,
                )
            ):
                component.append(candidate)
            else:
                survivors.append(candidate)
        groups.append(component)
        remaining = survivors
    representatives: List[CandidateRecord] = []
    members: Dict[str, List[str]] = {}
    for component in groups:
        representative = component[0]
        representatives.append(representative)
        members[representative.candidate_id] = sorted(
            item.candidate_id for item in component
        )
    representatives.sort(
        key=lambda item: (
            candidate_frequency_at_reference(item, reference_kind, reference_value),
            item.candidate_id,
        )
    )
    return representatives, members


def _route_for_candidates(
    candidates: Sequence[CandidateRecord], low_hz: float, high_hz: float
) -> str:
    widths = [float(item.width_hz) for item in candidates]
    if max(widths) > high_hz:
        return "mid_resolution_handoff"
    if min(widths) < low_hz and max(widths) < low_hz:
        return "narrowband_or_stage3"
    if min(widths) < low_hz:
        return "manual_width_disagreement_review"
    return "high_resolution_stage4"


def _entry(
    group_id: str,
    observations: Sequence[ObservationRecord],
    station_ids: Sequence[str],
    candidates: Sequence[CandidateRecord],
    member_map: Mapping[str, Sequence[str]],
    reference_kind: str,
    reference_value: float,
    association_cost: Any,
    low_hz: float,
    high_hz: float,
    policy_id: str,
) -> Dict[str, Any]:
    by_station = {item.station_id: item for item in candidates}
    anchor = sorted(candidates, key=lambda item: (-float(item.snr), item.candidate_id))[
        0
    ]
    station_entries: Dict[str, Any] = {}
    for station_id in station_ids:
        candidate = by_station.get(station_id)
        if candidate is None:
            station_entries[station_id] = {
                "detected": False,
                "candidate_id": None,
                "candidate": None,
                "deduplicated_member_ids": [],
            }
        else:
            station_entries[station_id] = {
                "detected": True,
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.to_dict(include_truth=False),
                "deduplicated_member_ids": list(member_map[candidate.candidate_id]),
            }
    detection_state = "two_station" if len(candidates) == 2 else "one_station"
    union_id = stable_id(
        "union",
        group_id,
        sorted(item.candidate_id for item in candidates),
        reference_kind,
        reference_value,
        policy_id,
    )
    block_ids = sorted(
        {
            str(item.extras.get("resampling_block_id", "")).strip()
            for item in candidates
            if str(item.extras.get("resampling_block_id", "")).strip()
        }
    )
    return {
        "format_version": 1,
        "union_id": union_id,
        "simultaneous_group_id": group_id,
        "station_ids": list(station_ids),
        "detection_state": detection_state,
        "reference_kind": reference_kind,
        "reference_value": float(reference_value),
        "anchor_station_id": anchor.station_id,
        "anchor_candidate_id": anchor.candidate_id,
        "anchor_frequency_hz_at_reference": candidate_frequency_at_reference(
            anchor, reference_kind, reference_value
        ),
        "association_cost": (
            None if association_cost is None else float(association_cost)
        ),
        "association_policy_id": policy_id,
        "resampling_block_id": block_ids[0] if len(block_ids) == 1 else None,
        "resampling_block_disagreement": block_ids if len(block_ids) > 1 else [],
        "route": _route_for_candidates(candidates, low_hz, high_hz),
        "stations": station_entries,
        "truth": None,
    }


def build_group_union(
    group_id: str,
    observations: Sequence[ObservationRecord],
    candidates: Sequence[CandidateRecord],
    policy: Mapping[str, Any],
    route_low_hz: float,
    route_high_hz: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reference_kind, reference_value = group_reference_time(observations)
    station_ids = sorted(item.station_id for item in observations)
    station_candidates = {
        station_id: [item for item in candidates if item.station_id == station_id]
        for station_id in station_ids
    }
    representatives: Dict[str, List[CandidateRecord]] = {}
    member_map: Dict[str, List[str]] = {}
    for station_id in station_ids:
        reps, members = _deduplicate(
            station_candidates[station_id],
            reference_kind,
            reference_value,
            policy["deduplication_tolerances"],
        )
        representatives[station_id] = reps
        member_map.update(members)

    left, right = representatives[station_ids[0]], representatives[station_ids[1]]
    costs = np.full((len(left), len(right)), np.inf, dtype=float)
    for i, candidate_left in enumerate(left):
        for j, candidate_right in enumerate(right):
            costs[i, j] = _compatible_cost(
                candidate_left,
                candidate_right,
                reference_kind,
                reference_value,
                policy["association_tolerances"],
            )
    matched_left, matched_right = set(), set()
    entries: List[Dict[str, Any]] = []
    if costs.size and np.isfinite(costs).any():
        rows, cols = linear_sum_assignment(np.where(np.isfinite(costs), costs, 1e12))
        for i, j in zip(rows, cols):
            if not np.isfinite(costs[i, j]):
                continue
            matched_left.add(i)
            matched_right.add(j)
            entries.append(
                _entry(
                    group_id,
                    observations,
                    station_ids,
                    (left[i], right[j]),
                    member_map,
                    reference_kind,
                    reference_value,
                    costs[i, j],
                    route_low_hz,
                    route_high_hz,
                    str(policy.get("policy_id", "unknown")),
                )
            )
    for i, candidate in enumerate(left):
        if i not in matched_left:
            entries.append(
                _entry(
                    group_id,
                    observations,
                    station_ids,
                    (candidate,),
                    member_map,
                    reference_kind,
                    reference_value,
                    None,
                    route_low_hz,
                    route_high_hz,
                    str(policy.get("policy_id", "unknown")),
                )
            )
    for j, candidate in enumerate(right):
        if j not in matched_right:
            entries.append(
                _entry(
                    group_id,
                    observations,
                    station_ids,
                    (candidate,),
                    member_map,
                    reference_kind,
                    reference_value,
                    None,
                    route_low_hz,
                    route_high_hz,
                    str(policy.get("policy_id", "unknown")),
                )
            )
    entries.sort(
        key=lambda item: (item["anchor_frequency_hz_at_reference"], item["union_id"])
    )
    audit = {
        "simultaneous_group_id": group_id,
        "n_input_hits_by_station": {
            key: len(value) for key, value in station_candidates.items()
        },
        "n_deduplicated_hits_by_station": {
            key: len(value) for key, value in representatives.items()
        },
        "n_union_entries": len(entries),
        "n_two_station": sum(
            item["detection_state"] == "two_station" for item in entries
        ),
        "n_one_station": sum(
            item["detection_state"] == "one_station" for item in entries
        ),
        "reference_kind": reference_kind,
        "reference_value": reference_value,
    }
    return entries, audit


def main(args: argparse.Namespace) -> None:
    if not (0 < args.route_low_hz < args.route_high_hz):
        raise ValueError("route limits must satisfy 0 < low < high")
    observations_by_key = load_observations(args.observations, require_files=False)
    policy = load_policy(args.policy, args.allow_unfrozen_policy)
    candidates: List[CandidateRecord] = []
    for path in args.candidate_files:
        candidates.extend(load_candidates(path))
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique across all input files")
    observations_by_group: Dict[str, List[ObservationRecord]] = defaultdict(list)
    for observation in observations_by_key.values():
        observations_by_group[observation.simultaneous_group_id].append(observation)
    candidates_by_group: Dict[str, List[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.detected:
            raise ValueError(
                "candidate %s is marked undetected; non-detections must be represented "
                "by absence from the station hit list" % candidate.candidate_id
            )
        key = (candidate.simultaneous_group_id, candidate.station_id)
        if key not in observations_by_key:
            raise ValueError(
                "candidate %s references unknown station/group %r"
                % (candidate.candidate_id, key)
            )
        observation = observations_by_key[key]
        if candidate.observation_id != observation.observation_id:
            raise ValueError(
                "candidate %s observation_id disagrees with manifest"
                % candidate.candidate_id
            )
        candidates_by_group[candidate.simultaneous_group_id].append(candidate)

    all_entries: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for group_id, observations in sorted(observations_by_group.items()):
        validate_group_observations(observations, args.allow_normalized_proxy)
        entries, audit = build_group_union(
            group_id,
            observations,
            candidates_by_group.get(group_id, []),
            policy,
            args.route_low_hz,
            args.route_high_hz,
        )
        all_entries.extend(entries)
        audits.append(audit)
    block_disagreements = [
        item for item in all_entries if item.get("resampling_block_disagreement")
    ]
    if block_disagreements and args.fail_on_block_disagreement:
        raise ValueError(
            "%d associated entries contain inconsistent resampling_block_id values"
            % len(block_disagreements)
        )
    write_jsonl(args.output, all_entries)
    summary = {
        "format_version": 1,
        "n_input_candidates": len(candidates),
        "n_union_entries": len(all_entries),
        "n_two_station": sum(
            item["detection_state"] == "two_station" for item in all_entries
        ),
        "n_one_station": sum(
            item["detection_state"] == "one_station" for item in all_entries
        ),
        "n_resampling_block_disagreements": len(block_disagreements),
        "routes": {
            route: sum(item["route"] == route for item in all_entries)
            for route in sorted({item["route"] for item in all_entries})
        },
        "policy_path": str(Path(args.policy).resolve()),
        "policy_sha256": sha256_file(args.policy),
        "policy_status": policy.get("status"),
        "contains_normalized_proxy": any(
            item.time_alignment == "normalized_proxy"
            for item in observations_by_key.values()
        ),
        "groups": audits,
        "scientific_boundary": (
            "A union retains station-A-only, station-B-only and two-station entries. "
            "It is not the intersection of station hit lists."
        ),
    }
    write_json(str(Path(args.output).with_suffix(".summary.json")), summary)
    print(
        "Wrote %d union entries (%d two-station, %d one-station) to %s"
        % (
            len(all_entries),
            summary["n_two_station"],
            summary["n_one_station"],
            args.output,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the two-station union of independently recovered BLISS candidates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--route-low-hz", type=float, default=10.0)
    parser.add_argument("--route-high-hz", type=float, default=100.0)
    parser.add_argument("--allow-normalized-proxy", action="store_true")
    parser.add_argument("--allow-unfrozen-policy", action="store_true")
    parser.add_argument("--fail-on-block-disagreement", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
