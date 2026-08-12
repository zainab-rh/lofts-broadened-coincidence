#!/usr/bin/env python3
"""Create label-free frequency-shift controls for real Stage-4 pairs.

One station's candidate-centred view is replaced by a distant, preregistered
frequency location from the same observation and same time interval.  The
other station is unchanged.  This supplies a paired empirical null for score
diagnostics without inventing match labels or changing the frozen model.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from extract_candidate_pairs import _atomic_savez, _frequency_window, _plot_qa
from lofts_bliss_schema import (
    CandidateRecord,
    load_candidates,
    load_observations,
    read_json_records,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from lofts_filterbank import read_filterbank_window
from real_pair_geometry import search_coverage_audit

DEFAULT_CONTROL_CONFIGURATION = {
    "shifts_hz": [-300000.0, -100000.0, 100000.0, 300000.0],
    "minimum_controls_per_pair": 2,
    "candidate_exclusion_width_sum_fraction": 4.0,
    "candidate_exclusion_base_hz": 6.0,
    "edge_guard_widths": 4.0,
}


def _parse_shifts(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(item) for item in str(value).split(",") if item.strip()]


def configure_control_policy(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Apply and audit the control plan frozen before real-pair scoring.

    Direct CLI overrides remain available for generic development use when no
    policy is supplied. In the real LOFTS0050 workflow, ``--policy`` makes any
    conflicting post-freeze override a hard error.
    """

    if not args.policy:
        if args.shifts_hz is None:
            args.shifts_hz = ",".join(
                "%g" % value for value in DEFAULT_CONTROL_CONFIGURATION["shifts_hz"]
            )
        if args.minimum_controls_per_pair is None:
            args.minimum_controls_per_pair = int(
                DEFAULT_CONTROL_CONFIGURATION["minimum_controls_per_pair"]
            )
        if args.candidate_exclusion_widths is None:
            args.candidate_exclusion_widths = float(
                DEFAULT_CONTROL_CONFIGURATION["candidate_exclusion_width_sum_fraction"]
            )
        if args.candidate_exclusion_base_hz is None:
            args.candidate_exclusion_base_hz = float(
                DEFAULT_CONTROL_CONFIGURATION["candidate_exclusion_base_hz"]
            )
        if args.edge_guard_widths is None:
            args.edge_guard_widths = float(
                DEFAULT_CONTROL_CONFIGURATION["edge_guard_widths"]
            )
        return None

    policy_path = Path(args.policy).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "frozen_real_pair_pilot":
        raise ValueError("control policy must have status=frozen_real_pair_pilot")
    control = policy.get("controls")
    if not isinstance(control, dict):
        raise ValueError("frozen real-pair policy has no controls section")
    if control.get("kind") != "distant_frequency_shift":
        raise ValueError("unsupported frozen control kind %r" % control.get("kind"))
    expected_station_rule = (
        "shift_non_reporting_counterpart_for_one_station_entries_and_"
        "each_station_separately_for_two_station_entries"
    )
    if control.get("station_rule") != expected_station_rule:
        raise ValueError("frozen control station rule is unsupported")
    if args.shift_station is not None:
        raise ValueError(
            "--shift-station conflicts with the frozen automatic station rule"
        )

    frozen_values = {
        "shifts_hz": [float(value) for value in control["shifts_hz"]],
        "minimum_controls_per_pair": int(control["minimum_controls_per_pair"]),
        "candidate_exclusion_widths": float(
            control["candidate_exclusion_width_sum_fraction"]
        ),
        "candidate_exclusion_base_hz": float(control["candidate_exclusion_base_hz"]),
        "edge_guard_widths": float(control["edge_guard_widths"]),
    }
    requested_shifts = _parse_shifts(args.shifts_hz)
    if requested_shifts is not None and requested_shifts != frozen_values["shifts_hz"]:
        raise ValueError("--shifts-hz conflicts with the frozen control policy")
    for attribute in (
        "minimum_controls_per_pair",
        "candidate_exclusion_widths",
        "candidate_exclusion_base_hz",
        "edge_guard_widths",
    ):
        requested = getattr(args, attribute)
        if requested is not None and float(requested) != float(
            frozen_values[attribute]
        ):
            raise ValueError(
                "--%s conflicts with the frozen control policy"
                % attribute.replace("_", "-")
            )
    args.shifts_hz = ",".join("%g" % value for value in frozen_values["shifts_hz"])
    args.minimum_controls_per_pair = frozen_values["minimum_controls_per_pair"]
    args.candidate_exclusion_widths = frozen_values["candidate_exclusion_widths"]
    args.candidate_exclusion_base_hz = frozen_values["candidate_exclusion_base_hz"]
    args.edge_guard_widths = frozen_values["edge_guard_widths"]
    return {
        "path": str(policy_path),
        "sha256": sha256_file(str(policy_path)),
        "policy_id": policy.get("policy_id"),
        "controls": control,
    }


class CandidateFrequencyIndex:
    """Lazy physical-frequency index for clean-control exclusion queries."""

    def __init__(self, candidates: List[CandidateRecord]):
        self._groups: Dict[Tuple[str, str], List[CandidateRecord]] = {}
        grouped: Dict[Tuple[str, str], List[CandidateRecord]] = {}
        for candidate in candidates:
            grouped.setdefault(
                (candidate.simultaneous_group_id, candidate.station_id), []
            ).append(candidate)
        self._groups = grouped
        self._cache: Dict[
            Tuple[str, str, str, float],
            Tuple[List[float], List[CandidateRecord], float],
        ] = {}

    def _at_reference(
        self,
        group_id: str,
        station_id: str,
        reference_kind: str,
        reference_value: float,
    ) -> Tuple[List[float], List[CandidateRecord], float]:
        key = (
            str(group_id),
            str(station_id),
            str(reference_kind),
            float(reference_value),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        candidates = self._groups.get((str(group_id), str(station_id)), [])
        positioned = []
        for candidate in candidates:
            frequency = (
                candidate.frequency_at_mjd(reference_value)
                if reference_kind == "mjd"
                else candidate.frequency_at_offset_s(reference_value)
            )
            positioned.append((float(frequency), candidate))
        positioned.sort(key=lambda item: (item[0], item[1].candidate_id))
        frequencies = [item[0] for item in positioned]
        ordered = [item[1] for item in positioned]
        maximum_width = max((float(item.width_hz) for item in ordered), default=0.0)
        result = (frequencies, ordered, maximum_width)
        self._cache[key] = result
        return result

    def first_conflict(
        self,
        group_id: str,
        station_id: str,
        reference_kind: str,
        reference_value: float,
        control_frequency_hz: float,
        control_width_hz: float,
        base_hz: float,
        width_sum_fraction: float,
    ):
        frequencies, candidates, maximum_width = self._at_reference(
            group_id, station_id, reference_kind, reference_value
        )
        maximum_gate = float(base_hz) + float(width_sum_fraction) * 0.5 * (
            float(control_width_hz) + maximum_width
        )
        start = bisect.bisect_left(frequencies, control_frequency_hz - maximum_gate)
        stop = bisect.bisect_right(frequencies, control_frequency_hz + maximum_gate)
        conflicts = []
        for index in range(start, stop):
            candidate = candidates[index]
            separation = abs(frequencies[index] - float(control_frequency_hz))
            gate = float(base_hz) + float(width_sum_fraction) * 0.5 * (
                float(control_width_hz) + float(candidate.width_hz)
            )
            if separation <= gate:
                conflicts.append((separation, candidate.candidate_id, gate))
        return min(conflicts) if conflicts else None


def _load_primary_arrays(record: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    path = str(record["array_file"])
    if sha256_file(path) != str(record["array_sha256"]):
        raise ValueError("primary pair checksum mismatch: %s" % path)
    with np.load(path, allow_pickle=False) as archive:
        raw_a = np.asarray(archive["raw_a"], dtype=np.float32)
        raw_b = np.asarray(archive["raw_b"], dtype=np.float32)
    return raw_a, raw_b


def make_control(
    record: Mapping[str, Any],
    observations,
    shift_station: str,
    shift_hz: float,
    arrays_dir: Path,
    edge_guard_widths: float,
    allow_masked_data: bool,
    overwrite: bool,
    catalog_index: CandidateFrequencyIndex,
    candidate_exclusion_widths: float,
    candidate_exclusion_base_hz: float,
    primary_arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    station_order = list(record["station_order"])
    if shift_station not in station_order:
        raise ValueError("shift station %s is absent from pair" % shift_station)
    raw_a, raw_b = (
        _load_primary_arrays(record)
        if primary_arrays is None
        else (primary_arrays[0].copy(), primary_arrays[1].copy())
    )
    station = copy.deepcopy(record["stations"][shift_station])
    observation = observations[(record["simultaneous_group_id"], shift_station)]
    shifted_frequency_hz = float(
        station["candidate_frequency_hz_at_local_reference"]
    ) + float(shift_hz)
    reference_kind = str(station["actual_reference_kind"])
    reference_value = float(station["actual_reference_value"])
    control_candidate = CandidateRecord(
        candidate_id=stable_id(
            "control_anchor", record["pair_id"], shift_station, float(shift_hz)
        ),
        observation_id=observation.observation_id,
        simultaneous_group_id=observation.simultaneous_group_id,
        station_id=observation.station_id,
        frequency_hz=shifted_frequency_hz,
        drift_hz_s=float(station["reported_drift_hz_s"]),
        width_hz=float(station["reported_width_fwhm_hz"]),
        width_definition="frequency-shift control inherits operational BLISS width",
        snr=0.0,
        snr_definition="not applicable to a label-free frequency-shift control",
        frequency_ref_mjd=(reference_value if reference_kind == "mjd" else None),
        frequency_ref_offset_s=(
            reference_value if reference_kind == "offset_s" else None
        ),
        detected=False,
        metadata_source="preregistered_frequency_shift_control",
        extras={},
        truth={},
    )
    control_candidate.validate()
    coverage = search_coverage_audit(
        control_candidate, observation, guard_fwhm_fraction=0.5
    )
    if not coverage["searched_clean_band_covered"]:
        raise ValueError(
            "shifted control is not in the station's BLISS clean search band: %s"
            % coverage["reason"]
        )
    conflict = catalog_index.first_conflict(
        record["simultaneous_group_id"],
        shift_station,
        reference_kind,
        reference_value,
        shifted_frequency_hz,
        float(control_candidate.width_hz),
        candidate_exclusion_base_hz,
        candidate_exclusion_widths,
    )
    if conflict is not None:
        separation, candidate_id, gate = conflict
        raise ValueError(
            "shifted control is within %.3f Hz of known candidate %s "
            "(exclusion gate %.3f Hz)" % (separation, candidate_id, gate)
        )
    n_rows = int(station["n_rows"])
    n_cols = int(station["n_cols"])
    reference_row = float(station["candidate_reference_row"])
    f_start, local_center, containment = _frequency_window(
        observation,
        shifted_frequency_hz,
        float(station["reported_drift_hz_s"]),
        float(station["reported_width_fwhm_hz"]),
        n_rows,
        n_cols,
        reference_row,
        edge_guard_widths,
    )
    shifted_raw, read_audit = read_filterbank_window(
        observation,
        t_start=int(station["t_start"]),
        f_start=f_start,
        n_rows=n_rows,
        n_cols=n_cols,
        if_index=int(station["if_index"]),
        fail_on_masked=not allow_masked_data,
        return_audit=True,
    )
    if station_order.index(shift_station) == 0:
        raw_a = shifted_raw
    else:
        raw_b = shifted_raw
    control_id = stable_id(
        "freq_control", record["pair_id"], shift_station, float(shift_hz)
    )
    pair_path = arrays_dir / (control_id + ".npz")
    if pair_path.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite %s" % pair_path)
    _atomic_savez(
        pair_path,
        raw_a=raw_a,
        raw_b=raw_b,
        station_ids=np.asarray(station_order),
        union_id=np.asarray(str(record["union_id"])),
    )
    stations = copy.deepcopy(record["stations"])
    stations[shift_station].update(
        {
            "f_start": f_start,
            "candidate_frequency_hz_at_local_reference": shifted_frequency_hz,
            "candidate_center_channel": local_center,
            "read_audit": read_audit,
            "control_frequency_shift_hz": float(shift_hz),
            "control_search_coverage": coverage,
            **containment,
        }
    )
    output = dict(record)
    output.update(
        {
            "format_version": 2,
            "pair_id": control_id,
            "source_pair_id": str(record["pair_id"]),
            "source_detection_state": record["detection_state"],
            "detection_state": "frequency_shift_control",
            "control_kind": "frequency_shift",
            "control_shift_station_id": shift_station,
            "control_shift_hz": float(shift_hz),
            "array_file": str(pair_path.resolve()),
            "array_sha256": sha256_file(str(pair_path)),
            "stations": stations,
            "truth": None,
        }
    )
    return output, raw_a, raw_b


def main(args: argparse.Namespace) -> None:
    frozen_control_policy = configure_control_policy(args)
    shifts = [float(value) for value in args.shifts_hz.split(",") if value.strip()]
    if (
        not shifts
        or any(value == 0 for value in shifts)
        or len(shifts) != len(set(shifts))
    ):
        raise ValueError("--shifts-hz must contain unique non-zero values")
    observations = load_observations(args.observations, require_files=True)
    primary = read_json_records(args.pair_manifest)
    catalog_candidates: List[CandidateRecord] = []
    for path in args.candidate_files:
        catalog_candidates.extend(load_candidates(path))
    catalog_index = CandidateFrequencyIndex(catalog_candidates)
    out_dir = Path(args.out_dir)
    arrays_dir = out_dir / "pairs"
    qa_dir = out_dir / "qa_shifted_pairs"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    if args.qa_count:
        qa_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    requested = 0
    for source in primary:
        primary_arrays = _load_primary_arrays(source)
        if args.shift_station:
            shift_stations = [args.shift_station]
        elif source.get("detection_state") == "two_station":
            shift_stations = list(source["station_order"])
        else:
            shift_stations = [
                value
                for value in source["station_order"]
                if value != source["anchor_station_id"]
            ]
        if not shift_stations:
            raise ValueError(
                "could not determine a counterpart station for %s" % source["pair_id"]
            )
        for shift_station in shift_stations:
            for shift_hz in shifts:
                requested += 1
                try:
                    record, raw_a, raw_b = make_control(
                        source,
                        observations,
                        shift_station,
                        shift_hz,
                        arrays_dir,
                        args.edge_guard_widths,
                        args.allow_masked_data,
                        args.overwrite,
                        catalog_index,
                        args.candidate_exclusion_widths,
                        args.candidate_exclusion_base_hz,
                        primary_arrays=primary_arrays,
                    )
                    records.append(record)
                    if len(records) <= args.qa_count:
                        _plot_qa(
                            qa_dir / (record["pair_id"] + ".png"),
                            raw_a,
                            raw_b,
                            "%s | shift %s %+.0f Hz"
                            % (source["pair_id"], shift_station, shift_hz),
                        )
                except Exception as exc:
                    excluded.append(
                        {
                            "source_pair_id": source.get("pair_id"),
                            "shift_station": shift_station,
                            "shift_hz": shift_hz,
                            "reason": "%s: %s" % (type(exc).__name__, exc),
                        }
                    )
                    if args.require_all_controls:
                        raise
    write_jsonl(str(out_dir / "control_pair_manifest.jsonl"), records)
    write_jsonl(str(out_dir / "excluded_controls.jsonl"), excluded)
    counts = Counter(item["control_shift_hz"] for item in records)
    per_source = Counter(item["source_pair_id"] for item in records)
    insufficient = [
        str(item["pair_id"])
        for item in primary
        if per_source[str(item["pair_id"])] < args.minimum_controls_per_pair
    ]
    summary = {
        "format_version": 2,
        "n_primary_pairs": len(primary),
        "n_requested_controls": requested,
        "n_extracted_controls": len(records),
        "n_excluded_controls": len(excluded),
        "shift_station": args.shift_station,
        "shift_rule": (
            "fixed_station"
            if args.shift_station
            else "non_reporting_counterpart_for_one_station_and_each_station_for_two_station"
        ),
        "shifts_hz": shifts,
        "extracted_by_shift": {
            str(key): value for key, value in sorted(counts.items())
        },
        "minimum_controls_per_pair": args.minimum_controls_per_pair,
        "n_primary_below_minimum_controls": len(insufficient),
        "primary_below_minimum_controls": insufficient,
        "known_candidate_exclusion": {
            "candidate_files": [
                str(Path(value).resolve()) for value in args.candidate_files
            ],
            "base_hz": args.candidate_exclusion_base_hz,
            "width_sum_fraction": args.candidate_exclusion_widths,
        },
        "frozen_control_policy": frozen_control_policy,
        "pair_manifest": str(Path(args.pair_manifest).resolve()),
        "pair_manifest_sha256": sha256_file(args.pair_manifest),
        "labels_used": False,
        "scientific_boundary": (
            "Frequency-shifted pairs are empirical null controls for score "
            "diagnostics. They are not labeled astrophysical negatives and do "
            "not provide sensitivity or false-positive rates."
        ),
    }
    write_json(str(out_dir / "control_extraction_summary.json"), summary)
    if insufficient:
        raise RuntimeError(
            "%d primary pairs have fewer than %d valid controls; inspect the "
            "exclusion log and preregister a replacement shift set before inference"
            % (len(insufficient), args.minimum_controls_per_pair)
        )
    print(
        "Extracted %d/%d requested frequency-shift controls" % (len(records), requested)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract preregistered frequency-shift controls for real pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument(
        "--policy",
        default=None,
        help=(
            "frozen real-pair policy containing the preregistered control plan; "
            "required by the LOFTS0050 runner"
        ),
    )
    parser.add_argument(
        "--shift-station",
        default=None,
        help=(
            "fixed station override; by default one-station pairs shift the "
            "non-reporting counterpart and two-station pairs shift each station"
        ),
    )
    parser.add_argument("--shifts-hz", default=None)
    parser.add_argument("--edge-guard-widths", type=float, default=None)
    parser.add_argument("--candidate-exclusion-widths", type=float, default=None)
    parser.add_argument("--candidate-exclusion-base-hz", type=float, default=None)
    parser.add_argument("--minimum-controls-per-pair", type=int, default=None)
    parser.add_argument("--qa-count", type=int, default=12)
    parser.add_argument("--allow-masked-data", action="store_true")
    parser.add_argument("--require-all-controls", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
