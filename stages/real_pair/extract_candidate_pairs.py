#!/usr/bin/env python3
"""Extract fixed-shape station pairs for every retained candidate-union entry.

For a two-station entry, each station uses its own recovered BLISS metadata.
For a one-station entry, the reporting station's barycentric frequency, drift
and width anchor the corresponding extraction at the non-reporting station.
No edge padding, frequency wrapping, truth lookup or intersection-only filter
is permitted.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from lofts_bliss_schema import (
    SECONDS_PER_DAY,
    CandidateRecord,
    ObservationRecord,
    load_observations,
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from lofts_filterbank import read_filterbank_window, validate_window_bounds


def _round_nearest(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _time_window(
    observation: ObservationRecord,
    reference_kind: str,
    reference_value: float,
    n_rows: int,
) -> Tuple[int, float, float]:
    reference_row = 0.5 * (n_rows - 1.0)
    if reference_kind == "mjd":
        desired_row = (
            (float(reference_value) - observation.start_mjd)
            * SECONDS_PER_DAY
            / observation.tsamp_s
        )
    elif reference_kind == "offset_s":
        desired_row = float(reference_value) / observation.tsamp_s
    else:
        raise ValueError("unknown union reference kind %r" % reference_kind)
    t_start = _round_nearest(desired_row - reference_row)
    validate_window_bounds(observation, t_start, 0, n_rows, 1)
    actual_mid_row = t_start + reference_row
    if reference_kind == "mjd":
        actual_reference = observation.start_mjd + (
            actual_mid_row * observation.tsamp_s / SECONDS_PER_DAY
        )
        error_s = (actual_reference - float(reference_value)) * SECONDS_PER_DAY
    else:
        actual_reference = actual_mid_row * observation.tsamp_s
        error_s = actual_reference - float(reference_value)
    return t_start, actual_reference, error_s


def _candidate_at_station_reference(
    candidate: CandidateRecord,
    reference_kind: str,
    actual_reference: float,
) -> Tuple[float, float, float]:
    if reference_kind == "mjd":
        frequency_hz = candidate.frequency_at_mjd(actual_reference)
    else:
        frequency_hz = candidate.frequency_at_offset_s(actual_reference)
    return frequency_hz, float(candidate.drift_hz_s), float(candidate.width_hz)


def _station_candidate(entry: Mapping[str, Any], station_id: str) -> CandidateRecord:
    station = entry["stations"][station_id]
    if station.get("metadata_anchor"):
        return CandidateRecord.from_dict(station["metadata_anchor"])
    if station.get("detected"):
        return CandidateRecord.from_dict(station["candidate"])
    anchor_station = str(entry["anchor_station_id"])
    anchor = entry["stations"][anchor_station]
    if not anchor.get("detected") or not anchor.get("candidate"):
        raise ValueError("one-station union entry has no valid anchor candidate")
    return CandidateRecord.from_dict(anchor["candidate"])


def _frequency_window(
    observation: ObservationRecord,
    frequency_hz: float,
    drift_hz_s: float,
    width_hz: float,
    n_rows: int,
    n_cols: int,
    reference_row: float,
    edge_guard_widths: float,
) -> Tuple[int, float, Dict[str, float]]:
    global_center = observation.channel_for_frequency_hz(frequency_hz)
    f_start = int(math.floor(global_center - 0.5 * (n_cols - 1)))
    local_center = global_center - f_start
    rows = np.arange(n_rows, dtype=float)
    track = local_center + (
        drift_hz_s
        * (rows - float(reference_row))
        * observation.tsamp_s
        / observation.signed_foff_hz
    )
    guard_channels = max(
        2.0,
        float(edge_guard_widths) * width_hz / abs(observation.signed_foff_hz),
    )
    minimum_clearance = min(float(np.min(track)), float(n_cols - 1 - np.max(track)))
    if minimum_clearance < guard_channels:
        raise ValueError(
            "candidate track has %.2f channels of edge clearance; %.2f required"
            % (minimum_clearance, guard_channels)
        )
    validate_window_bounds(observation, 0, f_start, 1, n_cols)
    return (
        f_start,
        local_center,
        {
            "track_first_channel": float(track[0]),
            "track_last_channel": float(track[-1]),
            "track_min_channel": float(np.min(track)),
            "track_max_channel": float(np.max(track)),
            "edge_guard_channels": float(guard_channels),
            "minimum_edge_clearance_channels": float(minimum_clearance),
        },
    )


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _plot_qa(path: Path, raw_a: np.ndarray, raw_b: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = np.concatenate((raw_a.ravel(), raw_b.ravel()))
    lo, hi = np.quantile(combined[np.isfinite(combined)], [0.01, 0.99])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    for axis, data, label in zip(axes, (raw_a, raw_b), ("Station A", "Station B")):
        axis.imshow(data, origin="lower", aspect="auto", vmin=lo, vmax=hi)
        axis.set_title(label)
        axis.set_xlabel("Local frequency channel")
    axes[0].set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def extract_entry(
    entry: Mapping[str, Any],
    observations: Mapping[Tuple[str, str], ObservationRecord],
    arrays_dir: Path,
    n_rows: int,
    n_cols: int,
    if_index: int,
    edge_guard_widths: float,
    overwrite: bool,
    allow_masked_data: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    group_id = str(entry["simultaneous_group_id"])
    station_ids = list(entry["station_ids"])
    if len(station_ids) != 2:
        raise ValueError("union entry must contain exactly two station IDs")
    reference_kind = str(entry["reference_kind"])
    reference_value = float(entry["reference_value"])
    station_records: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    mid_errors: List[float] = []
    for station_id in station_ids:
        observation = observations[(group_id, station_id)]
        t_start, actual_reference, time_error_s = _time_window(
            observation, reference_kind, reference_value, n_rows
        )
        candidate = _station_candidate(entry, station_id)
        frequency_hz, drift_hz_s, width_hz = _candidate_at_station_reference(
            candidate, reference_kind, actual_reference
        )
        configured_width = entry["stations"][station_id].get("preprocessing_width_hz")
        if configured_width not in (None, ""):
            width_hz = float(configured_width)
        reference_row = 0.5 * (n_rows - 1.0)
        f_start, local_center, containment = _frequency_window(
            observation,
            frequency_hz,
            drift_hz_s,
            width_hz,
            n_rows,
            n_cols,
            reference_row,
            edge_guard_widths,
        )
        raw, read_audit = read_filterbank_window(
            observation,
            t_start=t_start,
            f_start=f_start,
            n_rows=n_rows,
            n_cols=n_cols,
            if_index=if_index,
            fail_on_masked=not allow_masked_data,
            return_audit=True,
        )
        arrays[station_id] = raw
        mid_errors.append(time_error_s)
        station_records[station_id] = {
            "detected_by_bliss": bool(entry["stations"][station_id]["detected"]),
            "metadata_anchor_candidate_id": candidate.candidate_id,
            "metadata_anchor_station_id": candidate.station_id,
            "observation_id": observation.observation_id,
            "filterbank_path": observation.filterbank_path,
            "header_fingerprint": observation.header_fingerprint,
            "barycentric_status": observation.barycentric_status,
            "time_alignment": observation.time_alignment,
            "t_start": t_start,
            "f_start": f_start,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "if_index": if_index,
            "read_audit": read_audit,
            "actual_reference_kind": reference_kind,
            "actual_reference_value": actual_reference,
            "reference_time_rounding_error_s": time_error_s,
            "candidate_frequency_hz_at_local_reference": frequency_hz,
            "candidate_center_channel": local_center,
            "candidate_reference_row": reference_row,
            "reported_drift_hz_s": drift_hz_s,
            "reported_width_fwhm_hz": width_hz,
            "reported_native_width_fwhm_hz": float(candidate.width_hz),
            "preprocessing_width_source": entry["stations"][station_id].get(
                "preprocessing_width_source", "native_candidate_width"
            ),
            "reported_snr": float(candidate.snr),
            "candidate_operational_extras": dict(candidate.extras),
            "union_search_coverage": dict(
                entry["stations"][station_id].get("coverage", {})
            ),
            "signed_foff_hz": observation.signed_foff_hz,
            "tsamp_s": observation.tsamp_s,
            **containment,
        }
    if abs(mid_errors[0] - mid_errors[1]) > 0.5 * sum(
        observations[(group_id, station_id)].tsamp_s for station_id in station_ids
    ):
        raise ValueError(
            "station cutout midpoint mismatch exceeds cadence rounding bound"
        )

    pair_path = arrays_dir / (str(entry["union_id"]) + ".npz")
    if pair_path.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite %s" % pair_path)
    raw_a, raw_b = arrays[station_ids[0]], arrays[station_ids[1]]
    _atomic_savez(
        pair_path,
        raw_a=raw_a,
        raw_b=raw_b,
        station_ids=np.asarray(station_ids),
        union_id=np.asarray(str(entry["union_id"])),
    )
    record = {
        "format_version": 1,
        "pair_id": str(entry["union_id"]),
        "union_id": str(entry["union_id"]),
        "simultaneous_group_id": group_id,
        "detection_state": entry["detection_state"],
        "operational_eligibility": entry.get("operational_eligibility"),
        "anchor_station_id": entry["anchor_station_id"],
        "anchor_candidate_id": entry["anchor_candidate_id"],
        "resampling_block_id": entry.get("resampling_block_id"),
        "resampling_block_disagreement": entry.get("resampling_block_disagreement", []),
        "route": entry["route"],
        "association": entry.get("association"),
        "width_routing": entry.get("width_routing"),
        "contains_broadband_rfi_like": bool(
            entry.get("contains_broadband_rfi_like", False)
        ),
        "array_file": str(pair_path.resolve()),
        "array_sha256": sha256_file(str(pair_path)),
        "station_order": station_ids,
        "stations": station_records,
        "association_policy_id": entry["association_policy_id"],
        "truth": None,
    }
    return record, raw_a, raw_b


def main(args: argparse.Namespace) -> None:
    if args.n_rows <= 0 or args.n_cols <= 0:
        raise ValueError("cutout dimensions must be positive")
    if args.edge_guard_widths <= 0:
        raise ValueError("edge_guard_widths must be positive")
    observations = load_observations(args.observations, require_files=True)
    union_entries = read_json_records(args.union)
    include_routes = {
        value.strip() for value in args.include_routes.split(",") if value.strip()
    }
    selected = [item for item in union_entries if item.get("route") in include_routes]
    out_dir = Path(args.out_dir)
    arrays_dir = out_dir / "pairs"
    qa_dir = out_dir / "qa_raw_pairs"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    if args.qa_count:
        qa_dir.mkdir(parents=True, exist_ok=True)
    records, excluded = [], []
    for entry in selected:
        try:
            record, raw_a, raw_b = extract_entry(
                entry,
                observations,
                arrays_dir,
                args.n_rows,
                args.n_cols,
                args.if_index,
                args.edge_guard_widths,
                args.overwrite,
                args.allow_masked_data,
            )
            records.append(record)
            if len(records) <= args.qa_count:
                _plot_qa(
                    qa_dir / (record["pair_id"] + ".png"),
                    raw_a,
                    raw_b,
                    "%s | %s" % (record["pair_id"], record["detection_state"]),
                )
        except Exception as exc:
            excluded.append(
                {
                    "union_id": entry.get("union_id"),
                    "route": entry.get("route"),
                    "reason": "%s: %s" % (type(exc).__name__, exc),
                }
            )
            if args.fail_on_exclusion:
                raise
    write_jsonl(str(out_dir / "pair_manifest.jsonl"), records)
    write_jsonl(str(out_dir / "excluded_union_entries.jsonl"), excluded)
    summary = {
        "format_version": 1,
        "n_union_entries": len(union_entries),
        "n_route_selected": len(selected),
        "n_extracted": len(records),
        "n_excluded": len(excluded),
        "include_routes": sorted(include_routes),
        "shape": [args.n_rows, args.n_cols],
        "edge_guard_widths": args.edge_guard_widths,
        "union_path": str(Path(args.union).resolve()),
        "union_sha256": sha256_file(args.union),
        "no_padding_or_wrapping": True,
        "truth_used": False,
        "masked_data_allowed": bool(args.allow_masked_data),
    }
    write_json(str(out_dir / "extraction_summary.json"), summary)
    if len(records) < args.minimum_extracted:
        raise RuntimeError(
            "extracted %d pairs, below --minimum-extracted=%d"
            % (len(records), args.minimum_extracted)
        )
    print(
        "Extracted %d/%d selected union pairs; %d excluded (see %s)"
        % (len(records), len(selected), len(excluded), out_dir)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract fixed-shape paired waterfalls from a BLISS candidate union",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--union", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-routes", default="high_resolution_stage4")
    parser.add_argument("--n-rows", type=int, default=16)
    parser.add_argument("--n-cols", type=int, default=1024)
    parser.add_argument("--if-index", type=int, default=0)
    parser.add_argument("--edge-guard-widths", type=float, default=4.0)
    parser.add_argument("--qa-count", type=int, default=20)
    parser.add_argument("--minimum-extracted", type=int, default=1)
    parser.add_argument("--fail-on-exclusion", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-masked-data",
        action="store_true",
        help=(
            "preserve underlying values at masked HDF5 samples. The default is "
            "to fail because no imputation policy has been registered."
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
