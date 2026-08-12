#!/usr/bin/env python3
"""Validate and adapt Naoise's raw blind-hit-finder export.

The raw (uncollapsed) catalog is the operational input because the dual-site
union must retain candidates that a single-station broadband-RFI heuristic
might otherwise collapse.  The heuristic, native winning template, and every
per-template response are preserved as diagnostics; none is treated as truth.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lofts_bliss_schema import (
    CandidateRecord,
    atomic_write_text,
    load_observations,
    parse_bool,
    read_csv_dicts,
    sha256_file,
    write_json,
    write_jsonl,
)
from real_pair_geometry import observation_frequency_bounds_hz, search_coverage_audit

RAW_REQUIRED = {
    "OBS",
    "CANDIDATE_ID",
    "STATION",
    "FREQ_MHZ",
    "DR_HZ_S",
    "WIDTH",
    "WIDTH_HZ",
    "CHAN_BW_HZ",
    "BANK_SNR",
    "STANDARD_SNR",
    "DETECTED",
    "FLAG",
    "TEMPLATES_SKIPPED",
}
PER_TEMPLATE_REQUIRED = {
    "OBS",
    "CANDIDATE_ID",
    "STATION",
    "FREQ_MHZ",
    "DR_HZ_S",
    "WINNING_WIDTH",
    "TEMPLATE_WIDTH",
    "TEMPLATE_WIDTH_HZ",
    "TEMPLATE_BANK_SNR",
    "STANDARD_SNR",
    "FLAG",
}


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric; received %r" % (name, value)) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be finite; received %r" % (name, value))
    return result


def _optional_finite(value: Any, name: str) -> Optional[float]:
    if value is None or not str(value).strip():
        return None
    return _finite(value, name)


def _int(value: Any, name: str) -> int:
    numeric = _finite(value, name)
    result = int(round(numeric))
    if abs(numeric - result) > 1e-9:
        raise ValueError("%s must be an integer; received %r" % (name, value))
    return result


def _assert_columns(
    rows: Sequence[Mapping[str, Any]], required: set[str], path: str
) -> None:
    if not rows:
        raise ValueError("table is empty: %s" % path)
    missing = required - set(rows[0])
    if missing:
        raise ValueError("%s is missing columns %s" % (path, sorted(missing)))


def _close(left: float, right: float, absolute: float, relative: float = 0.0) -> bool:
    return abs(float(left) - float(right)) <= max(
        float(absolute), float(relative) * max(abs(float(left)), abs(float(right)))
    )


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _per_template_index(
    rows: Sequence[Mapping[str, str]],
    expected_obs: str,
    station_id: str,
    bank_widths: Tuple[int, ...],
) -> Dict[str, Dict[int, Mapping[str, str]]]:
    result: Dict[str, Dict[int, Mapping[str, str]]] = defaultdict(dict)
    for row_number, row in enumerate(rows, 2):
        if str(row["OBS"]).strip() != expected_obs:
            raise ValueError("per-template row %d has unexpected OBS" % row_number)
        if str(row["STATION"]).strip() != station_id:
            raise ValueError("per-template row %d has unexpected STATION" % row_number)
        source_id = str(row["CANDIDATE_ID"]).strip()
        width = _int(row["TEMPLATE_WIDTH"], "TEMPLATE_WIDTH")
        if width not in bank_widths:
            raise ValueError(
                "candidate %s has unregistered template w%d" % (source_id, width)
            )
        if width in result[source_id]:
            raise ValueError("candidate %s repeats template w%d" % (source_id, width))
        result[source_id][width] = row
    for source_id, values in result.items():
        if set(values) != set(bank_widths):
            raise ValueError(
                "candidate %s has templates %s; expected %s"
                % (source_id, sorted(values), list(bank_widths))
            )
    return dict(result)


def adapt(
    args: argparse.Namespace,
) -> Tuple[List[CandidateRecord], Dict[str, Any], List[Dict[str, Any]]]:
    observations = load_observations(args.observations, require_files=False)
    key = (args.simultaneous_group_id, args.station_id)
    if key not in observations:
        raise ValueError("observation manifest has no station/group key %r" % (key,))
    observation = observations[key]
    if observation.observation_id != args.observation_id:
        raise ValueError("--observation-id does not match the observation manifest")
    if observation.search_git_commit and (
        observation.search_git_commit != args.search_git_commit
    ):
        raise ValueError("--search-git-commit disagrees with the observation manifest")
    if observation.search_floor is not None and not _close(
        float(observation.search_floor), float(args.floor), 1e-12
    ):
        raise ValueError("--floor disagrees with the observation manifest")

    raw_rows = read_csv_dicts(args.raw_csv)
    per_rows = read_csv_dicts(args.per_template_csv)
    _assert_columns(raw_rows, RAW_REQUIRED, args.raw_csv)
    _assert_columns(per_rows, PER_TEMPLATE_REQUIRED, args.per_template_csv)
    parsed_bank_widths = [
        int(value) for value in args.bank_widths.split(",") if value.strip()
    ]
    if len(parsed_bank_widths) != len(set(parsed_bank_widths)):
        raise ValueError("--bank-widths contains duplicate templates")
    bank_widths = tuple(sorted(parsed_bank_widths))
    if not bank_widths:
        raise ValueError("--bank-widths must not be empty")
    if observation.search_bank_width_channels:
        if tuple(observation.search_bank_width_channels) != bank_widths:
            raise ValueError(
                "catalog bank %s disagrees with observation search bank %s"
                % (bank_widths, observation.search_bank_width_channels)
            )
    per_index = _per_template_index(
        per_rows, args.expected_obs_label, args.station_id, bank_widths
    )

    records: List[CandidateRecord] = []
    metrics: List[Dict[str, Any]] = []
    seen_source_ids = set()
    frequency_low, frequency_high = observation_frequency_bounds_hz(observation)
    for row_number, row in enumerate(raw_rows, 2):
        source_id = str(row["CANDIDATE_ID"]).strip()
        if not source_id or source_id in seen_source_ids:
            raise ValueError("raw candidate IDs must be non-empty and unique")
        seen_source_ids.add(source_id)
        if source_id not in per_index:
            raise ValueError(
                "candidate %s is absent from per-template export" % source_id
            )
        if str(row["OBS"]).strip() != args.expected_obs_label:
            raise ValueError("raw candidate %s has unexpected OBS" % source_id)
        if str(row["STATION"]).strip() != args.station_id:
            raise ValueError("raw candidate %s has unexpected STATION" % source_id)
        if not parse_bool(row["DETECTED"], "DETECTED"):
            raise ValueError("raw hit catalog must contain detected candidates only")

        frequency_hz = _finite(row["FREQ_MHZ"], "FREQ_MHZ") * 1e6
        drift_hz_s = _finite(row["DR_HZ_S"], "DR_HZ_S")
        width_channels = _int(row["WIDTH"], "WIDTH")
        width_hz = _finite(row["WIDTH_HZ"], "WIDTH_HZ")
        channel_bw_hz = _finite(row["CHAN_BW_HZ"], "CHAN_BW_HZ")
        bank_snr = _finite(row["BANK_SNR"], "BANK_SNR")
        standard_snr = _optional_finite(row["STANDARD_SNR"], "STANDARD_SNR")
        if width_channels not in bank_widths:
            raise ValueError("candidate %s has unregistered winning width" % source_id)
        if not _close(
            channel_bw_hz,
            abs(observation.signed_foff_hz),
            args.channel_bw_tolerance_hz,
        ):
            raise ValueError(
                "candidate %s channel bandwidth disagrees with header" % source_id
            )
        if not _close(
            width_hz,
            width_channels * channel_bw_hz,
            args.width_hz_tolerance,
            relative=5e-5,
        ):
            raise ValueError("candidate %s WIDTH_HZ is inconsistent" % source_id)
        if not frequency_low <= frequency_hz <= frequency_high:
            raise ValueError("candidate %s lies outside its filterbank" % source_id)
        if bank_snr + args.snr_rounding_tolerance < args.floor:
            raise ValueError(
                "candidate %s falls below the declared bank floor" % source_id
            )

        template_rows = per_index[source_id]
        per_template: Dict[str, float] = {}
        for template_width, template_row in sorted(template_rows.items()):
            if _int(template_row["WINNING_WIDTH"], "WINNING_WIDTH") != width_channels:
                raise ValueError(
                    "candidate %s winning-width fields disagree" % source_id
                )
            if not _close(
                _finite(template_row["FREQ_MHZ"], "FREQ_MHZ") * 1e6,
                frequency_hz,
                args.frequency_rounding_tolerance_hz,
            ):
                raise ValueError("candidate %s frequency fields disagree" % source_id)
            if not _close(
                _finite(template_row["DR_HZ_S"], "DR_HZ_S"),
                drift_hz_s,
                args.drift_rounding_tolerance_hz_s,
            ):
                raise ValueError("candidate %s drift fields disagree" % source_id)
            template_hz = _finite(
                template_row["TEMPLATE_WIDTH_HZ"], "TEMPLATE_WIDTH_HZ"
            )
            if not _close(
                template_hz,
                template_width * channel_bw_hz,
                args.width_hz_tolerance,
                relative=5e-5,
            ):
                raise ValueError(
                    "candidate %s template-width conversion disagrees" % source_id
                )
            per_template[str(template_width)] = _finite(
                template_row["TEMPLATE_BANK_SNR"], "TEMPLATE_BANK_SNR"
            )
        selected_response = per_template[str(width_channels)]
        if not _close(selected_response, bank_snr, args.snr_rounding_tolerance):
            raise ValueError(
                "candidate %s BANK_SNR disagrees with selected template" % source_id
            )
        best_response = max(per_template.values())
        winning_within_tolerance = bool(
            selected_response + args.snr_rounding_tolerance
            >= best_response * (1.0 - args.width_tolerance)
        )
        if not winning_within_tolerance:
            raise ValueError(
                "candidate %s selected width is not within the declared width tolerance"
                % source_id
            )

        eligible = [
            width
            for width in bank_widths
            if args.stage4_width_min_hz
            <= width * channel_bw_hz
            <= args.stage4_width_max_hz
        ]
        restricted_width = None
        restricted_snr = None
        if eligible:
            eligible_best = max(per_template[str(width)] for width in eligible)
            threshold = eligible_best * (1.0 - args.width_tolerance)
            restricted_width = min(
                width
                for width in eligible
                if per_template[str(width)] + args.snr_rounding_tolerance >= threshold
            )
            restricted_snr = per_template[str(restricted_width)]

        ratio = (
            None
            if standard_snr is None or standard_snr <= 0
            else bank_snr / standard_snr
        )
        broadband_rfi_like = bool(
            width_channels >= args.wide_threshold_channels
            and (ratio is None or ratio >= args.rfi_ratio)
        )
        source_flag = str(row.get("FLAG", "") or "").strip()
        templates_skipped = str(row.get("TEMPLATES_SKIPPED", "") or "").strip()
        # The search chooses the narrowest template within ``width_tolerance``
        # of the best *before* values are rounded to three decimals in the CSV.
        # Reject an earlier width only when it is unambiguously eligible under
        # the declared rounding bound; exact threshold ties remain admissible.
        unambiguously_earlier = [
            width
            for width in bank_widths
            if width < width_channels
            and per_template[str(width)] - args.snr_rounding_tolerance
            >= (best_response + args.snr_rounding_tolerance)
            * (1.0 - args.width_tolerance)
        ]
        if unambiguously_earlier:
            raise ValueError(
                "candidate %s selected w%d despite unambiguously eligible "
                "earlier template(s) %s under Naoise's narrowest-within-"
                "tolerance rule" % (source_id, width_channels, unambiguously_earlier)
            )
        for template_width, template_row in sorted(template_rows.items()):
            template_standard = _optional_finite(
                template_row.get("STANDARD_SNR"), "STANDARD_SNR"
            )
            if (standard_snr is None) != (template_standard is None):
                raise ValueError(
                    "candidate %s standard-SNR presence differs in template w%d"
                    % (source_id, template_width)
                )
            if standard_snr is not None and not _close(
                standard_snr,
                float(template_standard),
                args.snr_rounding_tolerance,
            ):
                raise ValueError(
                    "candidate %s standard SNR differs in template w%d"
                    % (source_id, template_width)
                )
            if str(template_row.get("FLAG", "") or "").strip() != source_flag:
                raise ValueError(
                    "candidate %s flag differs in template w%d"
                    % (source_id, template_width)
                )
        candidate_id = "%s:%s" % (args.station_id, source_id)
        extras = {
            "native_width_channels": width_channels,
            "channel_bandwidth_hz": channel_bw_hz,
            "bank_snr": bank_snr,
            "standard_snr": standard_snr,
            "bank_standard_ratio": ratio,
            "source_flag": source_flag,
            "templates_skipped": templates_skipped,
            "broadband_rfi_like": broadband_rfi_like,
            "naoise_floor": float(args.floor),
            "bank_width_channels": list(bank_widths),
            "per_template_snr": per_template,
            "stage4_restricted_width_channels": restricted_width,
            "stage4_restricted_width_hz": (
                None if restricted_width is None else restricted_width * channel_bw_hz
            ),
            "stage4_restricted_snr": restricted_snr,
            "stage4_restricted_above_floor": bool(
                restricted_snr is not None
                and restricted_snr + args.snr_rounding_tolerance >= args.floor
            ),
            "winning_within_tolerance": winning_within_tolerance,
            "source_candidate_id": source_id,
            "search_git_commit": args.search_git_commit,
            "catalog_role": "raw_uncollapsed_precoincidence",
        }
        record = CandidateRecord(
            candidate_id=candidate_id,
            observation_id=observation.observation_id,
            simultaneous_group_id=observation.simultaneous_group_id,
            station_id=observation.station_id,
            frequency_hz=frequency_hz,
            drift_hz_s=drift_hz_s,
            width_hz=width_hz,
            width_definition="Naoise selected template nominal Lorentzian FWHM",
            snr=bank_snr,
            snr_definition=(
                "unit-L2 matched-template response divided by per-drift-row noise "
                "sigma; max candidate accepted at floor %.6g" % args.floor
            ),
            frequency_ref_mjd=observation.start_mjd,
            frequency_ref_offset_s=None,
            detected=True,
            metadata_source="Naoise blind_hit_finder %s" % args.search_git_commit,
            source_table=str(Path(args.raw_csv).resolve()),
            source_row=row_number,
            extras=extras,
            truth={},
        )
        record.validate()
        own_coverage = search_coverage_audit(
            record, observation, guard_fwhm_fraction=0.0
        )
        # Preserve every emitted raw hit.  Refinement can move the reported
        # centroid/track partly into a roll-off region even though the search
        # seed that generated it was valid.  That is an audit flag, not grounds
        # to delete a union member.  Coverage is used as a hard semantic gate
        # only for a *missing counterpart* at the other station.
        record.extras["own_search_coverage"] = own_coverage
        records.append(record)
        metrics.append(
            {
                "candidate_id": candidate_id,
                "source_candidate_id": source_id,
                "frequency_mhz": frequency_hz / 1e6,
                "drift_hz_s": drift_hz_s,
                "native_width_channels": width_channels,
                "native_width_hz": width_hz,
                "bank_snr": bank_snr,
                "standard_snr": "" if standard_snr is None else standard_snr,
                "bank_standard_ratio": "" if ratio is None else ratio,
                "source_flag": source_flag,
                "broadband_rfi_like": broadband_rfi_like,
                "restricted_width_channels": (
                    "" if restricted_width is None else restricted_width
                ),
                "restricted_width_hz": (
                    "" if restricted_width is None else restricted_width * channel_bw_hz
                ),
                "restricted_snr": "" if restricted_snr is None else restricted_snr,
                "restricted_above_floor": bool(
                    restricted_snr is not None
                    and restricted_snr + args.snr_rounding_tolerance >= args.floor
                ),
                "own_track_fully_in_clean_search_band": bool(
                    own_coverage["searched_clean_band_covered"]
                ),
                "own_search_coverage_reason": own_coverage["reason"],
            }
        )

    if set(per_index) != seen_source_ids:
        extras = sorted(set(per_index) - seen_source_ids)
        raise ValueError(
            "per-template export contains candidates absent from raw CSV: %s"
            % extras[:10]
        )
    records.sort(key=lambda item: (item.frequency_hz, item.candidate_id))
    metrics.sort(key=lambda item: (item["frequency_mhz"], item["candidate_id"]))
    width_counts = Counter(
        int(item.extras["native_width_channels"]) for item in records
    )
    flag_counts = Counter(
        str(item.extras["source_flag"] or "unflagged") for item in records
    )
    summary = {
        "format_version": 1,
        "catalog_role": "raw_uncollapsed_precoincidence",
        "station_id": args.station_id,
        "observation_id": args.observation_id,
        "simultaneous_group_id": args.simultaneous_group_id,
        "expected_obs_label": args.expected_obs_label,
        "n_candidates": len(records),
        "n_per_template_rows": len(per_rows),
        "per_template_rows_per_candidate": len(bank_widths),
        "bank_width_channels": list(bank_widths),
        "bank_floor": float(args.floor),
        "width_tolerance": float(args.width_tolerance),
        "native_width_counts": {
            str(key): value for key, value in sorted(width_counts.items())
        },
        "source_flag_counts": dict(sorted(flag_counts.items())),
        "n_broadband_rfi_like": sum(
            bool(item.extras["broadband_rfi_like"]) for item in records
        ),
        "n_stage4_restricted_above_floor": sum(
            bool(item.extras["stage4_restricted_above_floor"]) for item in records
        ),
        "n_own_tracks_not_fully_in_clean_search_band": sum(
            not bool(item.extras["own_search_coverage"]["searched_clean_band_covered"])
            for item in records
        ),
        "raw_csv": str(Path(args.raw_csv).resolve()),
        "raw_csv_sha256": sha256_file(args.raw_csv),
        "per_template_csv": str(Path(args.per_template_csv).resolve()),
        "per_template_csv_sha256": sha256_file(args.per_template_csv),
        "observation_manifest": str(Path(args.observations).resolve()),
        "observation_manifest_sha256": sha256_file(args.observations),
        "search_git_commit": args.search_git_commit,
        "truth_fields_present": False,
        "collapsed_catalog_used_for_union": False,
        "scientific_boundary": (
            "The single-station broadband-RFI-like flag is a retained diagnostic, "
            "not an exclusion rule. Candidate widths are detector template outputs, "
            "not ground-truth signal widths. An emitted hit remains in the raw "
            "union even when its refined track is not fully inside the clean "
            "roll-off search region; coverage gates only the interpretation of "
            "a missing counterpart at the other station."
        ),
    }
    return records, summary, metrics


def main(args: argparse.Namespace) -> None:
    records, summary, metrics = adapt(args)
    write_jsonl(args.output, [item.to_dict(include_truth=False) for item in records])
    write_json(str(Path(args.output).with_suffix(".audit.json")), summary)
    _write_csv(str(Path(args.output).with_suffix(".metrics.csv")), metrics)
    print(
        "Adapted %d raw %s candidates with %d per-template responses each"
        % (len(records), args.station_id, summary["per_template_rows_per_candidate"])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate/adapt a Naoise raw blind-hit-finder catalog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--per-template-csv", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--simultaneous-group-id", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--expected-obs-label", required=True)
    parser.add_argument("--search-git-commit", required=True)
    parser.add_argument("--bank-widths", default="1,3,8,20,50,120")
    parser.add_argument("--floor", type=float, default=20.0)
    parser.add_argument("--width-tolerance", type=float, default=0.10)
    parser.add_argument("--wide-threshold-channels", type=int, default=120)
    parser.add_argument("--rfi-ratio", type=float, default=8.0)
    parser.add_argument("--stage4-width-min-hz", type=float, default=10.0)
    parser.add_argument("--stage4-width-max-hz", type=float, default=100.0)
    parser.add_argument("--channel-bw-tolerance-hz", type=float, default=0.01)
    parser.add_argument("--width-hz-tolerance", type=float, default=0.06)
    parser.add_argument("--frequency-rounding-tolerance-hz", type=float, default=0.6)
    parser.add_argument("--drift-rounding-tolerance-hz-s", type=float, default=5e-6)
    parser.add_argument("--snr-rounding-tolerance", type=float, default=0.01)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
