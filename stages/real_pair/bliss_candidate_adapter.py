#!/usr/bin/env python3
"""Convert project BLISS tables into the canonical LOFTS candidate schema.

The public BLISS interfaces and project-specific broadened-signal branches do
not share a guaranteed column vocabulary.  This adapter therefore uses an
explicit JSON mapping instead of guessing column names or units.  It never
computes detections and never substitutes injected truth for recovered
metadata.

Example mapping for a candidate CSV::

    {
      "columns": {
        "candidate_id": "hit_id",
        "frequency": "frequency_mhz",
        "drift": "drift_rate_hz_s",
        "width": "fwhm_hz",
        "snr": "snr"
      },
      "constants": {
        "observation_id": "LOFTS0192_IE",
        "simultaneous_group_id": "LOFTS0192",
        "station_id": "IE",
        "width_definition": "FWHM",
        "snr_definition": "BLISS reported S/N",
        "metadata_source": "project broadened-signal BLISS",
        "resampling_block_id": "injection_block_0001"
      },
      "units": {"frequency": "MHz", "drift": "Hz/s", "width": "Hz"},
      "frequency_reference": {"mode": "observation_start"},
      "width_to_fwhm_scale": 1.0
    }

Use ``--kind truth`` with a separate mapping/file for injection truth.  That
separation is deliberate: candidate union and inference must not be able to
read injected parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lofts_bliss_schema import (
    CandidateRecord,
    InjectionTruthRecord,
    ObservationRecord,
    finite_float,
    load_observations,
    nonempty_text,
    parse_bool,
    read_csv_dicts,
    read_json_records,
    sha256_file,
    stable_id,
    unit_scale,
    write_json,
    write_jsonl,
)

ALLOWED_REFERENCE_MODES = {
    "column_mjd",
    "column_offset_s",
    "observation_start",
    "observation_midpoint",
}


def _load_rows(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        if suffix == ".tsv":
            import csv

            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if not reader.fieldnames:
                    raise ValueError("table has no header: %s" % path)
                return [dict(row) for row in reader]
        return read_csv_dicts(path)
    return read_json_records(path)


def _load_mapping(path: str) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("mapping JSON root must be an object")
    for key in ("columns", "constants", "units", "frequency_reference"):
        if key in value and not isinstance(value[key], dict):
            raise ValueError("mapping.%s must be an object" % key)

    def visit(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, "%s.%s" % (location, key))
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, "%s[%d]" % (location, index))
        elif isinstance(item, str) and item.strip().upper().startswith("EDIT_"):
            raise ValueError("unresolved configuration placeholder at %s" % location)

    visit(value, "mapping")
    return value


def _value(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    canonical_name: str,
    required: bool = True,
    default: Any = None,
) -> Any:
    columns = mapping.get("columns", {})
    constants = mapping.get("constants", {})
    if canonical_name in columns:
        source_name = str(columns[canonical_name])
        if source_name not in row:
            raise ValueError(
                "mapped source column %r for %s is absent"
                % (source_name, canonical_name)
            )
        value = row[source_name]
    elif canonical_name in constants:
        value = constants[canonical_name]
    else:
        value = default
    if required and (value is None or str(value).strip() == ""):
        raise ValueError("mapping must provide %s" % canonical_name)
    return value


def _observation_for_row(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    observations: Mapping[Tuple[str, str], ObservationRecord],
) -> ObservationRecord:
    group_id = nonempty_text(
        _value(row, mapping, "simultaneous_group_id"), "simultaneous_group_id"
    )
    station_id = nonempty_text(_value(row, mapping, "station_id"), "station_id")
    key = (group_id, station_id)
    if key not in observations:
        raise ValueError("candidate references unknown observation key %r" % (key,))
    observation = observations[key]
    supplied_observation_id = _value(
        row,
        mapping,
        "observation_id",
        required=False,
        default=observation.observation_id,
    )
    if str(supplied_observation_id).strip() != observation.observation_id:
        raise ValueError(
            "observation_id %r does not match manifest %r for %r"
            % (supplied_observation_id, observation.observation_id, key)
        )
    return observation


def _scaled(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    canonical_name: str,
    quantity: str,
) -> float:
    raw = _value(row, mapping, canonical_name)
    units = mapping.get("units", {})
    if canonical_name not in units:
        raise ValueError("mapping.units must explicitly specify %s" % canonical_name)
    return finite_float(raw, canonical_name) * unit_scale(
        units[canonical_name], quantity
    )


def _frequency_reference(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    observation: ObservationRecord,
) -> Tuple[Optional[float], Optional[float]]:
    config = mapping.get("frequency_reference", {})
    mode = str(config.get("mode", "")).strip().lower()
    if mode not in ALLOWED_REFERENCE_MODES:
        raise ValueError(
            "frequency_reference.mode must be one of %s; received %r"
            % (sorted(ALLOWED_REFERENCE_MODES), mode)
        )
    if mode == "column_mjd":
        column = nonempty_text(config.get("column"), "frequency_reference.column")
        if column not in row:
            raise ValueError("frequency-reference column %r is absent" % column)
        if observation.time_alignment != "absolute_mjd":
            raise ValueError("column_mjd is invalid for a normalized proxy observation")
        return finite_float(row[column], column), None
    if mode == "column_offset_s":
        column = nonempty_text(config.get("column"), "frequency_reference.column")
        if column not in row:
            raise ValueError("frequency-reference column %r is absent" % column)
        scale = unit_scale(str(config.get("unit", "s")), "time")
        if observation.time_alignment != "normalized_proxy":
            raise ValueError(
                "column_offset_s is reserved for normalized proxy observations"
            )
        return None, finite_float(row[column], column) * scale
    if mode == "observation_start":
        if observation.time_alignment == "absolute_mjd":
            return float(observation.start_mjd), None
        return None, 0.0
    if observation.time_alignment == "absolute_mjd":
        return float(observation.midpoint_mjd), None
    return None, 0.5 * (float(observation.n_time) - 1.0) * observation.tsamp_s


def _fwhm_hz(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> float:
    width = _scaled(row, mapping, "width", "width")
    scale = finite_float(mapping.get("width_to_fwhm_scale", 1.0), "width_to_fwhm_scale")
    if scale <= 0:
        raise ValueError("width_to_fwhm_scale must be positive")
    width *= scale
    definition = (
        str(_value(row, mapping, "width_definition", required=False, default=""))
        .strip()
        .lower()
    )
    if not definition:
        raise ValueError(
            "width_definition must be explicit (for example FWHM); do not infer it"
        )
    canonical = definition.replace("_", " ").replace("-", " ").strip()
    if canonical not in {"fwhm", "full width at half maximum"}:
        if "width_to_fwhm_scale" not in mapping:
            raise ValueError(
                "non-FWHM width definition %r requires an explicit width_to_fwhm_scale"
                % definition
            )
    if not (width > 0):
        raise ValueError("converted FWHM must be positive")
    return width


def _candidate_record(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    observations: Mapping[Tuple[str, str], ObservationRecord],
    source_table: str,
    source_row: int,
) -> CandidateRecord:
    observation = _observation_for_row(row, mapping, observations)
    frequency_hz = _scaled(row, mapping, "frequency", "frequency")
    drift_hz_s = _scaled(row, mapping, "drift", "drift")
    width_hz = _fwhm_hz(row, mapping)
    snr = finite_float(_value(row, mapping, "snr"), "snr")
    frequency_ref_mjd, frequency_ref_offset_s = _frequency_reference(
        row, mapping, observation
    )
    supplied_id = _value(row, mapping, "candidate_id", required=False)
    id_prefix = str(mapping.get("id_prefix", ""))
    candidate_id = (
        id_prefix + str(supplied_id).strip()
        if supplied_id is not None and str(supplied_id).strip()
        else stable_id(
            "cand",
            observation.observation_id,
            source_table,
            source_row,
            frequency_hz,
            drift_hz_s,
            width_hz,
        )
    )
    detected_raw = _value(row, mapping, "detected", required=False, default=True)
    # Candidate records are allow-listed rather than copying arbitrary source
    # columns. This prevents injected truth/labels in an upstream audit table from
    # leaking into union construction or inference artifacts.
    extras: Dict[str, Any] = {}
    block_id = _value(row, mapping, "resampling_block_id", required=False, default=None)
    if block_id is not None and str(block_id).strip():
        extras["resampling_block_id"] = str(block_id).strip()
    item = CandidateRecord(
        candidate_id=candidate_id,
        observation_id=observation.observation_id,
        simultaneous_group_id=observation.simultaneous_group_id,
        station_id=observation.station_id,
        frequency_hz=frequency_hz,
        drift_hz_s=drift_hz_s,
        width_hz=width_hz,
        width_definition="FWHM",
        snr=snr,
        snr_definition=nonempty_text(
            _value(row, mapping, "snr_definition"), "snr_definition"
        ),
        frequency_ref_mjd=frequency_ref_mjd,
        frequency_ref_offset_s=frequency_ref_offset_s,
        detected=parse_bool(detected_raw, "detected"),
        metadata_source=str(
            _value(row, mapping, "metadata_source", required=False, default="BLISS")
        ),
        source_table=str(Path(source_table).resolve()),
        source_row=source_row,
        extras=extras,
        truth={},
    )
    item.validate()
    if not item.detected:
        raise ValueError(
            "candidate adapter accepts detected hits only; non-detections are represented "
            "by absence from a station list"
        )
    return item


def _truth_record(
    row: Mapping[str, Any],
    mapping: Mapping[str, Any],
    observations: Mapping[Tuple[str, str], ObservationRecord],
    source_row: int,
) -> InjectionTruthRecord:
    observation = _observation_for_row(row, mapping, observations)
    frequency_ref_mjd, frequency_ref_offset_s = _frequency_reference(
        row, mapping, observation
    )
    supplied_id = _value(row, mapping, "injection_id", required=False)
    id_prefix = str(mapping.get("id_prefix", ""))
    injection_id = (
        id_prefix + str(supplied_id).strip()
        if supplied_id is not None and str(supplied_id).strip()
        else stable_id("inj", observation.observation_id, source_row)
    )
    label_raw = _value(row, mapping, "pair_label", required=False)
    pair_label = None if label_raw in (None, "") else int(label_raw)
    consumed_source_columns = {
        str(value) for value in mapping.get("columns", {}).values()
    }
    extras = {
        str(key): value
        for key, value in row.items()
        if str(key) not in consumed_source_columns
    }
    for field_name in (
        "population",
        "evaluation_cell_id",
        "resampling_block_id",
    ):
        value = _value(row, mapping, field_name, required=False, default=None)
        if value is not None and str(value).strip():
            extras[field_name] = str(value).strip()
    item = InjectionTruthRecord(
        injection_id=injection_id,
        event_id=str(
            _value(row, mapping, "event_id", required=False, default=injection_id)
        ).strip(),
        observation_id=observation.observation_id,
        simultaneous_group_id=observation.simultaneous_group_id,
        station_id=observation.station_id,
        frequency_hz=_scaled(row, mapping, "frequency", "frequency"),
        drift_hz_s=_scaled(row, mapping, "drift", "drift"),
        width_hz=_fwhm_hz(row, mapping),
        snr=finite_float(_value(row, mapping, "snr"), "snr"),
        frequency_ref_mjd=frequency_ref_mjd,
        frequency_ref_offset_s=frequency_ref_offset_s,
        shape=str(_value(row, mapping, "shape", required=False, default="unknown")),
        pair_label=pair_label,
        case=str(_value(row, mapping, "case", required=False, default="unknown")),
        source_row=source_row,
        extras=extras,
    )
    item.validate()
    return item


def main(args: argparse.Namespace) -> None:
    mapping = _load_mapping(args.mapping)
    observations = load_observations(args.observations, require_files=False)
    rows = _load_rows(args.input)
    if not rows:
        raise ValueError("input table is empty")
    records: Sequence[Any]
    if args.kind == "candidate":
        records = [
            _candidate_record(row, mapping, observations, args.input, index)
            for index, row in enumerate(rows, 1)
        ]
        ids = [item.candidate_id for item in records]
        recovery_links = []
        for row, record in zip(rows, records):
            link = _value(
                row, mapping, "recovery_link_id", required=False, default=None
            )
            if link is not None and str(link).strip():
                recovery_links.append(
                    {
                        "candidate_id": record.candidate_id,
                        "recovery_link_id": str(mapping.get("id_prefix", ""))
                        + str(link).strip(),
                    }
                )
    else:
        records = [
            _truth_record(row, mapping, observations, index)
            for index, row in enumerate(rows, 1)
        ]
        ids = [item.injection_id for item in records]
        recovery_links = []
    if len(ids) != len(set(ids)):
        raise ValueError("adapted record IDs must be unique")
    write_jsonl(args.output, [item.to_dict() for item in records])
    recovery_links_output = args.recovery_links_output
    if args.kind == "candidate":
        if recovery_links_output is None:
            output = Path(args.output)
            recovery_links_output = str(
                output.with_name(output.stem + ".recovery_links.jsonl")
            )
        write_jsonl(recovery_links_output, recovery_links)
    audit = {
        "format_version": 1,
        "kind": args.kind,
        "source_table": str(Path(args.input).resolve()),
        "source_table_sha256": sha256_file(args.input),
        "mapping": str(Path(args.mapping).resolve()),
        "mapping_sha256": sha256_file(args.mapping),
        "observation_manifest": str(Path(args.observations).resolve()),
        "observation_manifest_sha256": sha256_file(args.observations),
        "n_input_rows": len(rows),
        "n_output_records": len(records),
        "frequency_reference_mode": mapping.get("frequency_reference", {}).get("mode"),
        "canonical_units": {
            "frequency": "Hz",
            "drift": "Hz/s",
            "width": "FWHM Hz",
        },
        "truth_segregated": bool(args.kind == "truth"),
        "candidate_records_contain_recovery_links": False,
        "recovery_links_output": (
            str(Path(recovery_links_output).resolve())
            if recovery_links_output is not None
            else None
        ),
        "n_recovery_links": len(recovery_links),
        "candidate_source_columns_not_propagated": (
            sorted(
                set(str(key) for key in rows[0])
                - set(str(value) for value in mapping.get("columns", {}).values())
            )
            if args.kind == "candidate"
            else []
        ),
    }
    write_json(str(Path(args.output).with_suffix(".audit.json")), audit)
    print(
        "Wrote %d canonical %s records to %s" % (len(records), args.kind, args.output)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt an explicitly mapped BLISS table to LOFTS JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="source CSV, TSV, JSON or JSONL")
    parser.add_argument(
        "--mapping", required=True, help="explicit JSON column/unit mapping"
    )
    parser.add_argument(
        "--observations", required=True, help="canonical observation JSONL"
    )
    parser.add_argument("--kind", choices=("candidate", "truth"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--recovery-links-output",
        default=None,
        help=(
            "candidate-ID to injection-ID sidecar; written separately so union and "
            "inference inputs remain truth-free"
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
