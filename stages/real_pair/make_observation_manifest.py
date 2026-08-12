#!/usr/bin/env python3
"""Inspect filterbank headers and create the canonical observation manifest.

Input CSV columns (required):

``observation_id,simultaneous_group_id,station_id,filterbank_path,``
``barycentric_status,time_alignment``

Optional columns record the established barycentric tool/version and expected
header values.  The script never infers barycentric status from a filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from lofts_bliss_schema import (
    ObservationRecord,
    finite_float,
    read_csv_dicts,
    sha256_file,
    validate_group_observations,
    write_json,
    write_jsonl,
)
from real_pair_geometry import common_frequency_bounds_hz

REQUIRED_COLUMNS = {
    "observation_id",
    "simultaneous_group_id",
    "station_id",
    "filterbank_path",
    "barycentric_status",
    "time_alignment",
}


def _header_fingerprint(audit: Dict[str, Any]) -> str:
    """Hash scientific header content, independent of path/reader choice."""

    payload = {
        key: value
        for key, value in audit.items()
        if key not in {"path", "reader", "header_fingerprint"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _shape3(shape):
    values = tuple(int(value) for value in shape)
    if len(values) == 3:
        return values
    if len(values) == 2:
        return values[0], 1, values[1]
    raise ValueError("unsupported filterbank shape %r" % (shape,))


def normalize_hdf5_attribute(value):
    """Convert scalar or array HDF5 attributes into JSON-safe Python values.

    HDF5 filterbank headers include a multi-element ``DIMENSION_LABELS``
    attribute.  Calling ``.item()`` on that NumPy array raises ``ValueError``.
    Normalise arrays recursively while preserving ordinary scalar attributes.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [normalize_hdf5_attribute(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if converted is not value:
            return normalize_hdf5_attribute(converted)
    if hasattr(value, "item"):
        try:
            return normalize_hdf5_attribute(value.item())
        except ValueError:
            pass
    return value


def inspect_filterbank(path: str) -> Dict[str, Any]:
    """Read a BLISS HDF5/SIGPROC header without loading waterfall data."""

    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        import h5py

        with h5py.File(target, "r") as handle:
            if "data" not in handle:
                raise ValueError("HDF5 file has no /data dataset")
            dataset = handle["data"]
            header = {str(key): dataset.attrs[key] for key in dataset.attrs}
            n_time, n_ifs, n_channels = _shape3(dataset.shape)

        header = {key: normalize_hdf5_attribute(value) for key, value in header.items()}
        required = ("fch1", "foff", "tsamp", "tstart")
        missing = [key for key in required if key not in header]
        if missing:
            raise ValueError("filterbank header is missing %s" % missing)
        audit = {
            "path": str(target),
            "fch1_mhz": float(header["fch1"]),
            "foff_mhz": float(header["foff"]),
            "tsamp_s": float(header["tsamp"]),
            "tstart_mjd": float(header["tstart"]),
            "n_time": n_time,
            "n_ifs": n_ifs,
            "n_channels": n_channels,
            "nbits": int(header.get("nbits", 32)),
            "header_barycentric": (
                None if "barycentric" not in header else int(header["barycentric"])
            ),
            "source_name": str(header.get("source_name", "unknown")),
            "source_ra_sigproc": (
                None if "src_raj" not in header else float(header["src_raj"])
            ),
            "source_dec_sigproc": (
                None if "src_dej" not in header else float(header["src_dej"])
            ),
            "telescope_id": (
                None if "telescope_id" not in header else int(header["telescope_id"])
            ),
            "machine_id": (
                None if "machine_id" not in header else int(header["machine_id"])
            ),
            "data_type": (
                None if "data_type" not in header else int(header["data_type"])
            ),
            "reader": "h5py",
        }
        audit["header_fingerprint"] = _header_fingerprint(audit)
        return audit
    except ImportError:
        pass
    except OSError:
        if target.suffix.lower() in {".h5", ".hdf5"}:
            raise

    try:
        import blimpy as bl
    except ImportError as exc:
        raise ImportError(
            "h5py or blimpy is required to inspect filterbank headers"
        ) from exc
    waterfall = bl.Waterfall(str(target), load_data=False)
    header = dict(waterfall.header)
    n_time, n_ifs, n_channels = _shape3(waterfall.file_shape)
    required = ("fch1", "foff", "tsamp", "tstart")
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError("filterbank header is missing %s" % missing)

    audit = {
        "path": str(target),
        "fch1_mhz": float(header["fch1"]),
        "foff_mhz": float(header["foff"]),
        "tsamp_s": float(header["tsamp"]),
        "tstart_mjd": float(header["tstart"]),
        "n_time": n_time,
        "n_ifs": n_ifs,
        "n_channels": n_channels,
        "nbits": int(header.get("nbits", 32)),
        "header_barycentric": (
            None if "barycentric" not in header else int(header["barycentric"])
        ),
        "source_name": str(header.get("source_name", "unknown")),
        "source_ra_sigproc": (
            None if "src_raj" not in header else float(header["src_raj"])
        ),
        "source_dec_sigproc": (
            None if "src_dej" not in header else float(header["src_dej"])
        ),
        "telescope_id": (
            None if "telescope_id" not in header else int(header["telescope_id"])
        ),
        "machine_id": (
            None if "machine_id" not in header else int(header["machine_id"])
        ),
        "data_type": (None if "data_type" not in header else int(header["data_type"])),
        "reader": "blimpy",
    }
    audit["header_fingerprint"] = _header_fingerprint(audit)
    return audit


def _optional_expected(row: Dict[str, str], key: str):
    value = row.get(key, "")
    return None if value is None or not str(value).strip() else finite_float(value, key)


def build_record(row: Dict[str, str], args) -> ObservationRecord:
    audit = inspect_filterbank(row["filterbank_path"])
    expected_step = _optional_expected(row, "expected_signed_foff_hz")
    expected_tsamp = _optional_expected(row, "expected_tsamp_s")
    actual_step = audit["foff_mhz"] * 1e6
    if (
        expected_step is not None
        and abs(actual_step - expected_step) > args.foff_tolerance_hz
    ):
        raise ValueError(
            "%s signed foff %.12g Hz differs from expected %.12g Hz"
            % (row["observation_id"], actual_step, expected_step)
        )
    if (
        expected_tsamp is not None
        and abs(audit["tsamp_s"] - expected_tsamp) > args.tsamp_tolerance_s
    ):
        raise ValueError(
            "%s tsamp %.12g s differs from expected %.12g s"
            % (row["observation_id"], audit["tsamp_s"], expected_tsamp)
        )
    expected_sha256 = str(row.get("expected_file_sha256", "") or "").strip().lower()
    actual_sha256 = ""
    if expected_sha256:
        if len(expected_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in expected_sha256
        ):
            raise ValueError(
                "expected_file_sha256 must be a 64-character hexadecimal digest"
            )
        actual_sha256 = sha256_file(audit["path"])
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "%s file SHA-256 mismatch: expected %s, observed %s"
                % (row["observation_id"], expected_sha256, actual_sha256)
            )

    bank_widths = tuple(
        int(value.strip())
        for value in str(row.get("search_bank_width_channels", "") or "").split(",")
        if value.strip()
    )
    record = ObservationRecord(
        observation_id=str(row["observation_id"]).strip(),
        simultaneous_group_id=str(row["simultaneous_group_id"]).strip(),
        station_id=str(row["station_id"]).strip(),
        filterbank_path=audit["path"],
        barycentric_status=str(row["barycentric_status"]).strip().lower(),
        time_alignment=str(row["time_alignment"]).strip().lower(),
        start_mjd=audit["tstart_mjd"],
        n_time=audit["n_time"],
        n_channels=audit["n_channels"],
        n_ifs=audit["n_ifs"],
        fch1_hz=audit["fch1_mhz"] * 1e6,
        signed_foff_hz=actual_step,
        tsamp_s=audit["tsamp_s"],
        nbits=audit["nbits"],
        header_barycentric=audit["header_barycentric"],
        barycentric_tool=str(row.get("barycentric_tool", "unknown") or "unknown"),
        barycentric_version=str(row.get("barycentric_version", "unknown") or "unknown"),
        source_name=audit["source_name"],
        source_ra_sigproc=audit["source_ra_sigproc"],
        source_dec_sigproc=audit["source_dec_sigproc"],
        telescope_id=audit["telescope_id"],
        machine_id=audit["machine_id"],
        data_type=audit["data_type"],
        header_fingerprint=audit["header_fingerprint"],
        file_sha256=actual_sha256,
        search_fine_channels_per_coarse=int(
            str(row.get("search_fine_channels_per_coarse", "0") or "0").strip()
        ),
        search_rolloff_fraction=float(
            str(row.get("search_rolloff_fraction", "0") or "0").strip()
        ),
        search_drift_min_hz_s=_optional_expected(row, "search_drift_min_hz_s"),
        search_drift_max_hz_s=_optional_expected(row, "search_drift_max_hz_s"),
        search_floor=_optional_expected(row, "search_floor"),
        search_bank_width_channels=bank_widths,
        search_git_commit=str(row.get("search_git_commit", "") or "").strip(),
        search_code_sha256=str(row.get("search_code_sha256", "") or "").strip(),
    )
    record.validate(require_file=True)
    if record.barycentric_status == "barycentric" and record.header_barycentric != 1:
        raise ValueError(
            "%s is labelled barycentric but its HDF5/SIGPROC header does not "
            "contain barycentric=1" % record.observation_id
        )
    if (
        record.time_alignment == "normalized_proxy"
        and record.barycentric_status != "synthetic_proxy"
    ):
        raise ValueError(
            "normalized_proxy records must be labelled barycentric_status=synthetic_proxy"
        )
    if args.require_barycentric_provenance and record.time_alignment == "absolute_mjd":
        if record.barycentric_status != "barycentric":
            raise ValueError(
                "strict real-pair mode requires barycentric_status=barycentric"
            )
        tool = record.barycentric_tool.strip().lower()
        version = record.barycentric_version.strip().lower()
        placeholders = ("edit", "todo", "tbd", "unknown", "pending", "unverified")
        if not tool or any(tool.startswith(value) for value in placeholders):
            raise ValueError("strict real-pair mode requires the barycentric tool name")
        if not version or any(version.startswith(value) for value in placeholders):
            raise ValueError(
                "strict real-pair mode requires the barycentric tool version"
            )
    return record


def main(args) -> None:
    rows = read_csv_dicts(args.input_csv)
    if not rows:
        raise ValueError("observation input CSV is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(
            "observation input CSV is missing columns %s" % sorted(missing)
        )
    records = [build_record(row, args) for row in rows]
    keys = [(item.simultaneous_group_id, item.station_id) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate simultaneous_group_id/station_id record")

    groups = defaultdict(list)
    for record in records:
        groups[record.simultaneous_group_id].append(record)
    for group_id, group_records in sorted(groups.items()):
        try:
            validate_group_observations(group_records, args.allow_normalized_proxy)
        except Exception as exc:
            raise ValueError(
                "invalid simultaneous group %s: %s" % (group_id, exc)
            ) from exc

    write_jsonl(args.output, [item.to_dict() for item in records])
    group_geometry = {}
    for group_id, group_records in sorted(groups.items()):
        frequency_low, frequency_high = common_frequency_bounds_hz(group_records)
        overlap_start = max(item.start_mjd for item in group_records)
        overlap_last_sample = min(item.last_sample_mjd for item in group_records)
        integration_overlap_end = min(item.end_mjd for item in group_records)
        group_geometry[group_id] = {
            "common_frequency_low_hz": frequency_low,
            "common_frequency_high_hz": frequency_high,
            "common_bandwidth_hz": frequency_high - frequency_low,
            "time_overlap_start_mjd": overlap_start,
            "time_overlap_last_sample_mjd": overlap_last_sample,
            "sample_center_overlap_span_seconds": (overlap_last_sample - overlap_start)
            * 86400.0,
            "integration_overlap_end_mjd_exclusive": integration_overlap_end,
            "integration_overlap_seconds": (integration_overlap_end - overlap_start)
            * 86400.0,
            "start_time_difference_seconds": abs(
                group_records[0].start_mjd - group_records[1].start_mjd
            )
            * 86400.0,
            "source_names": [item.source_name for item in group_records],
            "source_name_agreement": len(
                {
                    item.source_name.strip().lower()
                    for item in group_records
                    if item.source_name.strip().lower() not in {"", "unknown", "none"}
                }
            )
            <= 1,
            "source_ra_sigproc": [item.source_ra_sigproc for item in group_records],
            "source_dec_sigproc": [item.source_dec_sigproc for item in group_records],
        }
    summary = {
        "format_version": 1,
        "barycentric_provenance_gate": (
            "verified_tool_and_version_required"
            if args.require_barycentric_provenance
            else "pending_upstream_confirmation_engineering_pilot"
        ),
        "all_headers_declare_barycentric": all(
            item.header_barycentric == 1 for item in records
        ),
        "n_observations": len(records),
        "n_groups": len(groups),
        "station_ids": sorted({item.station_id for item in records}),
        "contains_normalized_proxy": any(
            item.time_alignment == "normalized_proxy" for item in records
        ),
        "observations": [item.to_dict() for item in records],
        "group_geometry": group_geometry,
    }
    write_json(str(Path(args.output).with_suffix(".summary.json")), summary)
    print(
        "Wrote %d observations in %d groups to %s"
        % (len(records), len(groups), args.output)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a header-verified LOFTS observation manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-normalized-proxy", action="store_true")
    parser.add_argument("--foff-tolerance-hz", type=float, default=1e-6)
    parser.add_argument("--tsamp-tolerance-s", type=float, default=1e-9)
    parser.add_argument("--require-barycentric-provenance", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
