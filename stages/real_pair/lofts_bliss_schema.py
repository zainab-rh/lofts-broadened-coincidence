#!/usr/bin/env python3
"""Canonical, auditable schemas for the LOFTS BLISS -> Stage-4 interface.

The operational pipeline deliberately exchanges JSONL records rather than
passing implementation-specific Python objects between BLISS and Stage 4.
This makes units, frequency epochs, station identity, barycentric status and
provenance explicit and reviewable.

Operational candidate records contain no injected truth identifiers.  Any
Synthetic-Test-B candidate-to-injection links live in a separate sidecar that
is first read after label-blind inference has been frozen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

SCHEMA_VERSION = 1
SECONDS_PER_DAY = 86400.0
ALLOWED_TIME_ALIGNMENT = {"absolute_mjd", "normalized_proxy"}
ALLOWED_BARYCENTRIC_STATUS = {
    "barycentric",
    "topocentric",
    "synthetic_proxy",
    "unknown",
}
FORBIDDEN_CANDIDATE_EXTRA_KEYS = {
    "event_id",
    "injection_id",
    "label",
    "pair_label",
    "recovery_link_id",
    "truth",
}
ALLOWED_CANDIDATE_EXTRA_KEYS = {
    "resampling_block_id",
    # Naoise blind-hit-finder operational metadata.  These fields are all
    # detector outputs or provenance; injected truth remains forbidden.
    "native_width_channels",
    "channel_bandwidth_hz",
    "bank_snr",
    "standard_snr",
    "bank_standard_ratio",
    "source_flag",
    "templates_skipped",
    "broadband_rfi_like",
    "naoise_floor",
    "bank_width_channels",
    "per_template_snr",
    "stage4_restricted_width_channels",
    "stage4_restricted_width_hz",
    "stage4_restricted_snr",
    "stage4_restricted_above_floor",
    "winning_within_tolerance",
    "source_candidate_id",
    "search_git_commit",
    "catalog_role",
    "own_search_coverage",
}


def finite_float(value: Any, name: str) -> float:
    """Convert *value* to a finite float with a field-specific error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric; received %r" % (name, value)) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be finite; received %r" % (name, value))
    return result


def optional_finite_float(value: Any, name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return finite_float(value, name)


def nonempty_text(value: Any, name: str) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise ValueError("%s must be non-empty" % name)
    return result


def parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError("%s must be boolean; received %r" % (name, value))


def unit_scale(unit: str, quantity: str) -> float:
    """Return a strict multiplicative scale into canonical SI-like units."""

    key = str(unit).strip().lower().replace(" ", "")
    tables = {
        "frequency": {
            "hz": 1.0,
            "khz": 1e3,
            "mhz": 1e6,
            "ghz": 1e9,
        },
        "width": {
            "hz": 1.0,
            "khz": 1e3,
            "mhz": 1e6,
        },
        "drift": {
            "hz/s": 1.0,
            "hzs-1": 1.0,
            "hzsec-1": 1.0,
            "khz/s": 1e3,
            "mhz/s": 1e6,
        },
        "time": {
            "s": 1.0,
            "sec": 1.0,
            "ms": 1e-3,
            "day": SECONDS_PER_DAY,
            "d": SECONDS_PER_DAY,
        },
    }
    if quantity not in tables or key not in tables[quantity]:
        raise ValueError(
            "unsupported %s unit %r; allowed=%s"
            % (quantity, unit, sorted(tables.get(quantity, {})))
        )
    return tables[quantity][key]


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[: int(length)]
    return "%s_%s" % (prefix, digest)


def sha256_file(path: str, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy-like scalar values without importing NumPy."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def atomic_write_text(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(target))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json(path: str, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    )


def write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(json_safe(record), sort_keys=True) for record in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_json_records(path: str) -> List[Dict[str, Any]]:
    """Read a JSON array/object or JSONL file into dictionaries."""

    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if value is not None:
            if isinstance(value, list):
                records = value
            elif isinstance(value, dict) and isinstance(value.get("records"), list):
                records = value["records"]
            elif isinstance(value, dict):
                records = [value]
            else:
                raise ValueError("JSON root must be an object or array")
            if not all(isinstance(record, dict) for record in records):
                raise ValueError("all JSON records must be objects")
            return list(records)
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSONL at line %d" % line_number) from exc
        if not isinstance(record, dict):
            raise ValueError("JSONL line %d is not an object" % line_number)
        records.append(record)
    return records


def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: %s" % path)
        return [dict(row) for row in reader]


@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    simultaneous_group_id: str
    station_id: str
    filterbank_path: str
    barycentric_status: str
    time_alignment: str
    start_mjd: float
    n_time: int
    n_channels: int
    n_ifs: int
    fch1_hz: float
    signed_foff_hz: float
    tsamp_s: float
    nbits: int
    header_barycentric: Optional[int] = None
    barycentric_tool: str = "unknown"
    barycentric_version: str = "unknown"
    source_name: str = "unknown"
    source_ra_sigproc: Optional[float] = None
    source_dec_sigproc: Optional[float] = None
    telescope_id: Optional[int] = None
    machine_id: Optional[int] = None
    data_type: Optional[int] = None
    header_fingerprint: str = ""
    file_sha256: str = ""
    search_fine_channels_per_coarse: int = 0
    search_rolloff_fraction: float = 0.0
    search_drift_min_hz_s: Optional[float] = None
    search_drift_max_hz_s: Optional[float] = None
    search_floor: Optional[float] = None
    search_bank_width_channels: Tuple[int, ...] = field(default_factory=tuple)
    search_git_commit: str = ""
    search_code_sha256: str = ""
    schema_version: int = SCHEMA_VERSION

    def validate(self, require_file: bool = False) -> None:
        for name in (
            "observation_id",
            "simultaneous_group_id",
            "station_id",
            "filterbank_path",
        ):
            nonempty_text(getattr(self, name), name)
        if self.barycentric_status not in ALLOWED_BARYCENTRIC_STATUS:
            raise ValueError("invalid barycentric_status %r" % self.barycentric_status)
        if self.time_alignment not in ALLOWED_TIME_ALIGNMENT:
            raise ValueError("invalid time_alignment %r" % self.time_alignment)
        finite_float(self.start_mjd, "start_mjd")
        finite_float(self.fch1_hz, "fch1_hz")
        if finite_float(self.signed_foff_hz, "signed_foff_hz") == 0:
            raise ValueError("signed_foff_hz must not be zero")
        if finite_float(self.tsamp_s, "tsamp_s") <= 0:
            raise ValueError("tsamp_s must be positive")
        if (
            min(
                int(self.n_time), int(self.n_channels), int(self.n_ifs), int(self.nbits)
            )
            <= 0
        ):
            raise ValueError("shape and nbits fields must be positive")
        if self.source_ra_sigproc is not None:
            finite_float(self.source_ra_sigproc, "source_ra_sigproc")
        if self.source_dec_sigproc is not None:
            finite_float(self.source_dec_sigproc, "source_dec_sigproc")
        for name in ("telescope_id", "machine_id", "data_type"):
            value = getattr(self, name)
            if value is not None:
                int(value)
        if self.header_barycentric is not None and int(self.header_barycentric) not in (
            0,
            1,
        ):
            raise ValueError("header_barycentric must be 0, 1, or null")
        if int(self.search_fine_channels_per_coarse) < 0:
            raise ValueError("search_fine_channels_per_coarse must be non-negative")
        rolloff = finite_float(self.search_rolloff_fraction, "search_rolloff_fraction")
        if not (0.0 <= rolloff < 0.5):
            raise ValueError("search_rolloff_fraction must satisfy 0 <= value < 0.5")
        if (self.search_drift_min_hz_s is None) != (self.search_drift_max_hz_s is None):
            raise ValueError(
                "search drift limits must either both be present or both be absent"
            )
        if self.search_drift_min_hz_s is not None:
            low = finite_float(self.search_drift_min_hz_s, "search_drift_min_hz_s")
            high = finite_float(self.search_drift_max_hz_s, "search_drift_max_hz_s")
            if not low < high:
                raise ValueError("search drift limits must satisfy low < high")
        if (
            self.search_floor is not None
            and finite_float(self.search_floor, "search_floor") <= 0
        ):
            raise ValueError("search_floor must be positive")
        widths = tuple(int(value) for value in self.search_bank_width_channels)
        if any(value <= 0 for value in widths) or len(widths) != len(set(widths)):
            raise ValueError(
                "search_bank_width_channels must be unique positive integers"
            )
        if require_file and not Path(self.filterbank_path).is_file():
            raise FileNotFoundError(self.filterbank_path)

    @property
    def duration_s(self) -> float:
        return float(self.n_time) * float(self.tsamp_s)

    @property
    def midpoint_mjd(self) -> float:
        return (
            float(self.start_mjd)
            + 0.5 * (float(self.n_time) - 1.0) * float(self.tsamp_s) / SECONDS_PER_DAY
        )

    @property
    def last_sample_mjd(self) -> float:
        """Timestamp of the final sampled integration on the row-index grid."""

        return (
            float(self.start_mjd)
            + (float(self.n_time) - 1.0) * float(self.tsamp_s) / SECONDS_PER_DAY
        )

    @property
    def end_mjd(self) -> float:
        """Exclusive integration-span endpoint, one cadence after the last row."""

        return (
            float(self.start_mjd)
            + float(self.n_time) * float(self.tsamp_s) / SECONDS_PER_DAY
        )

    def frequency_hz_for_channel(self, channel: float) -> float:
        return float(self.fch1_hz) + float(channel) * float(self.signed_foff_hz)

    def channel_for_frequency_hz(self, frequency_hz: float) -> float:
        return (float(frequency_hz) - float(self.fch1_hz)) / float(self.signed_foff_hz)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "ObservationRecord":
        item = cls(
            observation_id=nonempty_text(
                record.get("observation_id"), "observation_id"
            ),
            simultaneous_group_id=nonempty_text(
                record.get("simultaneous_group_id"), "simultaneous_group_id"
            ),
            station_id=nonempty_text(record.get("station_id"), "station_id"),
            filterbank_path=nonempty_text(
                record.get("filterbank_path"), "filterbank_path"
            ),
            barycentric_status=nonempty_text(
                record.get("barycentric_status"), "barycentric_status"
            ).lower(),
            time_alignment=nonempty_text(
                record.get("time_alignment", "absolute_mjd"), "time_alignment"
            ).lower(),
            start_mjd=finite_float(record.get("start_mjd"), "start_mjd"),
            n_time=int(record.get("n_time")),
            n_channels=int(record.get("n_channels")),
            n_ifs=int(record.get("n_ifs", 1)),
            fch1_hz=finite_float(record.get("fch1_hz"), "fch1_hz"),
            signed_foff_hz=finite_float(record.get("signed_foff_hz"), "signed_foff_hz"),
            tsamp_s=finite_float(record.get("tsamp_s"), "tsamp_s"),
            nbits=int(record.get("nbits", 32)),
            header_barycentric=(
                None
                if record.get("header_barycentric") in (None, "")
                else int(record.get("header_barycentric"))
            ),
            barycentric_tool=str(record.get("barycentric_tool", "unknown")),
            barycentric_version=str(record.get("barycentric_version", "unknown")),
            source_name=str(record.get("source_name", "unknown")),
            source_ra_sigproc=optional_finite_float(
                record.get("source_ra_sigproc"), "source_ra_sigproc"
            ),
            source_dec_sigproc=optional_finite_float(
                record.get("source_dec_sigproc"), "source_dec_sigproc"
            ),
            telescope_id=(
                None
                if record.get("telescope_id") in (None, "")
                else int(record.get("telescope_id"))
            ),
            machine_id=(
                None
                if record.get("machine_id") in (None, "")
                else int(record.get("machine_id"))
            ),
            data_type=(
                None
                if record.get("data_type") in (None, "")
                else int(record.get("data_type"))
            ),
            header_fingerprint=str(record.get("header_fingerprint", "")),
            file_sha256=str(record.get("file_sha256", "")),
            search_fine_channels_per_coarse=int(
                record.get("search_fine_channels_per_coarse", 0) or 0
            ),
            search_rolloff_fraction=finite_float(
                record.get("search_rolloff_fraction", 0.0) or 0.0,
                "search_rolloff_fraction",
            ),
            search_drift_min_hz_s=optional_finite_float(
                record.get("search_drift_min_hz_s"), "search_drift_min_hz_s"
            ),
            search_drift_max_hz_s=optional_finite_float(
                record.get("search_drift_max_hz_s"), "search_drift_max_hz_s"
            ),
            search_floor=optional_finite_float(
                record.get("search_floor"), "search_floor"
            ),
            search_bank_width_channels=tuple(
                int(value) for value in record.get("search_bank_width_channels", ())
            ),
            search_git_commit=str(record.get("search_git_commit", "")),
            search_code_sha256=str(record.get("search_code_sha256", "")),
            schema_version=int(record.get("schema_version", SCHEMA_VERSION)),
        )
        item.validate(require_file=False)
        return item


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    observation_id: str
    simultaneous_group_id: str
    station_id: str
    frequency_hz: float
    drift_hz_s: float
    width_hz: float
    width_definition: str
    snr: float
    snr_definition: str
    frequency_ref_mjd: Optional[float]
    frequency_ref_offset_s: Optional[float]
    detected: bool = True
    metadata_source: str = "BLISS"
    source_table: str = ""
    source_row: int = -1
    extras: Dict[str, Any] = field(default_factory=dict)
    truth: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        for name in (
            "candidate_id",
            "observation_id",
            "simultaneous_group_id",
            "station_id",
        ):
            nonempty_text(getattr(self, name), name)
        finite_float(self.frequency_hz, "frequency_hz")
        finite_float(self.drift_hz_s, "drift_hz_s")
        if finite_float(self.width_hz, "width_hz") <= 0:
            raise ValueError("width_hz must be positive")
        finite_float(self.snr, "snr")
        nonempty_text(self.width_definition, "width_definition")
        nonempty_text(self.snr_definition, "snr_definition")
        extra_keys = {str(key).strip().lower() for key in self.extras}
        forbidden = FORBIDDEN_CANDIDATE_EXTRA_KEYS.intersection(extra_keys)
        if forbidden:
            raise ValueError(
                "candidate extras contain truth-adjacent keys: %s" % sorted(forbidden)
            )
        unsupported = extra_keys - ALLOWED_CANDIDATE_EXTRA_KEYS
        if unsupported:
            raise ValueError(
                "candidate extras are not operationally allow-listed: %s"
                % sorted(unsupported)
            )
        if self.truth:
            raise ValueError(
                "operational CandidateRecord must not contain injected truth"
            )
        if (self.frequency_ref_mjd is None) == (self.frequency_ref_offset_s is None):
            raise ValueError(
                "exactly one of frequency_ref_mjd or frequency_ref_offset_s must be set"
            )
        if self.frequency_ref_mjd is not None:
            finite_float(self.frequency_ref_mjd, "frequency_ref_mjd")
        if self.frequency_ref_offset_s is not None:
            finite_float(self.frequency_ref_offset_s, "frequency_ref_offset_s")

    def frequency_at_mjd(self, target_mjd: float) -> float:
        if self.frequency_ref_mjd is None:
            raise ValueError("candidate uses normalized proxy time, not absolute MJD")
        elapsed_s = (
            float(target_mjd) - float(self.frequency_ref_mjd)
        ) * SECONDS_PER_DAY
        return float(self.frequency_hz) + float(self.drift_hz_s) * elapsed_s

    def frequency_at_offset_s(self, target_offset_s: float) -> float:
        if self.frequency_ref_offset_s is None:
            raise ValueError("candidate uses absolute MJD, not normalized proxy time")
        elapsed_s = float(target_offset_s) - float(self.frequency_ref_offset_s)
        return float(self.frequency_hz) + float(self.drift_hz_s) * elapsed_s

    def to_dict(self, include_truth: bool = True) -> Dict[str, Any]:
        result = asdict(self)
        if not include_truth:
            result.pop("truth", None)
        return result

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CandidateRecord":
        item = cls(
            candidate_id=nonempty_text(record.get("candidate_id"), "candidate_id"),
            observation_id=nonempty_text(
                record.get("observation_id"), "observation_id"
            ),
            simultaneous_group_id=nonempty_text(
                record.get("simultaneous_group_id"), "simultaneous_group_id"
            ),
            station_id=nonempty_text(record.get("station_id"), "station_id"),
            frequency_hz=finite_float(record.get("frequency_hz"), "frequency_hz"),
            drift_hz_s=finite_float(record.get("drift_hz_s"), "drift_hz_s"),
            width_hz=finite_float(record.get("width_hz"), "width_hz"),
            width_definition=nonempty_text(
                record.get("width_definition"), "width_definition"
            ),
            snr=finite_float(record.get("snr"), "snr"),
            snr_definition=nonempty_text(
                record.get("snr_definition"), "snr_definition"
            ),
            frequency_ref_mjd=optional_finite_float(
                record.get("frequency_ref_mjd"), "frequency_ref_mjd"
            ),
            frequency_ref_offset_s=optional_finite_float(
                record.get("frequency_ref_offset_s"), "frequency_ref_offset_s"
            ),
            detected=parse_bool(record.get("detected", True), "detected"),
            metadata_source=str(record.get("metadata_source", "BLISS")),
            source_table=str(record.get("source_table", "")),
            source_row=int(record.get("source_row", -1)),
            extras=dict(record.get("extras", {})),
            truth=dict(record.get("truth", {})),
            schema_version=int(record.get("schema_version", SCHEMA_VERSION)),
        )
        item.validate()
        return item


@dataclass(frozen=True)
class InjectionTruthRecord:
    injection_id: str
    event_id: str
    observation_id: str
    simultaneous_group_id: str
    station_id: str
    frequency_hz: float
    drift_hz_s: float
    width_hz: float
    snr: float
    frequency_ref_mjd: Optional[float]
    frequency_ref_offset_s: Optional[float]
    shape: str = "unknown"
    pair_label: Optional[int] = None
    case: str = "unknown"
    source_row: int = -1
    extras: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        for name in (
            "injection_id",
            "event_id",
            "observation_id",
            "simultaneous_group_id",
            "station_id",
        ):
            nonempty_text(getattr(self, name), name)
        finite_float(self.frequency_hz, "frequency_hz")
        finite_float(self.drift_hz_s, "drift_hz_s")
        if finite_float(self.width_hz, "width_hz") <= 0:
            raise ValueError("width_hz must be positive")
        finite_float(self.snr, "snr")
        if (self.frequency_ref_mjd is None) == (self.frequency_ref_offset_s is None):
            raise ValueError(
                "exactly one of frequency_ref_mjd or frequency_ref_offset_s must be set"
            )
        if self.pair_label is not None and int(self.pair_label) not in (0, 1):
            raise ValueError("pair_label must be 0, 1 or null")

    def frequency_at_mjd(self, target_mjd: float) -> float:
        if self.frequency_ref_mjd is None:
            raise ValueError("truth record uses normalized proxy time")
        return (
            float(self.frequency_hz)
            + float(self.drift_hz_s)
            * (float(target_mjd) - float(self.frequency_ref_mjd))
            * SECONDS_PER_DAY
        )

    def frequency_at_offset_s(self, target_offset_s: float) -> float:
        if self.frequency_ref_offset_s is None:
            raise ValueError("truth record uses absolute MJD")
        return float(self.frequency_hz) + float(self.drift_hz_s) * (
            float(target_offset_s) - float(self.frequency_ref_offset_s)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "InjectionTruthRecord":
        label = record.get("pair_label")
        item = cls(
            injection_id=nonempty_text(record.get("injection_id"), "injection_id"),
            event_id=nonempty_text(
                record.get("event_id", record.get("injection_id")), "event_id"
            ),
            observation_id=nonempty_text(
                record.get("observation_id"), "observation_id"
            ),
            simultaneous_group_id=nonempty_text(
                record.get("simultaneous_group_id"), "simultaneous_group_id"
            ),
            station_id=nonempty_text(record.get("station_id"), "station_id"),
            frequency_hz=finite_float(record.get("frequency_hz"), "frequency_hz"),
            drift_hz_s=finite_float(record.get("drift_hz_s"), "drift_hz_s"),
            width_hz=finite_float(record.get("width_hz"), "width_hz"),
            snr=finite_float(record.get("snr"), "snr"),
            frequency_ref_mjd=optional_finite_float(
                record.get("frequency_ref_mjd"), "frequency_ref_mjd"
            ),
            frequency_ref_offset_s=optional_finite_float(
                record.get("frequency_ref_offset_s"), "frequency_ref_offset_s"
            ),
            shape=str(record.get("shape", "unknown")),
            pair_label=None if label in (None, "") else int(label),
            case=str(record.get("case", "unknown")),
            source_row=int(record.get("source_row", -1)),
            extras=dict(record.get("extras", {})),
            schema_version=int(record.get("schema_version", SCHEMA_VERSION)),
        )
        item.validate()
        return item


def load_observations(
    path: str, require_files: bool = False
) -> Dict[Tuple[str, str], ObservationRecord]:
    result: Dict[Tuple[str, str], ObservationRecord] = {}
    for raw in read_json_records(path):
        item = ObservationRecord.from_dict(raw)
        item.validate(require_file=require_files)
        key = (item.simultaneous_group_id, item.station_id)
        if key in result:
            raise ValueError("duplicate observation key %r" % (key,))
        result[key] = item
    if not result:
        raise ValueError("observation manifest is empty")
    return result


def load_candidates(path: str) -> List[CandidateRecord]:
    result = [CandidateRecord.from_dict(raw) for raw in read_json_records(path)]
    ids = [item.candidate_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be globally unique")
    return result


def load_truth(path: str) -> List[InjectionTruthRecord]:
    result = [InjectionTruthRecord.from_dict(raw) for raw in read_json_records(path)]
    ids = [item.injection_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("injection_id values must be globally unique")
    return result


def validate_group_observations(
    observations: Sequence[ObservationRecord],
    allow_normalized_proxy: bool,
) -> None:
    if len(observations) != 2:
        raise ValueError("each simultaneous group must contain exactly two stations")
    if observations[0].station_id == observations[1].station_id:
        raise ValueError("station IDs must differ")
    modes = {item.time_alignment for item in observations}
    if len(modes) != 1:
        raise ValueError("both observations must use the same time_alignment")
    mode = next(iter(modes))
    if mode == "normalized_proxy":
        if not allow_normalized_proxy:
            raise ValueError(
                "normalized_proxy is synthetic-only; pass an explicit opt-in"
            )
        return
    overlap_start = max(item.start_mjd for item in observations)
    overlap_last_sample = min(item.last_sample_mjd for item in observations)
    if overlap_last_sample < overlap_start:
        raise ValueError(
            "absolute-MJD station observations have no overlapping sampled timestamps"
        )
    if any(item.barycentric_status != "barycentric" for item in observations):
        raise ValueError(
            "real absolute-MJD union requires barycentric products at both stations"
        )
    known_names = {
        item.source_name.strip().lower()
        for item in observations
        if item.source_name.strip().lower() not in {"", "unknown", "none"}
    }
    if len(known_names) > 1:
        raise ValueError(
            "real station products declare different source names: %s"
            % sorted(known_names)
        )
    for field_name in ("source_ra_sigproc", "source_dec_sigproc"):
        values = [getattr(item, field_name) for item in observations]
        if (
            all(value is not None for value in values)
            and abs(float(values[0]) - float(values[1])) > 1e-6
        ):
            raise ValueError(
                "real station products declare different %s values: %s"
                % (field_name, values)
            )


def group_reference_time(
    observations: Sequence[ObservationRecord],
) -> Tuple[str, float]:
    """Return (``mjd`` or ``offset_s``, comparison reference value)."""

    if observations[0].time_alignment == "normalized_proxy":
        return "offset_s", 0.5 * min(item.duration_s for item in observations)
    overlap_start = max(item.start_mjd for item in observations)
    overlap_last_sample = min(item.last_sample_mjd for item in observations)
    return "mjd", 0.5 * (overlap_start + overlap_last_sample)


def candidate_frequency_at_reference(
    candidate: CandidateRecord,
    reference_kind: str,
    reference_value: float,
) -> float:
    if reference_kind == "mjd":
        return candidate.frequency_at_mjd(reference_value)
    if reference_kind == "offset_s":
        return candidate.frequency_at_offset_s(reference_value)
    raise ValueError("unknown reference_kind %r" % reference_kind)
