#!/usr/bin/env python3
"""Create an auditable visual report for a paired Test-B engineering smoke run.

The program reads existing injection, HDF5, BLISS-catalogue and log artifacts.
It does not inject signals, rerun BLISS, change associations, or run Stage 4.
Truth is used only after the search as a diagnostic overlay. Consequently the
figures demonstrate software flow and one-event recovery behaviour; they are
not estimates of completeness, false-positive rate, AUC, or generalisation.

Both Naoise's ``paired_inject.py`` v1.1 smoke layout and the repository's
collision-free campaign layout are supported. Paths may be supplied directly,
or discovered below ``--run-dir`` when their names and schemas are unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lofts-matplotlib-cache")
)


RAW_CANDIDATE_REQUIRED = {
    "OBS",
    "CANDIDATE_ID",
    "STATION",
    "FREQ_MHZ",
    "DR_HZ_S",
    "WIDTH_HZ",
    "BANK_SNR",
}

COARSE_PATTERN = re.compile(
    r"coarse\s+(?P<index>\d+)\s*/\s*(?P<last>\d+).*?"
    r"(?P<low>[-+0-9.eE]+)-(?P<high>[-+0-9.eE]+)\s+MHz.*?"
    r"drift\[(?P<drift_low>[-+0-9.eE]+),(?P<drift_high>[-+0-9.eE]+)\].*?"
    r"pix=(?P<pixels>\d+)\s+cand=(?P<candidates>\d+)"
)

STATIONS = ("IRL", "SWE")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> Optional[bool]:
    if value is None or not str(value).strip():
        return None
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "y", "recovered", "found"}:
        return True
    if normalised in {"0", "false", "no", "n", "missed", "not_found"}:
        return False
    return None


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be numeric; received %r" % (name, value)) from exc
    if not math.isfinite(result):
        raise ValueError("%s must be finite; received %r" % (name, value))
    return result


def column_name(
    row: Mapping[str, Any], aliases: Sequence[str], required: bool = True
) -> Optional[str]:
    lookup = {str(key).strip().lower(): str(key) for key in row}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError("missing required column; expected one of %s" % list(aliases))
    return None


def one_file(paths: Sequence[Path], description: str) -> Optional[Path]:
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(
            "multiple %s files found; select one explicitly: %s"
            % (description, [str(path) for path in unique])
        )
    return unique[0]


def discover_named(run_dir: Path, name: str) -> Optional[Path]:
    return one_file(list(run_dir.rglob(name)), name)


def station_from_text(value: str) -> Optional[str]:
    upper = str(value).upper()
    if re.search(r"(^|[^A-Z])IRL([^A-Z]|$)", upper) or "IRELAND" in upper:
        return "IRL"
    if re.search(r"(^|[^A-Z])SWE([^A-Z]|$)", upper) or "SWEDEN" in upper:
        return "SWE"
    return None


def truth_index(rows: Sequence[Mapping[str, Any]], event_id: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not rows:
        raise ValueError("truth table is empty")
    station_key = column_name(rows[0], ("station", "station_id"))
    event_key = column_name(rows[0], ("event_id",), required=False)
    injected_key = column_name(rows[0], ("injected",), required=False)
    selected: Dict[str, Dict[str, Any]] = {}
    available_events = set()
    for raw in rows:
        row = dict(raw)
        if event_key and str(row.get(event_key, "")).strip():
            available_events.add(str(row[event_key]).strip())
        if event_id is not None and event_key and str(row.get(event_key)) != event_id:
            continue
        if injected_key and parse_bool(row.get(injected_key)) is False:
            continue
        station = station_from_text(str(row.get(station_key, "")))
        if station not in STATIONS:
            continue
        if station in selected:
            raise ValueError(
                "truth contains multiple %s rows; select one event with --event-id"
                % station
            )
        selected[station] = row
    if event_id is None and len(available_events) > 1:
        raise ValueError(
            "truth contains multiple events; use --event-id from %s"
            % sorted(available_events)
        )
    missing = sorted(set(STATIONS) - set(selected))
    if missing:
        raise ValueError("truth lacks injected rows for stations %s" % missing)
    return selected


def truth_value(row: Mapping[str, Any], aliases: Sequence[str]) -> float:
    key = column_name(row, aliases)
    return finite(row[key], key)


def parse_search_log(path: Path) -> List[List[Dict[str, Any]]]:
    """Parse one or more sequential blind-search progress streams."""

    searches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_index: Optional[int] = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = COARSE_PATTERN.search(line)
            if not match:
                continue
            record: Dict[str, Any] = {
                "index": int(match.group("index")),
                "last": int(match.group("last")),
                "frequency_low_mhz": float(match.group("low")),
                "frequency_high_mhz": float(match.group("high")),
                "drift_low_hz_s": float(match.group("drift_low")),
                "drift_high_hz_s": float(match.group("drift_high")),
                "pixels": int(match.group("pixels")),
                "candidates": int(match.group("candidates")),
            }
            if previous_index is not None and record["index"] <= previous_index:
                searches.append(current)
                current = []
            current.append(record)
            previous_index = record["index"]
    if current:
        searches.append(current)
    return searches


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, base: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in formats:
        path = base.with_suffix("." + suffix)
        fig.savefig(
            path,
            format=suffix,
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "LOFTS Test-B smoke-report builder"},
        )
        outputs.append(path)
    return outputs


def parse_formats(value: str) -> Tuple[str, ...]:
    formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(formats) - {"png", "pdf", "svg"})
    if not formats or invalid:
        raise argparse.ArgumentTypeError("formats must use png, pdf, or svg")
    return formats


def plot_search_progress(
    searches: Sequence[Sequence[Mapping[str, Any]]],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[Path], Dict[str, Any]]:
    if not searches:
        raise ValueError("search log contains no coarse-channel progress records")
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.3), sharex=False)
    palette = ("#326A8C", "#D97742", "#4C9F70", "#7C5AA6")
    summaries = []
    for number, records in enumerate(searches, start=1):
        indices = np.asarray([int(item["index"]) for item in records])
        candidates = np.asarray([int(item["candidates"]) for item in records])
        pixels = np.asarray([int(item["pixels"]) for item in records])
        total = int(records[-1]["last"]) + 1
        complete = int(records[-1]["index"]) == int(records[-1]["last"])
        label = "Search %d%s" % (number, " (complete)" if complete else " (running)")
        color = palette[(number - 1) % len(palette)]
        axes[0].plot(indices, np.cumsum(candidates), lw=2.2, color=color, label=label)
        axes[1].plot(indices, candidates, lw=1.2, color=color, alpha=0.9, label=label)
        summaries.append(
            {
                "search_number": number,
                "coarse_channels_recorded": len(records),
                "coarse_channels_expected": total,
                "complete": complete,
                "candidate_count_sum": int(candidates.sum()),
                "pixel_count_sum": int(pixels.sum()),
                "drift_range_hz_s": [
                    float(records[-1]["drift_low_hz_s"]),
                    float(records[-1]["drift_high_hz_s"]),
                ],
            }
        )
    axes[0].set_title("Cumulative blind-search candidates")
    axes[0].set_ylabel("candidates reported by coarse-channel searches")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].set_title("Candidate load by coarse channel")
    axes[1].set_xlabel("coarse-channel index")
    axes[1].set_ylabel("candidates")
    axes[1].grid(alpha=0.22)
    fig.suptitle(
        "Paired Test-B smoke: independent BLISS search progress",
        fontsize=18,
        fontweight="bold",
        color="#17324D",
    )
    fig.text(
        0.5,
        0.02,
        "Engineering diagnostic only. Candidate counts are search workload, not detections of injected truth.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.86, bottom=0.11, hspace=0.32)
    outputs = save_figure(fig, out_dir / "01_bliss_search_progress", formats, dpi)
    plt.close(fig)
    return outputs, {"searches": summaries}


def parse_station_paths(values: Sequence[str], option: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("%s values must use STATION=/path/to/file.csv" % option)
        station_text, path_text = value.split("=", 1)
        station = station_from_text(station_text)
        if station not in STATIONS or station in result:
            raise ValueError("%s must identify IRL or SWE exactly once" % option)
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[station] = path
    return result


def candidate_catalogues(
    run_dir: Path, explicit: Mapping[str, Path]
) -> Tuple[Dict[str, List[Dict[str, str]]], List[Path]]:
    catalogues: Dict[str, List[Dict[str, str]]] = {station: [] for station in STATIONS}
    used: List[Path] = []
    paths = list(explicit.values()) + [
        path for path in sorted(run_dir.rglob("*.csv")) if path.resolve() not in set(explicit.values())
    ]
    for path in paths:
        if path.name.lower() in {
            "truth.csv",
            "truth_irl.csv",
            "truth_swe.csv",
            "recovery.csv",
            "search_plan.csv",
            "observations.csv",
        }:
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            if not RAW_CANDIDATE_REQUIRED.issubset(columns):
                continue
            rows = [dict(row) for row in reader]
        if not rows:
            continue
        stations = {station_from_text(row.get("STATION", "")) for row in rows}
        stations.discard(None)
        if len(stations) != 1:
            raise ValueError("raw candidate table mixes station IDs: %s" % path)
        station = next(iter(stations))
        if explicit and path in explicit.values() and explicit.get(station) != path:
            raise ValueError("explicit candidate file station disagrees with its rows: %s" % path)
        catalogues[station].extend(rows)  # type: ignore[index]
        used.append(path.resolve())
    return catalogues, used


def select_nearby_candidate(
    rows: Sequence[Mapping[str, Any]],
    truth_row: Mapping[str, Any],
    frequency_epoch: str,
    frequency_gate_khz: float,
    drift_gate_hz_s: float,
) -> Dict[str, Any]:
    truth_frequency = truth_value(
        truth_row,
        ("f_first_MHz",) if frequency_epoch == "first_row" else ("f_ref_MHz",),
    )
    truth_drift = truth_value(truth_row, ("drift_Hz_s", "drift_hz_s"))
    nearby = []
    associated = []
    for row in rows:
        frequency = finite(row["FREQ_MHZ"], "FREQ_MHZ")
        drift = finite(row["DR_HZ_S"], "DR_HZ_S")
        df_khz = (frequency - truth_frequency) * 1000.0
        dd = drift - truth_drift
        item = {
            "row": dict(row),
            "frequency_residual_khz": df_khz,
            "drift_residual_hz_s": dd,
            "distance": math.hypot(
                df_khz / frequency_gate_khz, dd / drift_gate_hz_s
            ),
        }
        if abs(df_khz) <= max(5.0 * frequency_gate_khz, 2.0) and abs(dd) <= max(
            5.0 * drift_gate_hz_s, 0.04
        ):
            nearby.append(item)
        if abs(df_khz) <= frequency_gate_khz and abs(dd) <= drift_gate_hz_s:
            associated.append(item)
    associated.sort(key=lambda item: (item["distance"], str(item["row"]["CANDIDATE_ID"])))
    return {
        "truth_frequency_mhz": truth_frequency,
        "truth_drift_hz_s": truth_drift,
        "nearby": nearby,
        "associated": associated,
        "primary": None if not associated else associated[0],
    }


def plot_candidate_recovery(
    catalogues: Mapping[str, Sequence[Mapping[str, Any]]],
    truths: Mapping[str, Mapping[str, Any]],
    frequency_epoch: str,
    frequency_gate_khz: float,
    drift_gate_hz_s: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[Path], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    matches = {
        station: select_nearby_candidate(
            catalogues.get(station, []),
            truths[station],
            frequency_epoch,
            frequency_gate_khz,
            drift_gate_hz_s,
        )
        for station in STATIONS
    }
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), sharex=True, sharey=True)
    from matplotlib.patches import Rectangle

    all_snr = [
        finite(item["row"]["BANK_SNR"], "BANK_SNR")
        for station in STATIONS
        for item in matches[station]["nearby"]
    ]
    snr_low = min(all_snr) if all_snr else 0.0
    snr_high = max(all_snr) if all_snr else 1.0
    if snr_high <= snr_low:
        snr_high = snr_low + 1.0
    artist = None
    summary: Dict[str, Any] = {}
    for axis, station in zip(axes, STATIONS):
        nearby = matches[station]["nearby"]
        associated = matches[station]["associated"]
        if nearby:
            artist = axis.scatter(
                [item["frequency_residual_khz"] for item in nearby],
                [item["drift_residual_hz_s"] for item in nearby],
                c=[finite(item["row"]["BANK_SNR"], "BANK_SNR") for item in nearby],
                cmap="viridis",
                vmin=snr_low,
                vmax=snr_high,
                s=32,
                alpha=0.72,
                edgecolors="none",
            )
        axis.add_patch(
            Rectangle(
                (-frequency_gate_khz, -drift_gate_hz_s),
                2.0 * frequency_gate_khz,
                2.0 * drift_gate_hz_s,
                fill=False,
                edgecolor="#D97742",
                lw=2,
                linestyle="--",
                label="engineering association gate",
            )
        )
        axis.scatter([0], [0], marker="*", s=210, c="#C43D3D", edgecolors="white", linewidths=0.8, label="injected truth")
        if associated:
            primary = associated[0]
            axis.scatter(
                [primary["frequency_residual_khz"]],
                [primary["drift_residual_hz_s"]],
                marker="o",
                s=180,
                facecolors="none",
                edgecolors="#111827",
                linewidths=2,
                label="nearest in gate",
            )
        axis.axhline(0, color="#9CA3AF", lw=0.8)
        axis.axvline(0, color="#9CA3AF", lw=0.8)
        axis.grid(alpha=0.20)
        axis.set_title(
            "%s: %s"
            % (station, "candidate in gate" if associated else "no candidate in gate"),
            fontweight="bold",
            color="#17324D",
        )
        axis.set_xlabel("candidate − injected frequency (kHz)")
        summary[station] = {
            "n_catalogue_candidates": len(catalogues.get(station, [])),
            "n_nearby_candidates": len(nearby),
            "n_candidates_in_engineering_gate": len(associated),
            "recovered_in_engineering_gate": bool(associated),
            "primary_candidate": None if not associated else associated[0]["row"],
            "primary_frequency_residual_khz": (
                None if not associated else associated[0]["frequency_residual_khz"]
            ),
            "primary_drift_residual_hz_s": (
                None if not associated else associated[0]["drift_residual_hz_s"]
            ),
        }
    axes[0].set_ylabel("candidate − injected drift (Hz s$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.89))
    if artist is not None:
        bar = fig.colorbar(artist, ax=list(axes), fraction=0.025, pad=0.025)
        bar.set_label("BLISS bank S/N")
    fig.suptitle(
        "Blind BLISS recovery around the injected smoke event",
        fontsize=18,
        fontweight="bold",
        color="#17324D",
    )
    fig.text(
        0.5,
        0.025,
        "Post-search truth audit using %s-frequency convention. The dashed gate is diagnostic, not a calibrated Test-B linkage policy."
        % frequency_epoch.replace("_", "-"),
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.subplots_adjust(top=0.78, bottom=0.15, wspace=0.16)
    outputs = save_figure(fig, out_dir / "02_bliss_recovery_near_injection", formats, dpi)
    plt.close(fig)
    return outputs, summary, matches


@dataclass
class H5Window:
    station: str
    raw: np.ndarray
    standardised: np.ndarray
    truth_aligned: np.ndarray
    frequency_offset_khz: np.ndarray
    time_offset_s: np.ndarray
    truth_track_khz: np.ndarray
    recovered_track_khz: Optional[np.ndarray]
    fch1_mhz: float
    foff_mhz: float
    tsamp_s: float
    source_path: Path


def robust_standardise(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values)) or 1.0
    return np.asarray((values - median) / scale, dtype=np.float32)


def truth_dechirp(data: np.ndarray, drift_hz_s: float, dt_s: float, channel_hz: float) -> np.ndarray:
    n_time, n_frequency = data.shape
    reference = 0.5 * (n_time - 1)
    x = np.arange(n_frequency, dtype=float)
    output = np.full_like(data, np.nan, dtype=np.float32)
    for row in range(n_time):
        shift_channels = drift_hz_s * (row - reference) * dt_s / channel_hz
        output[row] = np.interp(
            x + shift_channels,
            x,
            data[row],
            left=np.nan,
            right=np.nan,
        )
    return output


def read_h5_window(
    station: str,
    path: Path,
    truth: Mapping[str, Any],
    match: Optional[Mapping[str, Any]],
    half_width_khz: float,
) -> H5Window:
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for smoke waterfall figures") from exc

    first_mhz = truth_value(truth, ("f_first_MHz",))
    drift_hz_s = truth_value(truth, ("drift_Hz_s", "drift_hz_s"))
    resolved = path.expanduser().resolve()
    with h5py.File(resolved, "r") as handle:
        if "data" not in handle:
            raise ValueError("%s lacks /data" % resolved)
        dataset = handle["data"]
        if dataset.ndim != 3 or dataset.shape[1] != 1:
            raise ValueError("%s /data must have shape (time, 1, frequency)" % resolved)
        fch1_mhz = finite(dataset.attrs["fch1"], "fch1")
        foff_mhz = finite(dataset.attrs["foff"], "foff")
        tsamp_s = finite(dataset.attrs["tsamp"], "tsamp")
        n_time, _, n_channels = dataset.shape
        time_s = np.arange(n_time, dtype=float) * tsamp_s
        midpoint_s = 0.5 * (n_time - 1) * tsamp_s
        truth_mid_mhz = first_mhz + drift_hz_s * midpoint_s / 1e6
        low_mhz = truth_mid_mhz - half_width_khz / 1000.0
        high_mhz = truth_mid_mhz + half_width_khz / 1000.0
        channels = sorted(
            (
                (low_mhz - fch1_mhz) / foff_mhz,
                (high_mhz - fch1_mhz) / foff_mhz,
            )
        )
        start = max(0, int(math.floor(channels[0])))
        stop = min(n_channels, int(math.ceil(channels[1])) + 1)
        if stop - start < 16:
            raise ValueError("requested smoke cutout contains fewer than 16 channels")
        raw = np.asarray(dataset[:, 0, start:stop], dtype=np.float32)
    frequency_mhz = fch1_mhz + np.arange(start, stop, dtype=float) * foff_mhz
    if frequency_mhz[-1] < frequency_mhz[0]:
        frequency_mhz = frequency_mhz[::-1]
        raw = raw[:, ::-1]
    standardised = robust_standardise(raw)
    truth_aligned = truth_dechirp(
        standardised,
        drift_hz_s,
        tsamp_s,
        abs(foff_mhz) * 1e6,
    )
    time_offset = time_s - midpoint_s
    frequency_offset = (frequency_mhz - truth_mid_mhz) * 1000.0
    truth_track = drift_hz_s * time_offset / 1000.0
    recovered_track = None
    if match is not None:
        row = match["row"]
        recovered_first_mhz = finite(row["FREQ_MHZ"], "FREQ_MHZ")
        recovered_drift = finite(row["DR_HZ_S"], "DR_HZ_S")
        recovered_track = (
            (recovered_first_mhz - truth_mid_mhz) * 1000.0
            + recovered_drift * time_s / 1000.0
        )
    return H5Window(
        station=station,
        raw=raw,
        standardised=standardised,
        truth_aligned=truth_aligned,
        frequency_offset_khz=frequency_offset,
        time_offset_s=time_offset,
        truth_track_khz=truth_track,
        recovered_track_khz=recovered_track,
        fch1_mhz=fch1_mhz,
        foff_mhz=foff_mhz,
        tsamp_s=tsamp_s,
        source_path=resolved,
    )


def discover_h5(run_dir: Path, explicit: Mapping[str, Optional[Path]]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    candidates = sorted(run_dir.rglob("*.h5"))
    for station in STATIONS:
        if explicit.get(station) is not None:
            path = explicit[station].expanduser().resolve()  # type: ignore[union-attr]
            if not path.is_file():
                raise FileNotFoundError(path)
            result[station] = path
            continue
        matching = [
            path
            for path in candidates
            if station_from_text(str(path.relative_to(run_dir))) == station
        ]
        chosen = one_file(matching, "%s injected HDF5" % station)
        if chosen is None:
            raise ValueError(
                "could not identify %s HDF5 below --run-dir; supply --%s-h5"
                % (station, station.lower())
            )
        result[station] = chosen
    if result["IRL"] == result["SWE"]:
        raise ValueError("IRL and SWE HDF5 paths must differ")
    return result


def plot_waterfalls(
    windows: Mapping[str, H5Window],
    truths: Mapping[str, Mapping[str, Any]],
    recovery_summary: Optional[Mapping[str, Mapping[str, Any]]],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[List[Path], Dict[str, Any]]:
    plt = configure_matplotlib()
    fig = plt.figure(figsize=(15.6, 8.6))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 0.035, 0.54),
        left=0.06,
        right=0.975,
        bottom=0.12,
        top=0.82,
        hspace=0.24,
        wspace=0.18,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    raw_color_axis = fig.add_subplot(grid[0, 2])
    aligned_color_axis = fig.add_subplot(grid[1, 2])
    info_axis = fig.add_subplot(grid[:, 3])
    display_values = np.concatenate(
        [windows[station].standardised[np.isfinite(windows[station].standardised)] for station in STATIONS]
    )
    low = max(-4.0, float(np.quantile(display_values, 0.01)))
    high = min(12.0, float(np.quantile(display_values, 0.995)))
    if high <= low:
        low, high = -3.0, 6.0
    raw_artist = aligned_artist = None
    for column, station in enumerate(STATIONS):
        window = windows[station]
        extent = (
            float(window.frequency_offset_khz[0]),
            float(window.frequency_offset_khz[-1]),
            float(window.time_offset_s[0] - 0.5 * window.tsamp_s),
            float(window.time_offset_s[-1] + 0.5 * window.tsamp_s),
        )
        raw_artist = axes[0, column].imshow(
            window.standardised,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=low,
            vmax=high,
            interpolation="nearest",
            rasterized=True,
        )
        aligned_artist = axes[1, column].imshow(
            window.truth_aligned,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=low,
            vmax=high,
            interpolation="nearest",
            rasterized=True,
        )
        axes[0, column].plot(
            window.truth_track_khz,
            window.time_offset_s,
            color="#FFDD57",
            lw=1.7,
            label="injected truth",
        )
        if window.recovered_track_khz is not None:
            axes[0, column].plot(
                window.recovered_track_khz,
                window.time_offset_s,
                color="#56CFE1",
                lw=1.3,
                linestyle="--",
                label="nearest BLISS hit",
            )
        axes[1, column].axvline(0, color="#FFDD57", lw=1.4)
        recovered = bool(
            recovery_summary
            and recovery_summary.get(station, {}).get("recovered_in_engineering_gate")
        )
        axes[0, column].set_title(
            "%s · %s" % (station, "recovered in gate" if recovered else "no hit in gate"),
            fontweight="bold",
        )
        axes[1, column].set_xlabel("frequency offset from injected midpoint (kHz)")
    axes[0, 0].set_ylabel("Injected HDF5 copy\ntime from midpoint (s)")
    axes[1, 0].set_ylabel("Post-run truth-aligned QA\ntime from midpoint (s)")
    axes[0, 0].legend(loc="upper right", fontsize=8, framealpha=0.88)
    raw_bar = fig.colorbar(raw_artist, cax=raw_color_axis)
    raw_bar.ax.set_title("Robust\nz", fontsize=8, pad=6)
    raw_bar.ax.yaxis.set_ticks_position("left")
    aligned_bar = fig.colorbar(aligned_artist, cax=aligned_color_axis)
    aligned_bar.ax.set_title("Robust\nz", fontsize=8, pad=6)
    aligned_bar.ax.yaxis.set_ticks_position("left")

    info_axis.axis("off")
    anchor = truths["IRL"]
    profile_key = column_name(anchor, ("profile",), required=False)
    width = truth_value(anchor, ("fwhm_Hz", "fwhm_hz"))
    strength = truth_value(anchor, ("strength",))
    drift = truth_value(anchor, ("drift_Hz_s", "drift_hz_s"))
    frequency = truth_value(anchor, ("f_ref_MHz", "f_first_MHz"))
    lines = [
        "Injected smoke event",
        "profile: %s" % (anchor.get(profile_key, "unknown") if profile_key else "unknown"),
        "width: %.1f Hz" % width,
        "requested strength: %.1f" % strength,
        "signed drift: %+.4f Hz s$^{-1}$" % drift,
        "reference frequency: %.6f MHz" % frequency,
        "",
        "Independent BLISS recovery",
    ]
    for station in STATIONS:
        station_result = None if recovery_summary is None else recovery_summary.get(station)
        if station_result is None:
            lines.append("%s: raw catalogue unavailable" % station)
        elif station_result["recovered_in_engineering_gate"]:
            lines.append(
                "%s: candidate in gate (%d association%s)"
                % (
                    station,
                    station_result["n_candidates_in_engineering_gate"],
                    "s" if station_result["n_candidates_in_engineering_gate"] != 1 else "",
                )
            )
        else:
            lines.append("%s: explicit smoke miss" % station)
    lines.extend(
        [
            "",
            "What this panel establishes",
            "• injected copies contain the intended track",
            "• station geometry and signed drift are visible",
            "• any cyan overlay came from blind BLISS output",
            "",
            "What it does not establish",
            "• completeness, AUC, or false-positive rate",
            "• a calibrated association policy",
            "• performance on astrophysical candidates",
        ]
    )
    info_axis.text(
        0.0,
        0.98,
        "Engineering smoke audit",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#17324D",
    )
    info_axis.text(0.0, 0.91, "\n".join(lines), va="top", fontsize=9.4, linespacing=1.35, color="#263238")
    fig.suptitle(
        "One paired broadened injection through the real IRL–SWE backgrounds",
        fontsize=18,
        fontweight="bold",
        color="#17324D",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        "Top: files searched independently by BLISS. Bottom: truth-aligned post-run QA; injection truth was not supplied to BLISS.",
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.text(
        0.5,
        0.025,
        "Engineering illustration only—one smoke event is not a quantitative Test-B result.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    outputs = save_figure(fig, out_dir / "03_paired_injection_waterfalls", formats, dpi)
    plt.close(fig)
    return outputs, {
        station: {
            "h5_path": str(windows[station].source_path),
            "h5_sha256": sha256_file(windows[station].source_path),
            "fch1_mhz": windows[station].fch1_mhz,
            "signed_foff_hz": windows[station].foff_mhz * 1e6,
            "tsamp_s": windows[station].tsamp_s,
            "cutout_shape": list(windows[station].raw.shape),
        }
        for station in STATIONS
    }


def recovery_file_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False}
    rows = read_csv(path)
    if not rows:
        return {"available": True, "rows": 0, "columns": []}
    station_key = column_name(rows[0], ("station", "station_id", "STATION"), required=False)
    recovered_key = column_name(rows[0], ("recovered", "RECOVERED", "found"), required=False)
    by_station: Dict[str, Dict[str, int]] = {}
    if station_key and recovered_key:
        for station in STATIONS:
            selected = [row for row in rows if station_from_text(row.get(station_key, "")) == station]
            parsed = [parse_bool(row.get(recovered_key)) for row in selected]
            by_station[station] = {
                "rows": len(selected),
                "recovered_rows": sum(value is True for value in parsed),
                "miss_rows": sum(value is False for value in parsed),
            }
    return {
        "available": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "columns": list(rows[0]),
        "by_station": by_station,
    }


def warning_summary(log_paths: Sequence[Path]) -> List[str]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in log_paths
        if path.is_file()
    )
    warnings = []
    if "pkg_resources is deprecated" in text:
        warnings.append("pkg_resources deprecation warning (non-fatal for this run)")
    if "nanobind: type" in text and "already registered" in text:
        warnings.append("nanobind duplicate-registration runtime warnings (non-fatal if the search completes)")
    if "plot_utils could not import `altair`" in text:
        warnings.append("Altair unavailable; optional scatter-matrix plotting disabled")
    return warnings


def write_report(path: Path, metadata: Mapping[str, Any]) -> None:
    search = metadata.get("search_progress", {}).get("searches", [])
    recovery = metadata.get("candidate_recovery", {})
    lines = [
        "# Paired Test-B engineering smoke report",
        "",
        "This report is a post-run engineering audit. It is not the locked Synthetic Test B and does not estimate AUC, recall, completeness, or false-positive rate.",
        "",
        "## Search execution",
        "",
    ]
    if search:
        for item in search:
            lines.append(
                "- Search %d: %d/%d coarse channels recorded; %s; summed coarse-channel candidate count %d."
                % (
                    item["search_number"],
                    item["coarse_channels_recorded"],
                    item["coarse_channels_expected"],
                    "complete" if item["complete"] else "still running or log incomplete",
                    item["candidate_count_sum"],
                )
            )
    else:
        lines.append("- No parseable search-progress log was supplied.")
    lines.extend(["", "## Post-search recovery audit", ""])
    if recovery:
        for station in STATIONS:
            item = recovery.get(station, {})
            lines.append(
                "- %s: %d raw catalogue candidates; %d within the declared engineering gate."
                % (
                    station,
                    int(item.get("n_catalogue_candidates", 0)),
                    int(item.get("n_candidates_in_engineering_gate", 0)),
                )
            )
    else:
        lines.append("- Raw blind candidate catalogues were not yet available.")
    if metadata.get("warnings"):
        lines.extend(["", "## Logged warnings", ""])
        lines.extend("- %s." % item for item in metadata["warnings"])
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "A complete smoke demonstrates that copied paired HDF5 products can be searched independently and that the outputs can be audited against one segregated injected event. Recovery or non-recovery of this single event is descriptive only. A quantitative end-to-end claim requires collision-free calibration/evaluation blocks, deliberate negatives, frozen association/linkage rules, truth-blind Stage-4 inference, and post-inference evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    metadata: Dict[str, Any] = {
        "format_version": 1,
        "analysis_role": "engineering_smoke_visual_audit",
        "scientific_boundary": (
            "One-event post-search diagnostic only; not a quantitative or locked "
            "Synthetic Test-B result."
        ),
        "run_dir": str(run_dir),
    }

    log_paths = [path.expanduser().resolve() for path in args.log]
    if not log_paths:
        log_paths = sorted(
            {
                path.resolve()
                for path in (
                    list(run_dir.rglob("*search*.log"))
                    + list(run_dir.parent.glob("*search*.log"))
                )
                if path.is_file()
            }
        )
    for path in log_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if log_paths:
        searches = []
        for path in log_paths:
            searches.extend(parse_search_log(path))
        if searches:
            outputs, summary = plot_search_progress(searches, out_dir, args.formats, args.dpi)
            created.extend(outputs)
            metadata["search_progress"] = summary
            metadata["search_logs"] = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in log_paths
            ]

    truth_path = args.truth.expanduser().resolve() if args.truth else discover_named(run_dir, "truth.csv")
    truths = None
    matches = None
    recovery_summary = None
    if truth_path is not None:
        truths = truth_index(read_csv(truth_path), args.event_id)
        metadata["truth"] = {"path": str(truth_path), "sha256": sha256_file(truth_path)}
        explicit_catalogues = parse_station_paths(args.candidate_csv, "--candidate-csv")
        catalogues, catalogue_paths = candidate_catalogues(run_dir, explicit_catalogues)
        if any(catalogues.values()):
            outputs, recovery_summary, matches = plot_candidate_recovery(
                catalogues,
                truths,
                args.candidate_frequency_epoch,
                args.assoc_frequency_khz,
                args.assoc_drift_hz_s,
                out_dir,
                args.formats,
                args.dpi,
            )
            created.extend(outputs)
            metadata["candidate_recovery"] = recovery_summary
            metadata["raw_candidate_catalogues"] = [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in catalogue_paths
            ]
        if not args.skip_waterfalls:
            h5_paths = discover_h5(run_dir, {"IRL": args.irl_h5, "SWE": args.swe_h5})
            windows = {}
            for station in STATIONS:
                primary = None if matches is None else matches[station]["primary"]
                windows[station] = read_h5_window(
                    station,
                    h5_paths[station],
                    truths[station],
                    primary,
                    args.waterfall_half_width_khz,
                )
            outputs, h5_summary = plot_waterfalls(
                windows,
                truths,
                recovery_summary,
                out_dir,
                args.formats,
                args.dpi,
            )
            created.extend(outputs)
            metadata["waterfalls"] = h5_summary

    recovery_path = args.recovery.expanduser().resolve() if args.recovery else discover_named(run_dir, "recovery.csv")
    metadata["upstream_recovery_csv"] = recovery_file_summary(recovery_path)
    metadata["warnings"] = warning_summary(log_paths)
    metadata["created_files"] = [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in created
    ]
    report_path = out_dir / "SMOKE_REPORT.md"
    write_report(report_path, metadata)
    created.append(report_path)
    metadata_path = out_dir / "smoke_report.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Created Test-B engineering smoke report:")
    for path in created:
        print("  %s" % path)
    print("  %s" % metadata_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot an existing paired Test-B engineering smoke run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        default=[],
        help="station search log; repeat for separate IRL and SWE logs",
    )
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--recovery", type=Path, default=None)
    parser.add_argument("--irl-h5", type=Path, default=None)
    parser.add_argument("--swe-h5", type=Path, default=None)
    parser.add_argument("--event-id", default=None)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="STATION=PATH",
        help="explicit raw BLISS candidate CSV; repeat once for IRL and SWE",
    )
    parser.add_argument(
        "--candidate-frequency-epoch",
        choices=("first_row", "reference"),
        default="first_row",
        help="epoch represented by raw BLISS FREQ_MHZ; verify against blind_hit_finder.py",
    )
    parser.add_argument("--assoc-frequency-khz", type=float, default=0.3)
    parser.add_argument("--assoc-drift-hz-s", type=float, default=0.007)
    parser.add_argument("--waterfall-half-width-khz", type=float, default=1.5)
    parser.add_argument("--skip-waterfalls", action="store_true")
    parser.add_argument("--formats", type=parse_formats, default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.dpi < 72:
        raise ValueError("dpi must be at least 72")
    if arguments.assoc_frequency_khz <= 0 or arguments.assoc_drift_hz_s <= 0:
        raise ValueError("association gates must be positive")
    if arguments.waterfall_half_width_khz <= 0:
        raise ValueError("waterfall half-width must be positive")
    main(arguments)
