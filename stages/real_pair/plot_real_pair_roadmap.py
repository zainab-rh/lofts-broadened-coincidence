#!/usr/bin/env python3
"""Create a presentation-ready, label-free plot suite for the real pilot.

The script is intentionally stage-aware: it plots every result whose input is
already available and records missing later-stage products as ``pending`` in a
machine-readable inventory.  It never invents labels and never reports AUC,
accuracy, recall, precision, FPR, or end-to-end completeness for real data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from analyze_real_pair import METHODS, clustered_bootstrap_mean, paired_control_rows
from lofts_bliss_schema import (
    load_candidates,
    read_json_records,
    sha256_file,
    write_json,
)

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lofts-matplotlib-cache")
)


SCIENTIFIC_BOUNDARY = (
    "These are descriptive diagnostics for an unlabeled, real, simultaneous, "
    "barycentric two-station pilot. They are not real-data classification "
    "performance, technosignature validation, or end-to-end completeness."
)

CONTROL_DESIGN_DESCRIPTION = (
    "Registered distant-frequency shifts plus documented score-blind "
    "geometry fallback."
)


def existing(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    target = Path(path).expanduser().resolve()
    return target if target.is_file() else None


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object in %s" % path)
    return value


def finite(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def shorten(value: Any, maximum: int = 28) -> str:
    text = str(value if value not in (None, "") else "unspecified")
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


class PlotSuite:
    def __init__(self, out_dir: Path, formats: Sequence[str], dpi: int):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.out_dir = out_dir
        self.formats = tuple(formats)
        self.dpi = dpi
        self.entries: List[Dict[str, Any]] = []
        self.pngs: List[Tuple[Path, str]] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        fig,
        plot_id: str,
        title: str,
        interpretation: str,
        inputs: Sequence[str],
    ) -> None:
        outputs = []
        for suffix in self.formats:
            path = self.out_dir / (plot_id + "." + suffix)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            outputs.append(
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(str(path)),
                    "format": suffix,
                }
            )
            if suffix == "png":
                self.pngs.append((path, title))
        self.plt.close(fig)
        self.entries.append(
            {
                "plot_id": plot_id,
                "title": title,
                "status": "generated",
                "interpretation": interpretation,
                "inputs": list(inputs),
                "outputs": outputs,
            }
        )

    def skip(
        self, plot_id: str, title: str, reason: str, inputs: Sequence[str]
    ) -> None:
        self.entries.append(
            {
                "plot_id": plot_id,
                "title": title,
                "status": "pending",
                "reason": reason,
                "inputs": list(inputs),
                "outputs": [],
            }
        )

    def contact_sheet(self) -> None:
        if not self.pngs:
            return
        columns = 3
        rows = int(math.ceil(len(self.pngs) / columns))
        fig, axes = self.plt.subplots(
            rows, columns, figsize=(15, 4.2 * rows), squeeze=False
        )
        for axis, item in zip(axes.ravel(), self.pngs):
            path, title = item
            axis.imshow(self.plt.imread(path))
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        for axis in axes.ravel()[len(self.pngs) :]:
            axis.axis("off")
        fig.suptitle(
            "LOFTS0050 real-pair roadmap: generated diagnostic figures", fontsize=15
        )
        fig.tight_layout()
        path = self.out_dir / "00_roadmap_contact_sheet.png"
        fig.savefig(path, dpi=min(self.dpi, 180), bbox_inches="tight")
        self.plt.close(fig)
        self.entries.insert(
            0,
            {
                "plot_id": "00_roadmap_contact_sheet",
                "title": "Roadmap plot contact sheet",
                "status": "generated",
                "interpretation": "Visual index only; use the full-resolution figures for interpretation.",
                "inputs": [str(item[0].resolve()) for item in self.pngs],
                "outputs": [
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256_file(str(path)),
                        "format": "png",
                    }
                ],
            },
        )


def plot_observation_geometry(
    suite: PlotSuite, summary: Mapping[str, Any], source: Path
) -> None:
    observations = list(summary.get("observations", []))
    if len(observations) < 2:
        suite.skip(
            "01_observation_geometry",
            "Barycentric frequency/time overlap",
            "at least two observations are required",
            [str(source)],
        )
        return
    stations = [str(item["station_id"]) for item in observations]
    bands = []
    times = []
    minimum_start = min(float(item["start_mjd"]) for item in observations)
    for item in observations:
        f0 = float(item["fch1_hz"])
        f1 = f0 + (int(item["n_channels"]) - 1) * float(item["signed_foff_hz"])
        bands.append((min(f0, f1) / 1e6, max(f0, f1) / 1e6))
        start = (float(item["start_mjd"]) - minimum_start) * 86400.0
        duration = int(item["n_time"]) * float(item["tsamp_s"])
        times.append((start, duration))

    plt = suite.plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colors = plt.cm.Set2(np.linspace(0.1, 0.85, len(stations)))
    for index, (station, band, timing, color) in enumerate(
        zip(stations, bands, times, colors)
    ):
        axes[0].barh(index, band[1] - band[0], left=band[0], color=color, alpha=0.85)
        axes[1].barh(index, timing[1], left=timing[0], color=color, alpha=0.85)
    axes[0].set_yticks(range(len(stations)), stations)
    axes[1].set_yticks(range(len(stations)), stations)
    axes[0].set_xlabel("Barycentric header frequency (MHz)")
    axes[1].set_xlabel("Seconds relative to earliest tstart")
    axes[0].set_title("Physical-frequency coverage")
    axes[1].set_title("Integration-time coverage")
    geometry = next(iter(summary.get("group_geometry", {}).values()), {})
    low = finite(geometry.get("common_frequency_low_hz"))
    high = finite(geometry.get("common_frequency_high_hz"))
    if low is not None and high is not None:
        axes[0].axvspan(
            low / 1e6, high / 1e6, color="black", alpha=0.10, label="common band"
        )
        axes[0].legend(loc="best")
    overlap_start = finite(geometry.get("time_overlap_start_mjd"))
    overlap_end = finite(geometry.get("integration_overlap_end_mjd_exclusive"))
    if overlap_start is not None and overlap_end is not None:
        left = (overlap_start - minimum_start) * 86400.0
        right = (overlap_end - minimum_start) * 86400.0
        axes[1].axvspan(
            left, right, color="black", alpha=0.10, label="common integrations"
        )
        axes[1].legend(loc="best")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "LOFTS0050 Ireland–Sweden input geometry (headers declare barycentric=1)"
    )
    fig.tight_layout()
    suite.save(
        fig,
        "01_observation_geometry",
        "Barycentric frequency/time overlap",
        "Confirms that station matching uses overlapping physical frequency and absolute MJD, not channel index.",
        [str(source)],
    )


def plot_bank_coverage(
    suite: PlotSuite, audit: Mapping[str, Any], source: Path
) -> None:
    stations = audit.get("stations", {})
    targets = [float(value) for value in audit.get("target_widths_hz", [])]
    if not stations or not targets:
        suite.skip(
            "02_width_bank_coverage",
            "Width-bank coverage",
            "bank audit is empty",
            [str(source)],
        )
        return
    plt = suite.plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(stations)))
    for offset, ((station, values), color) in enumerate(
        zip(sorted(stations.items()), colors)
    ):
        bank = [float(value) for value in values["native_bank_hz"]]
        x = np.arange(len(bank)) + (offset - (len(stations) - 1) / 2) * 0.12
        axes[0].plot(x, bank, "o-", color=color, label=station)
        mismatch = [
            float(item["multiplicative_mismatch"]) for item in values["targets"]
        ]
        axes[1].plot(targets, mismatch, "o-", color=color, label=station)
    axes[0].axhspan(
        10.0, 100.0, color="tab:green", alpha=0.10, label="Stage-4 nominal 10–100 Hz"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Native bank template index")
    axes[0].set_ylabel("Nominal Lorentzian FWHM (Hz)")
    axes[0].set_title("Locked native BLISS bank")
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xlabel("Target width (Hz)")
    axes[1].set_ylabel("Nearest-template multiplicative mismatch")
    axes[1].set_title("Physical-width mismatch")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Width-bank geometry audit — proposal only, not permission to change the locked run"
    )
    fig.tight_layout()
    suite.save(
        fig,
        "02_width_bank_coverage",
        "Width-bank coverage",
        "Shows how the six native templates map into physical Hz at each station and why a new bank would require recalibration.",
        [str(source)],
    )


def catalog_arrays(catalog_paths: Sequence[Path]):
    by_station = {}
    for path in catalog_paths:
        records = load_candidates(str(path))
        stations = {item.station_id for item in records}
        if len(stations) != 1:
            raise ValueError(
                "canonical catalog must contain exactly one station: %s" % path
            )
        station = next(iter(stations))
        by_station[station] = (path, records)
    return by_station


def plot_catalog_composition(suite: PlotSuite, catalogs) -> None:
    if not catalogs:
        suite.skip(
            "03_catalog_composition",
            "Raw-catalog composition",
            "canonical catalogs are pending",
            [],
        )
        return
    plt = suite.plt
    stations = sorted(catalogs)
    widths = sorted(
        {
            int(item.extras.get("native_width_channels", 0))
            for station in stations
            for item in catalogs[station][1]
        }
    )
    flags = sorted(
        {
            str(item.extras.get("source_flag") or "unflagged")
            for station in stations
            for item in catalogs[station][1]
        }
    )
    flag_totals = Counter(
        str(item.extras.get("source_flag") or "unflagged")
        for station in stations
        for item in catalogs[station][1]
    )
    flags = sorted(flags, key=lambda value: (-flag_totals[value], value))[:10]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    x = np.arange(len(widths))
    width = 0.8 / len(stations)
    for index, station in enumerate(stations):
        records = catalogs[station][1]
        counts = Counter(
            int(item.extras.get("native_width_channels", 0)) for item in records
        )
        axes[0].bar(
            x + (index - (len(stations) - 1) / 2) * width,
            [counts[item] for item in widths],
            width,
            label=station,
        )
        flag_counts = Counter(
            str(item.extras.get("source_flag") or "unflagged") for item in records
        )
        axes[1].barh(
            np.arange(len(flags)) + (index - (len(stations) - 1) / 2) * width,
            [flag_counts[item] for item in flags],
            height=width,
            label=station,
        )
    axes[0].set_xticks(x, [str(item) for item in widths])
    axes[0].set_xlabel("Winning native template (channels)")
    axes[0].set_ylabel("Uncollapsed candidate count")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("Template occupancy")
    axes[1].set_yticks(np.arange(len(flags)), [shorten(item) for item in flags])
    axes[1].set_xlabel("Uncollapsed candidate count")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_title("Retained source flags")
    fractions = []
    for station in stations:
        records = catalogs[station][1]
        fractions.append(
            np.mean(
                [bool(item.extras.get("broadband_rfi_like", False)) for item in records]
            )
        )
    axes[2].bar(stations, np.asarray(fractions) * 100.0)
    axes[2].set_ylabel("Broadband-RFI-like diagnostic (%)")
    axes[2].set_title("Metadata diagnostic (not exclusion)")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Naoise raw-catalog diagnostics; FLAG is retained as metadata")
    fig.tight_layout()
    suite.save(
        fig,
        "03_catalog_composition",
        "Raw-catalog composition",
        "Displays search-output composition without converting source flags into truth labels or hard exclusions.",
        [str(catalogs[item][0]) for item in stations],
    )


def deterministic_sample(records: Sequence[Any], maximum: int) -> Sequence[Any]:
    if len(records) <= maximum:
        return records
    indices = np.linspace(0, len(records) - 1, maximum, dtype=int)
    return [records[int(index)] for index in indices]


def plot_frequency_drift(suite: PlotSuite, catalogs, maximum: int) -> None:
    if not catalogs:
        suite.skip(
            "04_candidate_frequency_drift",
            "Frequency–drift candidate map",
            "canonical catalogs are pending",
            [],
        )
        return
    plt = suite.plt
    stations = sorted(catalogs)
    fig, axes = plt.subplots(
        len(stations), 1, figsize=(13, 4.2 * len(stations)), squeeze=False, sharex=True
    )
    for axis, station in zip(axes.ravel(), stations):
        records = deterministic_sample(catalogs[station][1], maximum)
        frequencies = np.asarray([item.frequency_hz / 1e6 for item in records])
        drifts = np.asarray([item.drift_hz_s for item in records])
        snr = np.asarray([max(float(item.snr), 1e-12) for item in records])
        scatter = axis.scatter(
            frequencies,
            drifts,
            c=np.log10(snr),
            s=5,
            alpha=0.45,
            cmap="viridis",
            rasterized=True,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel("Drift (Hz s$^{-1}$)")
        axis.set_title(
            "%s: %d of %d candidates shown"
            % (station, len(records), len(catalogs[station][1]))
        )
        axis.grid(alpha=0.15)
        fig.colorbar(scatter, ax=axis, label="log$_{10}$(bank S/N)")
    axes[-1, 0].set_xlabel("Barycentric reference frequency (MHz)")
    fig.suptitle(
        "Blind-search candidate distribution in physical frequency and signed drift"
    )
    fig.tight_layout()
    suite.save(
        fig,
        "04_candidate_frequency_drift",
        "Frequency–drift candidate map",
        "Reveals band structure, zero-drift concentration, and station differences while matching in physical coordinates.",
        [str(catalogs[item][0]) for item in stations],
    )


def plot_snr_diagnostics(suite: PlotSuite, catalogs, maximum: int) -> None:
    rows = []
    for station, (_, records) in catalogs.items():
        for item in deterministic_sample(records, maximum):
            bank = finite(item.extras.get("bank_snr", item.snr))
            standard = finite(item.extras.get("standard_snr"))
            ratio = finite(item.extras.get("bank_standard_ratio"))
            if bank is not None and standard is not None and standard > 0:
                rows.append((station, bank, standard, ratio))
    if not rows:
        suite.skip(
            "05_candidate_snr_diagnostics",
            "Bank-versus-standard S/N",
            "canonical catalogs lack comparable S/N values",
            [],
        )
        return
    plt = suite.plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for station in sorted(catalogs):
        selected = [item for item in rows if item[0] == station]
        if not selected:
            continue
        bank = np.asarray([item[1] for item in selected])
        standard = np.asarray([item[2] for item in selected])
        axes[0].scatter(standard, bank, s=7, alpha=0.35, label=station, rasterized=True)
        ratios = [item[3] for item in selected if item[3] is not None and item[3] > 0]
        if ratios:
            axes[1].hist(np.log10(ratios), bins=35, alpha=0.45, label=station)
    limits = [
        max(min(item[2] for item in rows), 1e-3),
        max(max(max(item[1], item[2]) for item in rows), 1.0),
    ]
    axes[0].plot(limits, limits, "k--", linewidth=1, label="equal")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Standard S/N")
    axes[0].set_ylabel("Winning bank S/N")
    axes[0].set_title("Matched-bank gain")
    axes[1].axvline(
        math.log10(8.0),
        color="black",
        linestyle="--",
        linewidth=1,
        label="ratio 8 diagnostic",
    )
    axes[1].set_xlabel("log$_{10}$(bank S/N ÷ standard S/N)")
    axes[1].set_ylabel("Candidates")
    axes[1].set_title("Retained broadband-RFI ratio diagnostic")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Search-response diagnostics; not a labeled signal/RFI classifier")
    fig.tight_layout()
    suite.save(
        fig,
        "05_candidate_snr_diagnostics",
        "Bank-versus-standard S/N",
        "Checks how broadened-template response differs from the standard statistic and visualises the retained ratio metadata.",
        [str(catalogs[item][0]) for item in sorted(catalogs)],
    )


def plot_union(
    suite: PlotSuite, union: Sequence[Mapping[str, Any]], source: Optional[Path]
) -> None:
    if not union:
        suite.skip(
            "06_union_outcomes",
            "Two-station union outcomes",
            "candidate union is pending",
            [] if source is None else [str(source)],
        )
        return
    plt = suite.plt
    definitions = (
        ("detection_state", "Detection state"),
        ("route", "Width/coverage route"),
        ("operational_eligibility", "Operational eligibility"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for axis, (key, label) in zip(axes, definitions):
        counts = Counter(shorten(item.get(key)) for item in union)
        names = [item for item, _ in counts.most_common()]
        values = [counts[item] for item in names]
        axis.barh(range(len(names)), values)
        axis.set_yticks(range(len(names)), names)
        axis.invert_yaxis()
        axis.set_xlabel("Union entries")
        axis.set_title(label)
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Sparse physical-coordinate Ireland ∪ Sweden candidate union")
    fig.tight_layout()
    suite.save(
        fig,
        "06_union_outcomes",
        "Two-station union outcomes",
        "Separates two-station matches, one-station candidates, rolloff/unsearched coverage, and routing decisions.",
        [] if source is None else [str(source)],
    )


def plot_association(
    suite: PlotSuite, union: Sequence[Mapping[str, Any]], source: Optional[Path]
) -> None:
    associated = [item["association"] for item in union if item.get("association")]
    if not associated:
        suite.skip(
            "07_association_residuals",
            "Frozen-policy association residuals",
            "no associated two-station entries are available",
            [] if source is None else [str(source)],
        )
        return
    plt = suite.plt
    fields = (
        ("frequency_delta_hz", "Frequency difference (Hz)"),
        ("drift_delta_hz_s", "Drift difference (Hz s$^{-1}$)"),
        ("log_width_delta", "Absolute log-width difference"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for axis, (field, label) in zip(axes, fields):
        values = [float(item[field]) for item in associated]
        axis.hist(values, bins=min(40, max(5, int(math.sqrt(len(values))))), alpha=0.85)
        axis.set_xlabel(label)
        axis.set_ylabel("Associated entries")
        axis.grid(alpha=0.2)
    fig.suptitle("Association residuals under the policy frozen before union/scoring")
    fig.tight_layout()
    suite.save(
        fig,
        "07_association_residuals",
        "Frozen-policy association residuals",
        "Audits the physical-coordinate separations admitted by the preregistered association gate.",
        [] if source is None else [str(source)],
    )


def plot_observed_controls(
    suite: PlotSuite,
    primary: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    sources: Sequence[Path],
) -> None:
    if not primary or not controls:
        suite.skip(
            "08_observed_vs_controls",
            "Observed versus shifted-control scores",
            "primary or control inference is pending",
            [str(item) for item in sources],
        )
        return
    plt = suite.plt
    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(4.2 * len(METHODS), 4.8), squeeze=False
    )
    for axis, (method, label) in zip(axes[0], METHODS.items()):
        axis.boxplot(
            [
                [float(item[method]) for item in primary],
                [float(item[method]) for item in controls],
            ],
            showfliers=False,
        )
        axis.set_xticks([1, 2], ["Observed", "Shifted"])
        axis.set_title(label)
        axis.set_ylabel("Ranking score")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Score distributions at observed and distant-frequency locations",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.015,
        CONTROL_DESIGN_DESCRIPTION
        + " Controls are comparison locations, not negative truth labels.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    suite.save(
        fig,
        "08_observed_vs_controls",
        "Observed and distant-frequency score distributions",
        CONTROL_DESIGN_DESCRIPTION
        + " The comparison is descriptive because the real pilot is unlabeled.",
        [str(item) for item in sources],
    )


def paired_rows_safe(primary, controls):
    if not primary or not controls:
        return []
    rows, _ = paired_control_rows(primary, controls)
    return rows


def plot_paired_deltas(
    suite: PlotSuite,
    paired: Sequence[Mapping[str, Any]],
    sources: Sequence[Path],
    n_boot: int,
    seed: int,
) -> None:
    if not paired:
        suite.skip(
            "09_paired_score_deltas",
            "Within-candidate paired score deltas",
            "no primary rows have valid shifted controls",
            [str(item) for item in sources],
        )
        return
    plt = suite.plt
    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(4.2 * len(METHODS), 4.8), squeeze=False
    )
    for index, (axis, (method, label)) in enumerate(zip(axes[0], METHODS.items())):
        values = [float(item[method + "_delta"]) for item in paired]
        blocks = [
            str(item.get("resampling_block_id") or item["pair_id"]) for item in paired
        ]
        stats = clustered_bootstrap_mean(
            values, blocks, n_boot=n_boot, seed=seed + index
        )
        axis.hist(values, bins=min(35, max(5, int(math.sqrt(len(values))))), alpha=0.85)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.axvline(
            stats["mean"], color="tab:red", linewidth=1.5, label="Mean contrast"
        )
        ci = stats.get("ci95", [None, None])
        if ci[0] is not None:
            axis.axvspan(
                ci[0],
                ci[1],
                color="tab:red",
                alpha=0.12,
                label="95% block-bootstrap CI",
            )
        axis.set_title(label)
        axis.set_xlabel("Observed − mean control")
        axis.set_ylabel("Candidate locations")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Within-candidate score contrasts relative to distant-frequency controls",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.015,
        CONTROL_DESIGN_DESCRIPTION
        + " Uncertainty resamples common physical-frequency blocks.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    suite.save(
        fig,
        "09_paired_score_deltas",
        "Within-candidate score contrasts",
        CONTROL_DESIGN_DESCRIPTION
        + " Observed-minus-control contrasts use physical-frequency-block uncertainty and do not estimate classification accuracy.",
        [str(item) for item in sources],
    )


def plot_score_correlations(suite: PlotSuite, primary, source: Optional[Path]) -> None:
    if len(primary) < 2:
        suite.skip(
            "10_score_correlations",
            "Method score correlations",
            "at least two primary scored pairs are required",
            [] if source is None else [str(source)],
        )
        return
    matrix = np.asarray([[float(item[key]) for key in METHODS] for item in primary])
    correlation = np.corrcoef(matrix, rowvar=False)
    plt = suite.plt
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [METHODS[key] for key in METHODS]
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlation[row, column]
            axis.text(
                column,
                row,
                "%.2f" % value,
                ha="center",
                va="center",
                color="white" if abs(value) > 0.55 else "black",
            )
    fig.colorbar(image, ax=axis, label="Pearson correlation across scored candidates")
    axis.set_title("Agreement among ranking statistics (not accuracy)")
    fig.tight_layout()
    suite.save(
        fig,
        "10_score_correlations",
        "Method score correlations",
        "Shows whether Stage 4 rankings track or diverge from Stage 3 and the transparent matched-filter baseline.",
        [] if source is None else [str(source)],
    )


def plot_stage4_context(suite: PlotSuite, primary, source: Optional[Path]) -> None:
    valid = [
        item
        for item in primary
        if finite(item.get("anchor_width_hz")) is not None
        and finite(item.get("anchor_snr")) is not None
    ]
    if not valid:
        suite.skip(
            "11_stage4_score_context",
            "Stage-4 score context",
            "primary score context is pending",
            [] if source is None else [str(source)],
        )
        return
    plt = suite.plt
    states = sorted({str(item.get("detection_state")) for item in valid})
    colors = {state: plt.cm.tab10(index % 10) for index, state in enumerate(states)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for state in states:
        selected = [item for item in valid if str(item.get("detection_state")) == state]
        score = [float(item["stage4_score"]) for item in selected]
        axes[0].scatter(
            [float(item["anchor_width_hz"]) for item in selected],
            score,
            s=14,
            alpha=0.55,
            color=colors[state],
            label=state,
        )
        axes[1].scatter(
            [float(item["anchor_snr"]) for item in selected],
            score,
            s=14,
            alpha=0.55,
            color=colors[state],
            label=state,
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Anchor candidate width (Hz)")
    axes[1].set_xlabel("Anchor candidate bank S/N")
    for axis in axes:
        axis.set_ylabel("Stage-4 ranking score")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_title("Score versus reported width")
    axes[1].set_title("Score versus search S/N")
    fig.suptitle("Stage-4 ranking context; score is not an astrophysical posterior")
    fig.tight_layout()
    suite.save(
        fig,
        "11_stage4_score_context",
        "Stage-4 score context",
        "Checks whether rankings are dominated by candidate width, search S/N, or detection-state strata.",
        [] if source is None else [str(source)],
    )


def plot_top_candidates(
    suite: PlotSuite, primary, source: Optional[Path], top_n: int
) -> None:
    if not primary:
        suite.skip(
            "12_top_candidate_scores",
            "Top-candidate score profile",
            "primary inference is pending",
            [] if source is None else [str(source)],
        )
        return
    selected = sorted(
        primary, key=lambda item: (-float(item["stage4_score"]), str(item["pair_id"]))
    )[:top_n]
    selected = list(reversed(selected))
    plt = suite.plt
    fig, axis = plt.subplots(figsize=(12, max(5.0, 0.30 * len(selected))))
    y = np.arange(len(selected))
    offsets = np.linspace(-0.24, 0.24, len(METHODS))
    for offset, (method, label) in zip(offsets, METHODS.items()):
        values = np.asarray([float(item[method]) for item in selected])
        # Each method has different units/scale. Robust z-scores show profile
        # agreement without pretending the raw scores are directly calibrated.
        median = float(np.median(values))
        scale = float(np.median(np.abs(values - median)))
        if scale <= 0:
            scale = float(np.std(values)) or 1.0
        z = (values - median) / scale
        axis.scatter(z, y + offset, s=18, label=label)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(y, [shorten(item["pair_id"], 36) for item in selected], fontsize=7)
    axis.set_xlabel("Within-top-list robust standardised score")
    axis.set_title(
        "Top %d Stage-4-ranked candidates: multi-method score profiles" % len(selected)
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    suite.save(
        fig,
        "12_top_candidate_scores",
        "Top-candidate score profile",
        "Supports candidate review by showing whether high Stage-4 ranks are corroborated by other ranking statistics.",
        [] if source is None else [str(source)],
    )


def plot_processing_counts(
    suite: PlotSuite, catalogs, union, primary, controls, paired, inputs: Sequence[str]
) -> None:
    values = []
    labels = []
    if catalogs:
        labels.append("Raw candidates\n(sum of stations)")
        values.append(sum(len(item[1]) for item in catalogs.values()))
    if union:
        labels.append("Union entries")
        values.append(len(union))
    if primary:
        labels.append("Primary scored")
        values.append(len(primary))
    if controls:
        labels.append("Control scores")
        values.append(len(controls))
    paired_with_two_controls = [
        item for item in paired if int(item.get("n_controls", 0)) >= 2
    ]
    if paired:
        labels.append("Primary with ≥2\nvalid controls")
        values.append(len(paired_with_two_controls))
    if not values:
        suite.skip(
            "13_processing_counts",
            "Pipeline processing counts",
            "no downstream products are available",
            inputs,
        )
        return
    plt = suite.plt
    fig, axis = plt.subplots(figsize=(11, 5.2))
    bars = axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Record count")
    if max(values) / max(min(values), 1) > 50:
        axis.set_yscale("log")
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )
    axis.grid(axis="y", alpha=0.2)
    axis.set_title(
        "Analysis sample sizes across processing stages",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.015,
        "Bars represent distinct analysis units and denominators; they do not form a detection-efficiency funnel.",
        ha="center",
        fontsize=9,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
    suite.save(
        fig,
        "13_processing_counts",
        "Analysis sample sizes",
        "Tracks distinct analysis units across stages; the final bar is computed from candidates with at least two valid controls.",
        inputs,
    )


def main(args: argparse.Namespace) -> None:
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError("unsupported plot formats: %s" % unsupported)
    if not formats:
        raise ValueError("at least one plot format is required")
    out_dir = Path(args.out_dir).expanduser().resolve()
    suite = PlotSuite(out_dir, formats, args.dpi)

    observations_path = existing(args.observations_summary)
    bank_path = existing(args.bank_audit)
    candidate_paths = [
        item for item in (existing(path) for path in args.candidate_files) if item
    ]
    union_path = existing(args.union)
    primary_path = existing(args.primary_predictions)
    control_path = existing(args.control_predictions)

    input_inventory = []
    named_paths = {
        "observations_summary": observations_path,
        "bank_audit": bank_path,
        "union": union_path,
        "primary_predictions": primary_path,
        "control_predictions": control_path,
    }
    for name, path in named_paths.items():
        requested = getattr(args, name)
        input_inventory.append(
            {
                "name": name,
                "requested_path": requested,
                "available": path is not None,
                "resolved_path": None if path is None else str(path),
                "sha256": None if path is None else sha256_file(str(path)),
            }
        )
    for index, requested in enumerate(args.candidate_files):
        path = existing(requested)
        input_inventory.append(
            {
                "name": "candidate_file_%d" % (index + 1),
                "requested_path": requested,
                "available": path is not None,
                "resolved_path": None if path is None else str(path),
                "sha256": None if path is None else sha256_file(str(path)),
            }
        )

    observations = None if observations_path is None else load_json(observations_path)
    bank = None if bank_path is None else load_json(bank_path)
    catalogs = catalog_arrays(candidate_paths) if candidate_paths else {}
    union = read_json_records(str(union_path)) if union_path else []
    primary = read_json_records(str(primary_path)) if primary_path else []
    controls = read_json_records(str(control_path)) if control_path else []
    paired = paired_rows_safe(primary, controls)

    if observations is None:
        suite.skip(
            "01_observation_geometry",
            "Barycentric frequency/time overlap",
            "observation summary is pending",
            [args.observations_summary] if args.observations_summary else [],
        )
    else:
        plot_observation_geometry(suite, observations, observations_path)
    if bank is None:
        suite.skip(
            "02_width_bank_coverage",
            "Width-bank coverage",
            "bank audit is pending",
            [args.bank_audit] if args.bank_audit else [],
        )
    else:
        plot_bank_coverage(suite, bank, bank_path)
    plot_catalog_composition(suite, catalogs)
    plot_frequency_drift(suite, catalogs, args.max_scatter)
    plot_snr_diagnostics(suite, catalogs, args.max_scatter)
    plot_union(suite, union, union_path)
    plot_association(suite, union, union_path)
    prediction_sources = [item for item in (primary_path, control_path) if item]
    plot_observed_controls(suite, primary, controls, prediction_sources)
    plot_paired_deltas(suite, paired, prediction_sources, args.n_boot, args.seed)
    plot_score_correlations(suite, primary, primary_path)
    plot_stage4_context(suite, primary, primary_path)
    plot_top_candidates(suite, primary, primary_path, args.top_n)
    all_sources = [str(path) for path in candidate_paths]
    all_sources += [
        str(item) for item in (union_path, primary_path, control_path) if item
    ]
    plot_processing_counts(
        suite, catalogs, union, primary, controls, paired, all_sources
    )
    suite.contact_sheet()

    generated = sum(item["status"] == "generated" for item in suite.entries)
    pending = sum(item["status"] == "pending" for item in suite.entries)
    inventory = {
        "format_version": 1,
        "dataset_role": "unlabeled_real_barycentric_pair",
        "scientific_boundary": SCIENTIFIC_BOUNDARY,
        "forbidden_claims": [
            "real-data AUC, accuracy, recall, precision, or false-positive rate",
            "end-to-end BLISS completeness without blind injection truth",
            "astrophysical posterior probability from a Stage-4 ranking score",
            "technosignature confirmation from two-station score agreement",
        ],
        "n_generated": generated,
        "n_pending": pending,
        "inputs": input_inventory,
        "plots": suite.entries,
    }
    write_json(str(out_dir / "roadmap_plot_inventory.json"), inventory)
    print(
        "Generated %d roadmap plots/contact sheets; %d remain pending. Inventory: %s"
        % (generated, pending, out_dir / "roadmap_plot_inventory.json")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot every available stage of the unlabeled real-pair roadmap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations-summary")
    parser.add_argument("--bank-audit")
    parser.add_argument("--candidate-files", nargs="*", default=[])
    parser.add_argument("--union")
    parser.add_argument("--primary-predictions")
    parser.add_argument("--control-predictions")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-scatter", type=int, default=50000)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
