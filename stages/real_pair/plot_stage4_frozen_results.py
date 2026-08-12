#!/usr/bin/env python3
"""Create publication figures from a frozen LOFTS Stage-4 evaluation.

This module is deliberately post-hoc and read-only with respect to the model
evaluation.  It consumes the machine-readable files written by
``evaluate_stage4.py`` and never reloads a checkpoint, generates examples, or
changes a threshold.  Derived summaries are labelled as such.

The script is compatible with Python 3.9 and requires only NumPy and
Matplotlib in addition to the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

# Shared systems often expose a read-only home directory.  Keep Matplotlib's
# cache in a writable location unless the caller has selected one explicitly.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lofts-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

SCRIPT_VERSION = "1.1.0"

METHODS = (
    "stage3_raw",
    "stage3_corrected",
    "matched_filter",
    "stage4",
)
METHOD_LABELS = {
    "stage3_raw": "Raw Stage 3",
    "stage3_corrected": "Corrected Stage 3",
    "matched_filter": "Transparent filter",
    "stage4": "Stage 4",
}
METHOD_COLORS = {
    "stage3_raw": "#4C78A8",
    "stage3_corrected": "#F58518",
    "matched_filter": "#54A24B",
    "stage4": "#E45756",
}
REGIME_LABELS = {
    "detected": "Detected-conditioned",
    "power": "Fixed-power",
}
REGIME_COLORS = {
    "detected": "#087E8B",
    "power": "#D17C00",
}
SHAPE_ORDER = ("lorentzian", "gaussian", "box")
SHAPE_LABELS = {
    "lorentzian": "Lorentzian",
    "gaussian": "Gaussian",
    "box": "Box",
}

NUMERIC_COLUMNS = {
    "width_hz",
    "n_match",
    "n_mismatch",
}
for _method in METHODS:
    NUMERIC_COLUMNS.update(
        {
            "%s_auc" % _method,
            "%s_auc_ci_lo" % _method,
            "%s_auc_ci_hi" % _method,
            "%s_average_precision" % _method,
        }
    )
for _method in ("stage3_corrected", "matched_filter", "stage4"):
    NUMERIC_COLUMNS.update(
        {
            "delta_%s_minus_raw" % _method,
            "delta_%s_minus_raw_ci_lo" % _method,
            "delta_%s_minus_raw_ci_hi" % _method,
        }
    )
for _method in ("stage3_raw", "stage4"):
    for _metric in (
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "threshold",
    ):
        NUMERIC_COLUMNS.add("%s_%s" % (_method, _metric))

REQUIRED_COLUMNS = {
    "shape",
    "snr_mode",
    "width_hz",
    "n_match",
    "n_mismatch",
    "stage3_raw_auc",
    "stage4_auc",
    "matched_filter_auc",
    "stage3_corrected_auc",
    "delta_stage4_minus_raw",
    "delta_stage4_minus_raw_ci_lo",
    "delta_stage4_minus_raw_ci_hi",
    "stage3_raw_tp",
    "stage3_raw_fp",
    "stage3_raw_fn",
    "stage3_raw_tn",
    "stage4_tp",
    "stage4_fp",
    "stage4_fn",
    "stage4_tn",
    "stage4_f1",
}


def configure_style() -> None:
    """Use a restrained, journal-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_formats(raw: str) -> Tuple[str, ...]:
    formats = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    allowed = {"png", "pdf", "svg"}
    if not formats:
        raise argparse.ArgumentTypeError("at least one output format is required")
    invalid = sorted(set(formats) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(
            "unsupported format(s): %s; choose png,pdf,svg" % ", ".join(invalid)
        )
    return formats


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s is not numeric: %r" % (label, value)) from exc
    if not math.isfinite(result):
        raise ValueError("%s is not finite: %r" % (label, value))
    return result


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: %s" % path)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required columns: %s" % ", ".join(missing))
        rows: List[Dict[str, object]] = []
        for row_index, raw_row in enumerate(reader, start=2):
            row: Dict[str, object] = dict(raw_row)
            for key in NUMERIC_COLUMNS.intersection(row):
                raw_value = row[key]
                if raw_value in (None, ""):
                    continue
                row[key] = finite_float(
                    raw_value, "row %d column %s" % (row_index, key)
                )
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no data rows: %s" % path)
    return rows


def load_evaluation(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("evaluation JSON must contain an object")
    for key in ("primary_detected_conditioned", "secondary_fixed_power"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError("evaluation JSON is missing object %r" % key)
    return data


def validate_inputs(
    rows: Sequence[Mapping[str, object]],
    evaluation: Mapping[str, object],
) -> None:
    seen = set()
    for row in rows:
        regime = str(row["snr_mode"])
        shape = str(row["shape"])
        width = float(row["width_hz"])
        key = (regime, shape, width)
        if key in seen:
            raise ValueError("duplicate CSV cell: %r" % (key,))
        seen.add(key)
        if regime not in REGIME_LABELS:
            raise ValueError("unsupported S/N mode: %r" % regime)
        if shape not in SHAPE_LABELS:
            raise ValueError("unsupported profile shape: %r" % shape)
        if width <= 0:
            raise ValueError("width_hz must be positive: %r" % width)
        for method in METHODS:
            auc = float(row["%s_auc" % method])
            lo = float(row["%s_auc_ci_lo" % method])
            hi = float(row["%s_auc_ci_hi" % method])
            if not (0.0 <= lo <= auc <= hi <= 1.0):
                raise ValueError(
                    "invalid AUC interval for %s at %r: %.6f [%.6f, %.6f]"
                    % (method, key, auc, lo, hi)
                )
        delta = float(row["delta_stage4_minus_raw"])
        delta_lo = float(row["delta_stage4_minus_raw_ci_lo"])
        delta_hi = float(row["delta_stage4_minus_raw_ci_hi"])
        if not delta_lo <= delta <= delta_hi:
            raise ValueError("invalid Stage-4 paired delta interval at %r" % (key,))

    for key, expected_mode in (
        ("primary_detected_conditioned", "detected"),
        ("secondary_fixed_power", "power"),
    ):
        summary = evaluation[key]
        if summary.get("snr_mode") != expected_mode:
            raise ValueError("%s has unexpected snr_mode" % key)
        methods = summary.get("methods")
        if not isinstance(methods, dict):
            raise ValueError("%s is missing methods" % key)
        for method in METHODS:
            if method not in methods:
                raise ValueError("%s is missing method %s" % (key, method))
            item = methods[method]
            auc = finite_float(item.get("auc"), "%s.%s.auc" % (key, method))
            lo = finite_float(item.get("ci_lo"), "%s.%s.ci_lo" % (key, method))
            hi = finite_float(item.get("ci_hi"), "%s.%s.ci_hi" % (key, method))
            if not (0.0 <= lo <= auc <= hi <= 1.0):
                raise ValueError(
                    "invalid pooled AUC interval for %s/%s" % (key, method)
                )


def select_rows(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    regime: str = "",
) -> List[Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if width_min <= float(row["width_hz"]) <= width_max
        and (not regime or row["snr_mode"] == regime)
    ]
    if not selected:
        raise ValueError(
            "no rows in requested interval %.3f--%.3f Hz" % (width_min, width_max)
        )
    return selected


def row_index(
    rows: Sequence[Mapping[str, object]],
) -> Dict[Tuple[str, str, float], Mapping[str, object]]:
    return {
        (str(row["snr_mode"]), str(row["shape"]), float(row["width_hz"])): row
        for row in rows
    }


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    outputs: List[Path] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        destination = base_path.with_suffix(".%s" % file_format)
        fd, temporary_name = tempfile.mkstemp(
            prefix=".%s." % base_path.name,
            suffix=".%s" % file_format,
            dir=str(base_path.parent),
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            fig.savefig(
                str(temporary_path),
                format=file_format,
                dpi=dpi if file_format == "png" else None,
                bbox_inches="tight",
                metadata={
                    "Creator": "LOFTS Stage-4 frozen-results plotter %s"
                    % SCRIPT_VERSION
                },
            )
            os.replace(str(temporary_path), str(destination))
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        outputs.append(destination)
    plt.close(fig)
    return outputs


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_primary_endpoint(
    evaluation: Mapping[str, object],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    primary = evaluation["primary_detected_conditioned"]
    primary_widths = primary.get("width_interval_hz", (10.0, 100.0))
    primary_width_label = "%g–%g Hz" % (
        float(primary_widths[0]),
        float(primary_widths[1]),
    )
    success = primary.get("success_criteria", {}).get("primary_success") is True
    methods = primary["methods"]
    deltas = primary["deltas_vs_stage3_raw"]
    order = list(METHODS)
    delta_order = ["stage3_corrected", "matched_filter", "stage4"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw={"wspace": 0.38})
    fig.suptitle(
        "Primary detected-conditioned coincidence performance",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.965,
        "Frozen synthetic evaluation, %s, pooled across all trained profile families"
        % primary_width_label,
        ha="center",
        fontsize=10.5,
        color="#444444",
    )

    ax = axes[0]
    y = np.arange(len(order))
    for yi, method in zip(y, order):
        item = methods[method]
        value = float(item["auc"])
        lo = float(item["ci_lo"])
        hi = float(item["ci_hi"])
        ax.errorbar(
            value,
            yi,
            xerr=np.asarray([[value - lo], [hi - value]]),
            fmt="o",
            ms=8.5,
            capsize=4,
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            elinewidth=2,
            zorder=3,
        )
        ax.text(
            min(0.997, hi + 0.006),
            yi,
            "%.3f" % value,
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )
    ax.axvspan(0.8, 1.0, color="#EAF4EA", zorder=0)
    ax.axvline(0.5, color="#777777", ls="--", lw=1.2)
    ax.axvline(0.8, color="#2F6F44", ls=":", lw=1.6)
    ax.set_xlim(0.48, 1.035)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("Pooled AUC-ROC (95% CI)")
    ax.set_title("Pooled discrimination performance", loc="left")
    ax.xaxis.grid(True, color="#DDDDDD", lw=0.8)
    ax.text(0.505, len(order) - 0.15, "chance", color="#666666", fontsize=8)
    ax.text(0.803, len(order) - 0.15, "registered target", color="#2F6F44", fontsize=8)
    add_panel_label(ax, "A")

    ax = axes[1]
    y = np.arange(len(delta_order))
    for yi, method in zip(y, delta_order):
        item = deltas[method]
        value = float(item["delta_auc"])
        lo = float(item["ci_lo"])
        hi = float(item["ci_hi"])
        ax.errorbar(
            value,
            yi,
            xerr=np.asarray([[value - lo], [hi - value]]),
            fmt="o",
            ms=8.5,
            capsize=4,
            color=METHOD_COLORS[method],
            ecolor=METHOD_COLORS[method],
            elinewidth=2,
            zorder=3,
        )
        ax.text(
            hi + 0.004,
            yi,
            "+%.3f" % value,
            va="center",
            fontsize=9,
            color="#333333",
        )
    ax.axvspan(0.0, 0.205, color="#EAF4EA", zorder=0)
    ax.axvline(0.0, color="#333333", ls="--", lw=1.2)
    ax.set_xlim(-0.012, 0.212)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in delta_order])
    ax.invert_yaxis()
    ax.set_xlabel("Paired ΔAUC versus raw Stage 3 (95% CI)")
    ax.set_title("Paired difference relative to raw Stage 3", loc="left")
    ax.xaxis.grid(True, color="#DDDDDD", lw=0.8)
    add_panel_label(ax, "B")

    status = (
        "Pre-specified criterion met" if success else "Pre-specified criterion not met"
    )
    fig.text(
        0.985,
        0.015,
        "%s  •  n=%s pairs  •  %s bootstrap replicates  •  %.0f%% CI"
        % (
            status,
            format(int(primary["n_pairs"]), ","),
            format(int(evaluation.get("n_boot", 0)), ","),
            100.0 * float(evaluation.get("ci_level", 0.95)),
        ),
        ha="right",
        va="bottom",
        fontsize=9,
        color="#205D36" if success else "#8B1E1E",
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.17, right=0.97)
    return save_figure(fig, out_dir / "01_primary_endpoint_forest", formats, dpi)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    widths: Sequence[float],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    value_format: str,
    colorbar_label: str,
    fig: plt.Figure,
    norm: object = None,
) -> None:
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
    )
    ax.set_title(title, loc="left")
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels(["%g" % width for width in widths])
    ax.set_yticks(np.arange(len(SHAPE_ORDER)))
    ax.set_yticklabels([SHAPE_LABELS[shape] for shape in SHAPE_ORDER])
    ax.set_xlabel("Broadening FWHM (Hz)")
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = float(matrix[row_i, col_i])
            color_position = (
                (value - vmin) / (vmax - vmin) if norm is None else norm(value)
            )
            red, green, blue, _ = plt.get_cmap(cmap)(color_position)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.50 else "#111111"
            ax.text(
                col_i,
                row_i,
                value_format.format(0.0 if abs(value) < 0.0005 else value),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color=text_color,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label(colorbar_label)


def plot_performance_landscape(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    selected = select_rows(rows, width_min, width_max)
    index = row_index(selected)
    widths = sorted({float(row["width_hz"]) for row in selected})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 7.8), constrained_layout=True)
    fig.suptitle(
        "Width- and profile-resolved Stage-4 performance",
        fontsize=16,
        fontweight="bold",
    )

    for column, regime in enumerate(("detected", "power")):
        auc_matrix = np.asarray(
            [
                [float(index[(regime, shape, width)]["stage4_auc"]) for width in widths]
                for shape in SHAPE_ORDER
            ]
        )
        delta_matrix = np.asarray(
            [
                [
                    float(index[(regime, shape, width)]["delta_stage4_minus_raw"])
                    for width in widths
                ]
                for shape in SHAPE_ORDER
            ]
        )
        heatmap(
            axes[0, column],
            auc_matrix,
            widths,
            "%s population: Stage-4 AUC-ROC" % REGIME_LABELS[regime],
            "viridis",
            0.5,
            1.0,
            "{:.3f}",
            "AUC-ROC",
            fig,
        )
        heatmap(
            axes[1, column],
            delta_matrix,
            widths,
            "%s population: paired ΔAUC versus raw Stage 3" % REGIME_LABELS[regime],
            "magma",
            0.0,
            0.46,
            "+{:.3f}",
            "Paired ΔAUC",
            fig,
        )
    add_panel_label(axes[0, 0], "A")
    add_panel_label(axes[0, 1], "B")
    add_panel_label(axes[1, 0], "C")
    add_panel_label(axes[1, 1], "D")
    fig.text(
        0.5,
        -0.015,
        "Each cell contains 300 matches and 300 mismatches; all values use the frozen test manifest.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    return save_figure(fig, out_dir / "02_performance_landscape", formats, dpi)


def shape_summary(
    rows: Sequence[Mapping[str, object]],
    regime: str,
    metric: str,
    widths: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: MutableMapping[float, List[float]] = defaultdict(list)
    for row in rows:
        if row["snr_mode"] == regime and float(row["width_hz"]) in widths:
            values[float(row["width_hz"])].append(float(row[metric]))
    mean = np.asarray([np.mean(values[width]) for width in widths])
    low = np.asarray([np.min(values[width]) for width in widths])
    high = np.asarray([np.max(values[width]) for width in widths])
    return mean, low, high


def plot_population_boundary(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    selected = select_rows(rows, width_min, width_max)
    widths = np.asarray(sorted({float(row["width_hz"]) for row in selected}))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"wspace": 0.28})
    fig.suptitle(
        "Detected-conditioned and fixed-power performance across broadening width",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    for regime, linestyle in (("detected", "-"), ("power", "--")):
        mean, low, high = shape_summary(selected, regime, "stage4_auc", widths)
        axes[0].fill_between(
            widths,
            low,
            high,
            color=REGIME_COLORS[regime],
            alpha=0.13,
            linewidth=0,
        )
        axes[0].plot(
            widths,
            mean,
            linestyle=linestyle,
            marker="o",
            color=REGIME_COLORS[regime],
            label="%s mean" % REGIME_LABELS[regime],
        )
        mean, low, high = shape_summary(selected, regime, "stage4_recall", widths)
        axes[1].fill_between(
            widths,
            low,
            high,
            color=REGIME_COLORS[regime],
            alpha=0.13,
            linewidth=0,
        )
        axes[1].plot(
            widths,
            mean,
            linestyle=linestyle,
            marker="o",
            color=REGIME_COLORS[regime],
            label="%s mean" % REGIME_LABELS[regime],
        )

    axes[0].axhline(0.8, color="#2F6F44", ls=":", lw=1.3, label="Registered AUC target")
    axes[0].axhline(0.5, color="#666666", ls="--", lw=1.1, label="Chance")
    axes[0].set_ylim(0.45, 1.015)
    axes[0].set_ylabel("Shape-macro AUC-ROC")
    axes[0].set_title("Shape-macro discrimination performance", loc="left")

    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Recall at the locked validation threshold")
    axes[1].set_title("Recall at the validation-selected threshold", loc="left")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(widths)
        ax.set_xticklabels(["%g" % width for width in widths])
        ax.set_xlabel("Broadening FWHM (Hz)")
        ax.grid(True, color="#E1E1E1", lw=0.8)
        ax.legend(loc="lower left")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    fig.text(
        0.5,
        0.005,
        "Points are unweighted means across Lorentzian, Gaussian, and box profiles; bands are shape ranges, not confidence intervals.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98)
    return save_figure(fig, out_dir / "03_population_boundary", formats, dpi)


def aggregate_confusion(
    rows: Sequence[Mapping[str, object]], method: str
) -> np.ndarray:
    tn = sum(int(round(float(row["%s_tn" % method]))) for row in rows)
    fp = sum(int(round(float(row["%s_fp" % method]))) for row in rows)
    fn = sum(int(round(float(row["%s_fn" % method]))) for row in rows)
    tp = sum(int(round(float(row["%s_tp" % method]))) for row in rows)
    return np.asarray([[tn, fp], [fn, tp]], dtype=int)


def confusion_metrics(matrix: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def plot_operating_point(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    selected = select_rows(rows, width_min, width_max, regime="detected")
    matrices = {
        "stage3_raw": aggregate_confusion(selected, "stage3_raw"),
        "stage4": aggregate_confusion(selected, "stage4"),
    }
    operating_metrics = {
        method: confusion_metrics(matrix) for method, matrix in matrices.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7))
    fig.suptitle(
        "Operating-point comparison on the detected-conditioned test population",
        fontsize=15,
        fontweight="bold",
    )
    for ax, method, panel in zip(axes, ("stage3_raw", "stage4"), ("A", "B")):
        matrix = matrices[method]
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = matrix / row_totals
        ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        for row_i in range(2):
            for col_i in range(2):
                percent = 100.0 * normalized[row_i, col_i]
                ax.text(
                    col_i,
                    row_i,
                    "%s\n%.1f%%" % (format(int(matrix[row_i, col_i]), ","), percent),
                    ha="center",
                    va="center",
                    color="white" if normalized[row_i, col_i] > 0.55 else "#111111",
                    fontsize=13,
                    fontweight="semibold",
                )
        metrics = operating_metrics[method]
        ax.set_xticks((0, 1))
        ax.set_xticklabels(("Predicted mismatch", "Predicted match"))
        ax.set_yticks((0, 1))
        ax.set_yticklabels(("True mismatch", "True match"))
        ax.set_title(METHOD_LABELS[method], pad=10)
        ax.text(
            0.5,
            -0.20,
            "Precision %.3f  •  Recall %.3f  •  F1 %.3f  •  FPR %.2f%%"
            % (
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                100.0 * metrics["fpr"],
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="semibold",
            color="#333333",
        )
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        add_panel_label(ax, panel)
    fig.text(
        0.5,
        0.005,
        "Detected-conditioned %g–%g Hz test cells pooled at each model's frozen threshold; balanced synthetic prevalence."
        % (width_min, width_max),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.08, right=0.98, wspace=0.32)
    return save_figure(fig, out_dir / "04_locked_operating_point", formats, dpi)


def plot_cellwise_delta_forest(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    selected = select_rows(rows, width_min, width_max)
    index = row_index(selected)
    widths = sorted({float(row["width_hz"]) for row in selected}, reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 6.0), sharex=True, sharey=True)
    fig.suptitle(
        "Cell-wise paired AUC differences relative to raw Stage 3",
        fontsize=15,
        fontweight="bold",
    )
    offsets = {"detected": -0.13, "power": 0.13}
    markers = {"detected": "o", "power": "D"}
    for ax, shape, panel in zip(axes, SHAPE_ORDER, ("A", "B", "C")):
        for regime in ("detected", "power"):
            points = []
            lows = []
            highs = []
            y_values = []
            for yi, width in enumerate(widths):
                row = index[(regime, shape, width)]
                value = float(row["delta_stage4_minus_raw"])
                points.append(value)
                lows.append(float(row["delta_stage4_minus_raw_ci_lo"]))
                highs.append(float(row["delta_stage4_minus_raw_ci_hi"]))
                y_values.append(yi + offsets[regime])
            points_array = np.asarray(points)
            ax.errorbar(
                points_array,
                y_values,
                xerr=np.vstack(
                    (points_array - np.asarray(lows), np.asarray(highs) - points_array)
                ),
                fmt=markers[regime],
                ms=6,
                capsize=3,
                elinewidth=1.6,
                color=REGIME_COLORS[regime],
                label=REGIME_LABELS[regime],
            )
        ax.axvline(0.0, color="#333333", ls="--", lw=1.1)
        ax.set_title(SHAPE_LABELS[shape])
        ax.set_yticks(np.arange(len(widths)))
        ax.set_yticklabels(["%g Hz" % width for width in widths])
        ax.set_xlim(-0.025, 0.515)
        ax.xaxis.grid(True, color="#E0E0E0", lw=0.8)
        ax.set_xlabel("Paired ΔAUC (Stage 4 − raw Stage 3)")
        add_panel_label(ax, panel)
    axes[0].set_ylabel("Broadening FWHM")
    axes[0].legend(loc="lower right")
    fig.text(
        0.5,
        0.02,
        "Error bars are the reported per-cell paired 95% bootstrap confidence intervals; 300 matches and 300 mismatches per cell.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.09, right=0.98, wspace=0.13)
    return save_figure(fig, out_dir / "05_cellwise_delta_forest", formats, dpi)


def load_paired_filter_comparison(path: Path) -> Dict[str, object]:
    """Load either supported locked Stage-4-minus-filter result schema.

    The first schema is emitted by ``rerun_frozen_test_a_pair_export.py`` and
    stores both detected-conditioned and fixed-power comparisons.  The second
    is the preserved-pair recovery artifact produced after the historical
    bootstrap ordering bug was identified.  Both encode the same registered
    detected-conditioned estimand.
    """

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("paired-filter result must be a JSON object")
    if bool(payload.get("model_or_threshold_tuning_performed", False)):
        raise ValueError("paired-filter result reports model or threshold tuning")
    validation = payload.get("pair_export_validation")
    if isinstance(validation, dict) and not bool(validation.get("validated", False)):
        raise ValueError("paired-filter result did not validate its frozen pair export")

    if isinstance(payload.get("comparisons"), dict):
        comparison = payload["comparisons"].get("detected")
        if not isinstance(comparison, dict):
            raise ValueError("paired-filter result lacks comparisons.detected")
        methods = comparison.get("methods")
        if not isinstance(methods, dict):
            raise ValueError("paired-filter comparison lacks method summaries")
        stage4_auc = finite_float(
            methods.get("stage4", {}).get("auc"), "methods.stage4.auc"
        )
        filter_auc = finite_float(
            methods.get("matched_filter", {}).get("auc"),
            "methods.matched_filter.auc",
        )
        delta = comparison.get("stage4_minus_matched_filter")
        n_pairs = int(comparison.get("n_pairs", 0))
        width_interval = comparison.get("width_interval_hz", [10.0, 100.0])
        bootstrap = comparison.get("bootstrap", {})
    else:
        if payload.get("regime") not in (None, "detected"):
            raise ValueError("paired-filter result is not detected-conditioned")
        stage4_auc = finite_float(payload.get("stage4_auc"), "stage4_auc")
        filter_auc = finite_float(
            payload.get("matched_filter_auc"), "matched_filter_auc"
        )
        delta = payload.get("stage4_minus_matched_filter")
        n_pairs = int(payload.get("n_pairs", 0))
        width_interval = payload.get("width_interval_hz", [10.0, 100.0])
        bootstrap = payload.get("bootstrap", {})

    if not isinstance(delta, dict):
        raise ValueError("paired-filter result lacks the paired delta summary")
    delta_auc = finite_float(delta.get("delta_auc"), "delta_auc")
    ci_lo = finite_float(delta.get("ci_lo"), "delta.ci_lo")
    ci_hi = finite_float(delta.get("ci_hi"), "delta.ci_hi")
    if not (0.0 <= stage4_auc <= 1.0 and 0.0 <= filter_auc <= 1.0):
        raise ValueError("paired-filter AUC values must lie in [0, 1]")
    if not ci_lo <= delta_auc <= ci_hi:
        raise ValueError("paired-filter delta is outside its confidence interval")
    if abs(delta_auc - (stage4_auc - filter_auc)) > 5e-9:
        raise ValueError("paired-filter delta is inconsistent with the two AUCs")
    if n_pairs <= 0:
        raise ValueError("paired-filter result has no evaluated pairs")
    if not isinstance(width_interval, (list, tuple)) or len(width_interval) != 2:
        raise ValueError("paired-filter width interval must have two endpoints")

    return {
        "n_pairs": n_pairs,
        "width_interval_hz": [float(width_interval[0]), float(width_interval[1])],
        "stage4_auc": stage4_auc,
        "filter_auc": filter_auc,
        "delta_auc": delta_auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "bootstrap": dict(bootstrap) if isinstance(bootstrap, dict) else {},
        "source": str(path.resolve()),
    }


def plot_paired_filter_summary(
    comparison: Mapping[str, object],
    evaluation: Mapping[str, object],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Plot the locked incremental Stage-4 value beyond the filter baseline."""

    primary = evaluation["primary_detected_conditioned"]
    methods = primary["methods"]
    stage4_auc = float(comparison["stage4_auc"])
    filter_auc = float(comparison["filter_auc"])
    if abs(stage4_auc - float(methods["stage4"]["auc"])) > 5e-6:
        raise ValueError("paired result does not reproduce the frozen Stage-4 AUC")
    if abs(filter_auc - float(methods["matched_filter"]["auc"])) > 5e-6:
        raise ValueError("paired result does not reproduce the frozen filter AUC")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.suptitle(
        "Incremental discrimination of Stage 4 beyond the transparent filter",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.91,
        "Frozen Synthetic Test A; detected-conditioned 10–100 Hz population",
        ha="center",
        fontsize=11,
        color="#444444",
    )

    labels = ("Transparent filter", "Stage 4")
    keys = ("matched_filter", "stage4")
    colors = (METHOD_COLORS["matched_filter"], METHOD_COLORS["stage4"])
    y = np.asarray([1.0, 0.0])
    aucs = np.asarray([filter_auc, stage4_auc])
    lows = np.asarray([float(methods[key]["ci_lo"]) for key in keys])
    highs = np.asarray([float(methods[key]["ci_hi"]) for key in keys])
    for yi, auc, low, high, color in zip(y, aucs, lows, highs, colors):
        axes[0].errorbar(
            auc,
            yi,
            xerr=[[auc - low], [high - auc]],
            fmt="o",
            color=color,
            ms=9,
            capsize=4,
            elinewidth=2,
        )
        axes[0].text(auc + 0.0012, yi, "%.4f" % auc, va="center", fontsize=10)
    axes[0].set_yticks(y, labels)
    lower = max(0.90, float(np.min(lows)) - 0.01)
    axes[0].set_xlim(lower, 1.002)
    axes[0].set_ylim(-0.65, 1.65)
    axes[0].set_xlabel("AUC-ROC (95% CI)")
    axes[0].set_title("Pooled discrimination performance")
    axes[0].grid(axis="x", color="#E0E0E0", linewidth=0.8)
    add_panel_label(axes[0], "A")

    delta = float(comparison["delta_auc"])
    ci_lo = float(comparison["ci_lo"])
    ci_hi = float(comparison["ci_hi"])
    axes[1].axvline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    axes[1].errorbar(
        delta,
        0.0,
        xerr=[[delta - ci_lo], [ci_hi - delta]],
        fmt="o",
        color=METHOD_COLORS["stage4"],
        ms=10,
        capsize=5,
        elinewidth=2.2,
    )
    axes[1].text(
        delta,
        -0.24,
        "ΔAUC = %+.4f\n95%% CI [%+.4f, %+.4f]" % (delta, ci_lo, ci_hi),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )
    margin = max(0.004, 0.22 * (ci_hi - ci_lo))
    axes[1].set_xlim(min(-0.001, ci_lo - margin), ci_hi + 2.5 * margin)
    axes[1].set_ylim(-0.62, 0.58)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Paired ΔAUC (Stage 4 − transparent filter)")
    axes[1].set_title("Incremental Stage-4 contribution")
    axes[1].grid(axis="x", color="#E0E0E0", linewidth=0.8)
    add_panel_label(axes[1], "B")

    bootstrap = comparison.get("bootstrap", {})
    replicates = (
        int(bootstrap.get("replicates", 0)) if isinstance(bootstrap, dict) else 0
    )
    footer = (
        "Paired bootstrap stratified by profile, width, class, and negative case; "
        "n=%s pairs" % format(int(comparison["n_pairs"]), ",")
    )
    if replicates:
        footer += "; %s bootstrap replicates" % format(replicates, ",")
    footer += "."
    fig.text(0.5, 0.025, footer, ha="center", fontsize=9, color="#4A4A4A")
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.12, right=0.98, wspace=0.34)
    return save_figure(
        fig,
        out_dir / "06_stage4_vs_transparent_filter_paired",
        formats,
        dpi,
    )


def plot_filter_increment(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
    output_stem: str = "06_stage4_minus_filter_descriptive",
) -> List[Path]:
    selected = select_rows(rows, width_min, width_max)
    index = row_index(selected)
    widths = sorted({float(row["width_hz"]) for row in selected})
    matrices = {}
    for regime in ("detected", "power"):
        matrices[regime] = np.asarray(
            [
                [
                    float(index[(regime, shape, width)]["stage4_auc"])
                    - float(index[(regime, shape, width)]["matched_filter_auc"])
                    for width in widths
                ]
                for shape in SHAPE_ORDER
            ]
        )
    limit = max(
        0.03,
        math.ceil(100.0 * max(np.max(np.abs(m)) for m in matrices.values())) / 100.0,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    fig.suptitle(
        "Cell-resolved AUC differences between Stage 4 and the transparent filter",
        fontsize=15,
        fontweight="bold",
    )
    for ax, regime, panel in zip(axes, ("detected", "power"), ("A", "B")):
        heatmap(
            ax,
            matrices[regime],
            widths,
            "%s population" % REGIME_LABELS[regime],
            "RdBu_r",
            -limit,
            limit,
            "{:+.3f}",
            "Stage-4 AUC − filter AUC",
            fig,
            norm=norm,
        )
        add_panel_label(ax, panel)
    fig.text(
        0.5,
        -0.025,
        "Cell values are descriptive point estimates; pooled paired uncertainty is reported in the incremental-value figure.",
        ha="center",
        fontsize=9,
        color="#8B3A3A",
        fontweight="semibold",
    )
    return save_figure(fig, out_dir / output_stem, formats, dpi)


def write_derived_summary(
    rows: Sequence[Mapping[str, object]],
    width_min: float,
    width_max: float,
    path: Path,
) -> None:
    selected = select_rows(rows, width_min, width_max)
    widths = sorted({float(row["width_hz"]) for row in selected})
    fieldnames = [
        "snr_mode",
        "width_hz",
        "n_shapes",
        "stage3_raw_auc_macro",
        "stage3_corrected_auc_macro",
        "matched_filter_auc_macro",
        "stage4_auc_macro",
        "stage4_auc_shape_min",
        "stage4_auc_shape_max",
        "stage4_recall_macro",
        "stage4_recall_shape_min",
        "stage4_recall_shape_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for regime in ("detected", "power"):
            for width in widths:
                cell = [
                    row
                    for row in selected
                    if row["snr_mode"] == regime and float(row["width_hz"]) == width
                ]
                stage4_auc = np.asarray([float(row["stage4_auc"]) for row in cell])
                stage4_recall = np.asarray(
                    [float(row["stage4_recall"]) for row in cell]
                )
                writer.writerow(
                    {
                        "snr_mode": regime,
                        "width_hz": "%g" % width,
                        "n_shapes": len(cell),
                        "stage3_raw_auc_macro": "%.9f"
                        % np.mean([float(row["stage3_raw_auc"]) for row in cell]),
                        "stage3_corrected_auc_macro": "%.9f"
                        % np.mean([float(row["stage3_corrected_auc"]) for row in cell]),
                        "matched_filter_auc_macro": "%.9f"
                        % np.mean([float(row["matched_filter_auc"]) for row in cell]),
                        "stage4_auc_macro": "%.9f" % np.mean(stage4_auc),
                        "stage4_auc_shape_min": "%.9f" % np.min(stage4_auc),
                        "stage4_auc_shape_max": "%.9f" % np.max(stage4_auc),
                        "stage4_recall_macro": "%.9f" % np.mean(stage4_recall),
                        "stage4_recall_shape_min": "%.9f" % np.min(stage4_recall),
                        "stage4_recall_shape_max": "%.9f" % np.max(stage4_recall),
                    }
                )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    inputs: Sequence[Path],
    outputs: Sequence[Path],
    width_min: float,
    width_max: float,
    paired_filter_included: bool,
) -> None:
    payload = {
        "format_version": 1,
        "plotter_version": SCRIPT_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "width_min_hz": width_min,
            "width_max_hz": width_max,
            "frozen_evaluation_only": True,
        },
        "inputs": [
            {"path": str(item.resolve()), "sha256": sha256(item)} for item in inputs
        ],
        "outputs": [
            {
                "path": str(item.resolve()),
                "sha256": sha256(item),
                "bytes": item.stat().st_size,
            }
            for item in outputs
        ],
        "interpretation_guards": [
            "Detected-conditioned performance is not end-to-end detection efficiency.",
            "Shape ranges in Figure 3 are descriptive ranges, not confidence intervals.",
            (
                "The pooled Stage-4-minus-filter result uses paired, stratified bootstrap uncertainty."
                if paired_filter_included
                else "The cellwise Stage-4-minus-filter figure contains point differences, not a paired test."
            ),
            "The result is synthetic coincidence on real Sweden-accessible background proxies, not real Ireland-Sweden validation.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create varied publication figures from frozen Stage-4 CSV/JSON outputs."
    )
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument(
        "--paired-filter-json",
        type=Path,
        default=None,
        help="Locked paired Stage-4-minus-transparent-filter result",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width-min", type=float, default=10.0)
    parser.add_argument("--width-max", type=float, default=100.0)
    parser.add_argument("--formats", type=parse_formats, default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=240)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.width_min <= 0 or args.width_max < args.width_min:
        raise ValueError("require 0 < width_min <= width_max")
    if args.dpi < 72:
        raise ValueError("dpi must be at least 72")
    configure_style()
    rows = load_rows(args.results_csv)
    evaluation = load_evaluation(args.evaluation_json)
    validate_inputs(rows, evaluation)
    paired_filter = (
        None
        if args.paired_filter_json is None
        else load_paired_filter_comparison(args.paired_filter_json)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    outputs.extend(
        plot_primary_endpoint(evaluation, args.out_dir, args.formats, args.dpi)
    )
    outputs.extend(
        plot_performance_landscape(
            rows, args.width_min, args.width_max, args.out_dir, args.formats, args.dpi
        )
    )
    outputs.extend(
        plot_population_boundary(
            rows, args.width_min, args.width_max, args.out_dir, args.formats, args.dpi
        )
    )
    outputs.extend(
        plot_operating_point(
            rows, args.width_min, args.width_max, args.out_dir, args.formats, args.dpi
        )
    )
    outputs.extend(
        plot_cellwise_delta_forest(
            rows, args.width_min, args.width_max, args.out_dir, args.formats, args.dpi
        )
    )
    if paired_filter is not None:
        outputs.extend(
            plot_paired_filter_summary(
                paired_filter, evaluation, args.out_dir, args.formats, args.dpi
            )
        )
        descriptive_stem = "07_stage4_vs_filter_cellwise_descriptive"
    else:
        descriptive_stem = "06_stage4_minus_filter_descriptive"
    outputs.extend(
        plot_filter_increment(
            rows,
            args.width_min,
            args.width_max,
            args.out_dir,
            args.formats,
            args.dpi,
            output_stem=descriptive_stem,
        )
    )

    derived_path = args.out_dir / "derived_shape_macro_summary.csv"
    write_derived_summary(rows, args.width_min, args.width_max, derived_path)
    outputs.append(derived_path)
    manifest_path = args.out_dir / "visualization_manifest.json"
    write_manifest(
        manifest_path,
        tuple(
            item
            for item in (
                args.results_csv,
                args.evaluation_json,
                args.paired_filter_json,
            )
            if item is not None
        ),
        outputs,
        args.width_min,
        args.width_max,
        paired_filter is not None,
    )
    outputs.append(manifest_path)
    print("Created %d frozen-result artifacts in %s" % (len(outputs), args.out_dir))
    for output in outputs:
        print("  %s" % output)


if __name__ == "__main__":
    main(build_parser().parse_args())
