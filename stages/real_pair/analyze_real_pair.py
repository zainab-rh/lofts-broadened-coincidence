#!/usr/bin/env python3
"""Descriptive, label-free analysis of a real BLISS → Stage-4 pair.

No ROC, recall, precision, FPR, or detection efficiency is computed because
real candidate labels are unavailable.  The principal diagnostic is a paired
comparison between each observed candidate pair and preregistered distant
frequency controls extracted from the same observations.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from lofts_bliss_schema import (
    atomic_write_text,
    load_candidates,
    read_json_records,
    sha256_file,
    write_json,
)

METHODS = {
    "stage3_raw_score": "Raw Stage 3",
    "stage3_corrected_score": "Corrected Stage 3",
    "matched_filter_score": "Transparent filter",
    "stage4_score": "Stage 4",
}


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def bootstrap_mean(values: Sequence[float], n_boot: int, seed: int) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.integers(0, array.size, array.size)
        estimates[index] = float(np.mean(array[sample]))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95": [float(value) for value in np.percentile(estimates, [2.5, 97.5])],
        "positive_fraction": float(np.mean(array > 0)),
    }


def clustered_bootstrap_mean(
    values: Sequence[float],
    blocks: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap the candidate-weighted mean by physical-frequency block."""

    array = np.asarray(values, dtype=float)
    block_array = np.asarray([str(value) for value in blocks])
    if array.shape != block_array.shape:
        raise ValueError("clustered bootstrap values/blocks must have equal shape")
    if array.size == 0:
        return {
            "n": 0,
            "n_blocks": 0,
            "mean": None,
            "median": None,
            "ci95": [None, None],
            "positive_fraction": None,
            "resampling_unit": "common physical-frequency block",
        }
    unique_blocks = np.unique(block_array)
    result = {
        "n": int(array.size),
        "n_blocks": int(unique_blocks.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "ci95": [None, None],
        "positive_fraction": float(np.mean(array > 0)),
        "resampling_unit": "common physical-frequency block",
    }
    if unique_blocks.size < 2:
        result["interval_unavailable_reason"] = (
            "fewer than two independent physical-frequency blocks contributed"
        )
        return result
    indices = {block: np.flatnonzero(block_array == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled_blocks = rng.choice(
            unique_blocks, size=unique_blocks.size, replace=True
        )
        sampled_indices = np.concatenate([indices[block] for block in sampled_blocks])
        estimates[index] = float(np.mean(array[sampled_indices]))
    result["ci95"] = [float(value) for value in np.percentile(estimates, [2.5, 97.5])]
    return result


def paired_control_rows(
    primary: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    primary_by_id = {str(item["pair_id"]): item for item in primary}
    controls_by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in controls:
        controls_by_source[str(item["source_pair_id"])].append(item)
    rows: List[Dict[str, Any]] = []
    shifts_per_source = Counter()
    for pair_id, observed in primary_by_id.items():
        candidates = controls_by_source.get(pair_id, [])
        if not candidates:
            continue
        shifts_per_source[len(candidates)] += 1
        row: Dict[str, Any] = {
            "pair_id": pair_id,
            "detection_state": observed.get("detection_state"),
            "operational_eligibility": observed.get("operational_eligibility"),
            "route": observed.get("route"),
            "n_controls": len(candidates),
            "resampling_block_id": observed.get("resampling_block_id"),
        }
        for method in METHODS:
            observed_score = float(observed[method])
            control_scores = np.asarray([float(item[method]) for item in candidates])
            control_mean = float(np.mean(control_scores))
            row[method + "_observed"] = observed_score
            row[method + "_control_mean"] = control_mean
            row[method + "_delta"] = observed_score - control_mean
        rows.append(row)
    rows.sort(key=lambda item: item["pair_id"])
    audit = {
        "n_primary": len(primary),
        "n_control_rows": len(controls),
        "n_primary_with_at_least_one_control": len(rows),
        "controls_per_source_distribution": {
            str(key): value for key, value in sorted(shifts_per_source.items())
        },
    }
    return rows, audit


def _catalog_summary(paths: Sequence[str]) -> Dict[str, Any]:
    output = {}
    for path in paths:
        candidates = load_candidates(path)
        if not candidates:
            continue
        station_ids = {item.station_id for item in candidates}
        if len(station_ids) != 1:
            raise ValueError("each canonical catalog must contain one station")
        station_id = next(iter(station_ids))
        widths = Counter(
            int(item.extras["native_width_channels"]) for item in candidates
        )
        flags = Counter(
            str(item.extras.get("source_flag") or "unflagged") for item in candidates
        )
        ratios = [
            float(item.extras["bank_standard_ratio"])
            for item in candidates
            if item.extras.get("bank_standard_ratio") not in (None, "")
        ]
        output[station_id] = {
            "n_candidates": len(candidates),
            "native_width_counts": {
                str(key): value for key, value in sorted(widths.items())
            },
            "source_flag_counts": dict(sorted(flags.items())),
            "n_broadband_rfi_like": sum(
                bool(item.extras.get("broadband_rfi_like", False))
                for item in candidates
            ),
            "broadband_rfi_like_fraction": float(
                np.mean(
                    [
                        bool(item.extras.get("broadband_rfi_like", False))
                        for item in candidates
                    ]
                )
            ),
            "bank_standard_ratio_median": (
                None if not ratios else float(np.median(ratios))
            ),
            "source": str(Path(path).resolve()),
            "source_sha256": sha256_file(path),
        }
    return output


def _plots(
    out_dir: Path,
    catalog_summary: Mapping[str, Any],
    union_entries: Sequence[Mapping[str, Any]],
    primary: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stations = sorted(catalog_summary)
    width_values = sorted(
        {
            int(width)
            for station in stations
            for width in catalog_summary[station]["native_width_counts"]
        }
    )
    fig, axes = plt.subplots(
        1,
        max(1, len(stations)),
        figsize=(6 * max(1, len(stations)), 4.8),
        squeeze=False,
    )
    for axis, station in zip(axes[0], stations):
        counts = catalog_summary[station]["native_width_counts"]
        axis.bar(
            [str(value) for value in width_values],
            [counts.get(str(value), 0) for value in width_values],
        )
        axis.set_title("%s raw catalog" % station)
        axis.set_xlabel("Native winning template (channels)")
        axis.set_ylabel("Candidate count")
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Naoise raw-catalog template occupancy (uncollapsed)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            out_dir / ("01_catalog_template_occupancy." + suffix),
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    route_counts = Counter(str(item.get("route")) for item in union_entries)
    eligibility_counts = Counter(
        str(item.get("operational_eligibility")) for item in union_entries
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].barh(list(route_counts), list(route_counts.values()))
    axes[0].set_title("Width/coverage routes")
    axes[0].set_xlabel("Union entries")
    axes[1].barh(list(eligibility_counts), list(eligibility_counts.values()))
    axes[1].set_title("Operational detection eligibility")
    axes[1].set_xlabel("Union entries")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Real Ireland–Sweden candidate-union composition")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            out_dir / ("02_union_composition." + suffix), dpi=220, bbox_inches="tight"
        )
    plt.close(fig)

    associated = [item for item in union_entries if item.get("association")]
    if associated:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        definitions = (
            ("frequency_delta_hz", "Frequency difference (Hz)"),
            ("drift_delta_hz_s", "Drift difference (Hz s$^{-1}$)"),
            ("log_width_delta", "Absolute log-width difference"),
        )
        for axis, (key, label) in zip(axes, definitions):
            values = [float(item["association"][key]) for item in associated]
            axis.hist(
                values, bins=min(30, max(5, int(np.sqrt(len(values))))), alpha=0.85
            )
            axis.set_xlabel(label)
            axis.set_ylabel("Associated pairs")
            axis.grid(alpha=0.2)
        fig.suptitle("Two-station association residuals under the frozen pilot policy")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                out_dir / ("03_association_residuals." + suffix),
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)

    if primary and controls:
        fig, axes = plt.subplots(
            1, len(METHODS), figsize=(4.2 * len(METHODS), 4.8), squeeze=False
        )
        for axis, (method, label) in zip(axes[0], METHODS.items()):
            observed = [float(item[method]) for item in primary]
            shifted = [float(item[method]) for item in controls]
            # Matplotlib renamed ``labels`` to ``tick_labels`` in 3.9.  Select
            # the supported name at runtime so the Sweden environment and new
            # developer environments are both warning-free.
            label_key = (
                "tick_labels"
                if "tick_labels" in inspect.signature(axis.boxplot).parameters
                else "labels"
            )
            boxplot_kwargs = {label_key: ["Observed", "Shifted"]}
            axis.boxplot([observed, shifted], showfliers=False, **boxplot_kwargs)
            axis.set_title(label)
            axis.set_ylabel("Ranking score")
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle(
            "Observed real pairs versus preregistered frequency-shift controls"
        )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                out_dir / ("04_observed_vs_shifted_scores." + suffix),
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)

    if paired_rows:
        fig, axes = plt.subplots(
            1, len(METHODS), figsize=(4.2 * len(METHODS), 4.8), squeeze=False
        )
        for axis, (method, label) in zip(axes[0], METHODS.items()):
            deltas = [float(item[method + "_delta"]) for item in paired_rows]
            axis.hist(
                deltas, bins=min(30, max(5, int(np.sqrt(len(deltas))))), alpha=0.85
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
            axis.set_title(label)
            axis.set_xlabel("Observed − mean shifted-control score")
            axis.set_ylabel("Paired candidates")
            axis.grid(alpha=0.2)
        fig.suptitle(
            "Within-candidate score contrast; descriptive, not classification accuracy"
        )
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                out_dir / ("05_paired_control_deltas." + suffix),
                dpi=220,
                bbox_inches="tight",
            )
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_summary = _catalog_summary(args.candidate_files)
    union_entries = read_json_records(args.union)
    primary = read_json_records(args.primary_predictions)
    controls = read_json_records(args.control_predictions)
    if not primary:
        raise ValueError("primary real-pair prediction table is empty")
    if not controls:
        raise ValueError("frequency-shift control prediction table is empty")
    paired_rows, paired_audit = paired_control_rows(primary, controls)
    if not paired_rows:
        raise ValueError("no primary predictions have valid paired controls")
    missing_blocks = [
        str(item["pair_id"])
        for item in paired_rows
        if not str(item.get("resampling_block_id") or "").strip()
    ]
    if missing_blocks:
        raise ValueError(
            "real predictions lack physical-frequency resampling blocks: %s"
            % missing_blocks[:5]
        )
    paired_statistics = {
        method: clustered_bootstrap_mean(
            [float(item[method + "_delta"]) for item in paired_rows],
            [str(item["resampling_block_id"]) for item in paired_rows],
            args.n_boot,
            args.seed + index,
        )
        for index, method in enumerate(METHODS)
    }
    top = sorted(
        primary, key=lambda item: (-float(item["stage4_score"]), str(item["pair_id"]))
    )[: args.top_n]
    _write_csv(str(out_dir / "real_pair_top_candidates.csv"), top)
    _write_csv(str(out_dir / "paired_frequency_shift_deltas.csv"), paired_rows)
    summary = {
        "format_version": 2,
        "dataset_role": "unlabeled_real_barycentric_pair",
        "catalogs": catalog_summary,
        "union": {
            "n_entries": len(union_entries),
            "detection_states": dict(
                sorted(
                    Counter(
                        str(item.get("detection_state")) for item in union_entries
                    ).items()
                )
            ),
            "operational_eligibility": dict(
                sorted(
                    Counter(
                        str(item.get("operational_eligibility"))
                        for item in union_entries
                    ).items()
                )
            ),
            "routes": dict(
                sorted(
                    Counter(str(item.get("route")) for item in union_entries).items()
                )
            ),
        },
        "inference": {
            "n_primary_scores": len(primary),
            "n_control_scores": len(controls),
            "n_above_frozen_synthetic_threshold": sum(
                bool(item.get("stage4_above_frozen_synthetic_threshold"))
                for item in primary
            ),
            "threshold_interpretation": "exploratory frozen Synthetic-Test-A reference only",
        },
        "paired_frequency_shift_control": {
            **paired_audit,
            "method_deltas": paired_statistics,
            "estimand": "observed score minus mean distant-frequency control score for the same source pair",
            "uncertainty_resampling_unit": "common physical-frequency block",
        },
        "inputs": {
            "union": str(Path(args.union).resolve()),
            "union_sha256": sha256_file(args.union),
            "primary_predictions": str(Path(args.primary_predictions).resolve()),
            "primary_predictions_sha256": sha256_file(args.primary_predictions),
            "control_predictions": str(Path(args.control_predictions).resolve()),
            "control_predictions_sha256": sha256_file(args.control_predictions),
        },
        "labels_used": False,
        "forbidden_claims": [
            "AUC or classification accuracy on real candidates",
            "real-candidate recall, precision, FPR, or end-to-end completeness",
            "astrophysical posterior probability from Stage-4 score",
            "real technosignature validation",
        ],
    }
    write_json(str(out_dir / "real_pair_analysis.json"), summary)
    report = [
        "# LOFTS real-pair BLISS → Stage-4 analysis",
        "",
        "This is a **real, barycentrically corrected, simultaneous Ireland–Sweden candidate-processing pilot**. It is unlabeled and therefore does not provide a real-data AUC, recall, precision, FPR, or end-to-end completeness estimate.",
        "",
        "## Catalog and union",
        "",
    ]
    for station, values in sorted(catalog_summary.items()):
        report.append(
            "- %s: %d raw candidates; %.1f%% broadband-RFI-like by Naoise's retained ratio diagnostic."
            % (
                station,
                values["n_candidates"],
                100.0 * values["broadband_rfi_like_fraction"],
            )
        )
    report.extend(
        [
            "- Union entries: %d." % len(union_entries),
            "- Scored primary high-resolution entries: %d." % len(primary),
            "- Frequency-shift control scores: %d." % len(controls),
            "",
            "## Paired score-control diagnostics",
            "",
        ]
    )
    for method, label in METHODS.items():
        values = paired_statistics[method]
        if values["n"]:
            if values["ci95"][0] is None:
                report.append(
                    "- %s observed-minus-control mean: %.6g (n=%d candidates; "
                    "interval withheld because only %d physical-frequency block "
                    "contributed)."
                    % (label, values["mean"], values["n"], values["n_blocks"])
                )
            else:
                report.append(
                    "- %s observed-minus-control mean: %.6g (frequency-block "
                    "bootstrap 95%% interval %.6g to %.6g; n=%d candidates in "
                    "%d blocks)."
                    % (
                        label,
                        values["mean"],
                        values["ci95"][0],
                        values["ci95"][1],
                        values["n"],
                        values["n_blocks"],
                    )
                )
    report.extend(
        [
            "",
            "These score contrasts are empirical diagnostics, not labeled classification performance. The frozen Stage-4 threshold remains a Synthetic-Test-A operating point and is not calibrated to real candidate prevalence.",
            "",
            "## Next quantitative gate",
            "",
            "Run blind injections through this exact Naoise commit and these exact barycentric backgrounds, freeze recovery-derived association tolerances, and then execute the retained Synthetic-Test-B labeling/evaluation path. That is the experiment that can report end-to-end completeness and conditional Stage-4 AUC.",
        ]
    )
    atomic_write_text(str(out_dir / "real_pair_report.md"), "\n".join(report) + "\n")
    _plots(out_dir, catalog_summary, union_entries, primary, controls, paired_rows)
    print("Wrote label-free real-pair analysis to %s" % out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze unlabeled real-pair predictions with shifted controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--union", required=True)
    parser.add_argument("--primary-predictions", required=True)
    parser.add_argument("--control-predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--top-n", type=int, default=50)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
