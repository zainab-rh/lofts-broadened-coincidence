#!/usr/bin/env python3
"""Deterministically re-run frozen Test A and export pair-level scores.

This wrapper imports the existing ``evaluate_stage4.py`` unchanged, captures
its in-memory labels/scores, verifies the regenerated manifest and aggregate
AUCs against the frozen files, and computes the missing paired
Stage-4-minus-filter AUC interval.  It neither retrains nor selects a model,
threshold, preprocessing option, test mixture, or test seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


def custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap the unchanged frozen Stage-4 evaluator, reproduce Test A, "
            "and export the direct paired Stage-4-minus-filter interval. "
            "All remaining evaluate_stage4.py arguments are forwarded."
        )
    )
    parser.add_argument("--stage4-code-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--reference-results-csv", required=True)
    parser.add_argument("--pair-output", required=True)
    parser.add_argument("--direct-output", required=True)
    parser.add_argument("--aggregate-tolerance", type=float, default=5e-6)
    parser.add_argument("--acknowledge-locked-posthoc-analysis", action="store_true")
    return parser


def csv_index(path: str) -> Dict[Tuple[str, str, float], Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (str(row["snr_mode"]), str(row["shape"]), float(row["width_hz"])): row
        for row in rows
    }


def main() -> None:
    custom, remaining = custom_parser().parse_known_args()
    if not custom.acknowledge_locked_posthoc_analysis:
        raise ValueError("use --acknowledge-locked-posthoc-analysis")
    code_dir = str(Path(custom.stage4_code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import evaluate_stage4 as evaluator
    from lofts_bliss_schema import atomic_write_text, sha256_file, write_json

    args = evaluator.build_parser().parse_args(remaining)
    out_dir = Path(args.out_dir).resolve()
    reference_manifest = Path(custom.reference_manifest).resolve()
    reference_results = Path(custom.reference_results_csv).resolve()
    if out_dir == reference_manifest.parent or out_dir == reference_results.parent:
        raise ValueError(
            "reanalysis output must be a new directory, not the frozen Test-A directory"
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            "reanalysis output directory must be new/empty: %s" % out_dir
        )
    if not reference_manifest.is_file() or not reference_results.is_file():
        raise FileNotFoundError("frozen manifest/results inputs are required")

    captured: Dict[Tuple[str, str, float], Mapping[str, Any]] = {}
    original_evaluate_cell = evaluator.evaluate_cell

    def capture_evaluate_cell(*positional, **keywords):
        data = original_evaluate_cell(*positional, **keywords)
        factory = keywords.get("factory", positional[0] if positional else None)
        width = float(
            keywords.get("width_hz", positional[1] if len(positional) > 1 else np.nan)
        )
        shape = str(
            keywords.get("shape", positional[2] if len(positional) > 2 else "unknown")
        )
        regime = str(
            getattr(factory, "snr_mode", getattr(factory, "snr_mode_name", "unknown"))
        )
        key = (regime, shape, width)
        if key in captured:
            raise ValueError("duplicate captured Test-A cell %r" % (key,))
        captured[key] = data
        return data

    evaluator.evaluate_cell = capture_evaluate_cell
    evaluator.main(args)

    regenerated_manifest = out_dir / "test_manifest.jsonl"
    regenerated_results = out_dir / "stage4_width_results.csv"
    if sha256_file(str(regenerated_manifest)) != sha256_file(str(reference_manifest)):
        raise RuntimeError(
            "regenerated manifest differs from frozen Test A; no paired reanalysis is permitted"
        )
    frozen_rows = csv_index(str(reference_results))
    regenerated_rows = csv_index(str(regenerated_results))
    if set(frozen_rows) != set(regenerated_rows):
        raise RuntimeError("regenerated Test-A cells differ from the frozen CSV")
    auc_columns = ["%s_auc" % method for method in evaluator.METHODS]
    discrepancies = []
    for key in sorted(frozen_rows):
        for column in auc_columns:
            difference = abs(
                float(frozen_rows[key][column]) - float(regenerated_rows[key][column])
            )
            if difference > custom.aggregate_tolerance:
                discrepancies.append(
                    {
                        "cell": list(key),
                        "metric": column,
                        "absolute_difference": difference,
                    }
                )
    if discrepancies:
        raise RuntimeError(
            "regenerated aggregate metrics do not reproduce frozen Test A within tolerance: %s"
            % discrepancies[:5]
        )

    pair_lines = []
    for (regime, shape, width), data in sorted(captured.items()):
        labels = np.asarray(data["labels"], dtype=np.int8)
        cases = np.asarray(data["cases"])
        scores = data["scores"]
        for index, (label, case) in enumerate(zip(labels, cases)):
            record = {
                "dataset_role": "frozen_synthetic_test_a_posthoc_pair_export",
                "snr_mode": regime,
                "shape": shape,
                "width_hz": width,
                "cell_order": index,
                "label": int(label),
                "case": str(case),
            }
            for method in evaluator.METHODS:
                record[method + "_score"] = float(scores[method][index])
            pair_lines.append(json.dumps(record, sort_keys=True))
    atomic_write_text(custom.pair_output, "\n".join(pair_lines) + "\n")

    comparisons = {}
    for regime in sorted({key[0] for key in captured}):
        selected = [
            (key, value)
            for key, value in captured.items()
            if key[0] == regime
            and args.primary_width_min <= key[2] <= args.primary_width_max
        ]
        labels = np.concatenate([value["labels"] for _, value in selected])
        scores = {
            method: np.concatenate([value["scores"][method] for _, value in selected])
            for method in evaluator.METHODS
        }
        strata = np.concatenate(
            [
                np.asarray(
                    [
                        "%s|%.9g|%d|%s" % (key[1], key[2], int(label), case)
                        for label, case in zip(value["labels"], value["cases"])
                    ]
                )
                for key, value in selected
            ]
        )
        boot = evaluator.paired_stratified_bootstrap(
            labels,
            scores,
            strata,
            n_boot=args.n_boot,
            ci_level=args.ci_level,
            # Reuse the evaluator's frozen pooled bootstrap seeds so the direct
            # comparison is evaluated on the identical paired resamples.
            seed=args.seed + (700_001 if regime == "detected" else 800_011),
            reference="matched_filter",
        )
        comparisons[regime] = {
            "n_pairs": int(labels.size),
            "width_interval_hz": [args.primary_width_min, args.primary_width_max],
            "methods": boot["methods"],
            "stage4_minus_matched_filter": boot["deltas"]["stage4"],
            "bootstrap": {
                "replicates": args.n_boot,
                "ci_level": args.ci_level,
                "paired_and_stratified_by": "shape|width|label|negative_case",
            },
        }
    payload = {
        "format_version": 1,
        "analysis_role": "locked_posthoc_incremental_value_analysis",
        "model_or_threshold_tuning_performed": False,
        "manifest_exactly_reproduced": True,
        "aggregate_auc_tolerance": custom.aggregate_tolerance,
        "aggregate_auc_discrepancies": discrepancies,
        "frozen_manifest": str(reference_manifest),
        "frozen_manifest_sha256": sha256_file(str(reference_manifest)),
        "frozen_results_csv": str(reference_results),
        "frozen_results_csv_sha256": sha256_file(str(reference_results)),
        "regenerated_manifest": str(regenerated_manifest),
        "regenerated_manifest_sha256": sha256_file(str(regenerated_manifest)),
        "pair_scores": str(Path(custom.pair_output).resolve()),
        "pair_scores_sha256": sha256_file(custom.pair_output),
        "comparisons": comparisons,
        "interpretation": (
            "This quantifies incremental Stage-4 ranking value beyond the transparent "
            "downstream filter on the already-frozen synthetic Test A. It is not a "
            "comparison against BLISS as the upstream detector."
        ),
    }
    write_json(custom.direct_output, payload)
    print("Pair export and direct Stage-4-minus-filter intervals written successfully")


if __name__ == "__main__":
    main()
