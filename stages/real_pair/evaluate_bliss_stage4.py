#!/usr/bin/env python3
"""Evaluate frozen Synthetic Test B with separate pipeline denominators.

The evaluator joins label-blind predictions to the post-inference truth file,
computes paired method comparisons on identical union entries, and reports:

1. BLISS-union entry rate over all injected positive events;
2. Stage-4 conditional coincidence performance among scored union entries;
3. end-to-end final recovery over all injected positive events.

Direct Stage-4-minus-filter intervals use the same bootstrap resample.  They
must not be approximated by subtracting independently reported confidence
intervals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from lofts_bliss_schema import (
    atomic_write_text,
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

METHOD_COLUMNS = {
    "Raw Stage 3": "stage3_raw_score",
    "Corrected Stage 3": "stage3_corrected_score",
    "Model-free filter": "matched_filter_score",
    "Stage 4": "stage4_score",
}


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    good = np.isfinite(scores)
    labels, scores = labels[good], scores[good]
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (np.sum(ranks[labels == 1]) - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, Any]:
    predicted = np.asarray(scores) >= float(threshold)
    labels = np.asarray(labels, dtype=np.int8)
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum((~predicted) & (labels == 1)))
    tn = int(np.sum((~predicted) & (labels == 0)))
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    specificity = tn / float(tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "accuracy": (tp + tn) / float(len(labels)) if len(labels) else float("nan"),
    }


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    result = {
        "auc_roc": auc_rank(labels, scores),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    for max_fpr in (0.01, 0.05, 0.10):
        try:
            result["standardized_partial_auc_max_fpr_%.2f" % max_fpr] = float(
                roc_auc_score(labels, scores, max_fpr=max_fpr)
            )
        except ValueError:
            result["standardized_partial_auc_max_fpr_%.2f" % max_fpr] = float("nan")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    for target in (0.90, 0.95, 0.99):
        valid = false_positive_rate[true_positive_rate >= target]
        result["minimum_fpr_at_tpr_%.2f" % target] = (
            float(np.min(valid)) if valid.size else float("nan")
        )
    return result


def _ece(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    order = np.argsort(probabilities)
    bins = [
        indices
        for indices in np.array_split(order, min(n_bins, len(order)))
        if len(indices)
    ]
    rows, ece = [], 0.0
    for index, indices in enumerate(bins):
        confidence = float(np.mean(probabilities[indices]))
        frequency = float(np.mean(labels[indices]))
        weight = len(indices) / float(len(labels))
        ece += weight * abs(confidence - frequency)
        rows.append(
            {
                "bin": index + 1,
                "n": len(indices),
                "mean_score": confidence,
                "observed_match_frequency": frequency,
            }
        )
    return {
        "equal_count_ece": float(ece),
        "n_bins": len(rows),
        "bins": rows,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "warning": "Stage-4 sigmoid outputs are ranking scores; calibration is empirical, not astrophysical.",
    }


def _stratified_indices(labels, cases, states, rng) -> np.ndarray:
    strata = np.asarray(
        [
            "%d|%s|%s" % (label, case, state)
            for label, case, state in zip(labels, cases, states)
        ]
    )
    parts = []
    for key in sorted(set(strata)):
        indices = np.flatnonzero(strata == key)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(parts)


def _cluster_indices(groups, rng) -> np.ndarray:
    unique = np.asarray(sorted(set(groups)))
    sampled = rng.choice(unique, size=len(unique), replace=True)
    parts = [np.flatnonzero(np.asarray(groups) == group) for group in sampled]
    return np.concatenate(parts)


def paired_bootstrap(
    labels: np.ndarray,
    score_map: Mapping[str, np.ndarray],
    cases: Sequence[str],
    states: Sequence[str],
    groups: Sequence[str],
    n_boot: int,
    ci_level: float,
    seed: int,
    bootstrap_unit: str,
) -> Dict[str, Any]:
    if not (0.0 < ci_level < 1.0) or n_boot <= 0:
        raise ValueError("invalid bootstrap configuration")
    n_groups = len(set(groups))
    if bootstrap_unit == "auto":
        unit = "simultaneous_group" if n_groups >= 5 else "pair_stratified"
    else:
        unit = bootstrap_unit
    if unit == "simultaneous_group" and n_groups < 2:
        raise ValueError("cluster bootstrap requires at least two simultaneous groups")
    rng = np.random.default_rng(seed)
    method_draws = {key: [] for key in score_map}
    delta_draws = {
        "stage4_minus_raw": [],
        "stage4_minus_corrected": [],
        "stage4_minus_filter": [],
    }
    attempts = 0
    while len(method_draws["Stage 4"]) < n_boot:
        attempts += 1
        if attempts > n_boot * 20:
            raise RuntimeError("too many invalid bootstrap samples")
        indices = (
            _cluster_indices(groups, rng)
            if unit == "simultaneous_group"
            else _stratified_indices(labels, cases, states, rng)
        )
        sampled_labels = labels[indices]
        if len(set(sampled_labels.tolist())) < 2:
            continue
        aucs = {
            key: auc_rank(sampled_labels, values[indices])
            for key, values in score_map.items()
        }
        if not all(np.isfinite(value) for value in aucs.values()):
            continue
        for key, value in aucs.items():
            method_draws[key].append(value)
        delta_draws["stage4_minus_raw"].append(aucs["Stage 4"] - aucs["Raw Stage 3"])
        delta_draws["stage4_minus_corrected"].append(
            aucs["Stage 4"] - aucs["Corrected Stage 3"]
        )
        delta_draws["stage4_minus_filter"].append(
            aucs["Stage 4"] - aucs["Model-free filter"]
        )
    alpha = 0.5 * (1.0 - ci_level)

    def interval(values):
        array = np.asarray(values)
        return {
            "ci_lo": float(np.quantile(array, alpha)),
            "ci_hi": float(np.quantile(array, 1.0 - alpha)),
        }

    methods = {}
    for key, values in score_map.items():
        methods[key] = {"auc": auc_rank(labels, values), **interval(method_draws[key])}
    deltas = {}
    point = {
        "stage4_minus_raw": methods["Stage 4"]["auc"] - methods["Raw Stage 3"]["auc"],
        "stage4_minus_corrected": methods["Stage 4"]["auc"]
        - methods["Corrected Stage 3"]["auc"],
        "stage4_minus_filter": methods["Stage 4"]["auc"]
        - methods["Model-free filter"]["auc"],
    }
    for key, draws in delta_draws.items():
        deltas[key] = {"delta_auc": float(point[key]), **interval(draws)}
    return {
        "methods": methods,
        "paired_deltas": deltas,
        "bootstrap_unit": unit,
        "n_simultaneous_groups": n_groups,
        "n_boot": n_boot,
        "ci_level": ci_level,
        "conditional_scope_warning": (
            None
            if unit == "simultaneous_group"
            else "Pair-stratified intervals are conditional on the supplied background groups."
        ),
    }


def _join(predictions, labels):
    label_by_id = {str(item["pair_id"]): item for item in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("label pair IDs must be unique")
    joined = []
    seen = set()
    for prediction in predictions:
        pair_id = str(prediction["pair_id"])
        if pair_id in seen:
            raise ValueError("prediction pair IDs must be unique")
        seen.add(pair_id)
        if pair_id not in label_by_id:
            raise ValueError("prediction %s has no Synthetic-Test-B label" % pair_id)
        joined.append(
            {
                **prediction,
                **{
                    "truth_" + key: value for key, value in label_by_id[pair_id].items()
                },
            }
        )
    return joined, len(labels) - len(joined)


def _pipeline_denominators(
    joined, events, width_min, width_max, population="detected_conditioned"
):
    event_by_id = {str(item["event_id"]): item for item in events}
    positive_events = {
        key: value
        for key, value in event_by_id.items()
        if int(value["pair_label"]) == 1
        and str(value.get("population", population)) == population
        and width_min <= float(value["width_hz"]) <= width_max
    }
    entered = {
        key
        for key, value in positive_events.items()
        if value["entered_candidate_union"]
    }
    scored = set()
    correct = set()
    for row in joined:
        event_id = row.get("truth_event_id")
        if event_id in positive_events and int(row["truth_label"]) == 1:
            scored.add(event_id)
            if bool(row["stage4_predicted_match"]):
                correct.add(event_id)
    impossible_scored = scored - entered
    if impossible_scored:
        raise ValueError(
            "positive events were scored despite being marked absent from the BLISS union: %s"
            % sorted(impossible_scored)
        )
    if not correct <= scored <= entered <= set(positive_events):
        raise AssertionError("pipeline denominator sets are not properly nested")
    total = len(positive_events)
    return {
        "width_interval_hz": [float(width_min), float(width_max)],
        "n_positive_injected_events": total,
        "n_entering_bliss_union": len(entered),
        "n_scored_by_stage4": len(scored),
        "n_final_correct_match_decisions": len(correct),
        "bliss_union_entry_fraction": (
            len(entered) / float(total) if total else float("nan")
        ),
        "post_union_extraction_and_scoring_fraction": (
            len(scored) / float(len(entered)) if entered else float("nan")
        ),
        "conditional_stage4_event_recall_among_scored": (
            len(correct) / float(len(scored)) if scored else float("nan")
        ),
        "post_union_pipeline_recovery_fraction": (
            len(correct) / float(len(entered)) if entered else float("nan")
        ),
        "end_to_end_final_recovery_fraction": (
            len(correct) / float(total) if total else float("nan")
        ),
        "denominator_note": (
            "BLISS misses, extraction exclusions and Stage-4 errors remain separate stages."
        ),
    }


def _negative_case_results(joined, stage4_threshold):
    positives = [row for row in joined if int(row["truth_label"]) == 1]
    cases = sorted(
        {str(row["truth_case"]) for row in joined if int(row["truth_label"]) == 0}
    )
    output = []
    for case in cases:
        negatives = [
            row
            for row in joined
            if int(row["truth_label"]) == 0 and str(row["truth_case"]) == case
        ]
        subset = positives + negatives
        labels = np.asarray([int(row["truth_label"]) for row in subset])
        scores = np.asarray([float(row["stage4_score"]) for row in subset])
        operating = _threshold_metrics(labels, scores, stage4_threshold)
        output.append(
            {
                "negative_case": case,
                "n_positive": len(positives),
                "n_negative": len(negatives),
                "auc_roc": (
                    auc_rank(labels, scores)
                    if positives and negatives
                    else float("nan")
                ),
                "false_positive_rate_at_frozen_threshold": operating[
                    "false_positive_rate"
                ],
                "n_false_positive": operating["fp"],
            }
        )
    return output


def _prevalence_projection(operating, prevalences=(0.001, 0.01, 0.1)):
    tpr, fpr = operating["recall"], operating["false_positive_rate"]
    rows = []
    for prevalence in prevalences:
        denominator = prevalence * tpr + (1.0 - prevalence) * fpr
        rows.append(
            {
                "assumed_true_match_prevalence": prevalence,
                "projected_precision": (
                    prevalence * tpr / denominator if denominator else float("nan")
                ),
                "assumption": "constant observed TPR/FPR under prevalence shift",
            }
        )
    return rows


def _conditional_subgroups(joined, stage4_threshold):
    """Report operational subgroups without selecting a new threshold."""

    definitions = {
        "detection_state": lambda row: str(row["truth_detection_state"]),
        "evaluation_cell_id": lambda row: str(
            row.get("truth_evaluation_cell_id") or "unlinked_false_hit"
        ),
        "injected_profile": lambda row: str(
            row.get("truth_injected_shape") or "unlinked_false_hit"
        ),
        "recovered_width_band_hz": lambda row: (
            "10-<20"
            if float(row["anchor_width_hz"]) < 20
            else (
                "20-<40"
                if float(row["anchor_width_hz"]) < 40
                else "40-<75" if float(row["anchor_width_hz"]) < 75 else "75-100+"
            )
        ),
        "recovered_snr_band": lambda row: (
            "<10"
            if float(row["anchor_snr"]) < 10
            else (
                "10-<12"
                if float(row["anchor_snr"]) < 12
                else "12-<20" if float(row["anchor_snr"]) < 20 else "20+"
            )
        ),
    }
    output: Dict[str, List[Dict[str, Any]]] = {}
    for name, key_function in definitions.items():
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in joined:
            grouped[key_function(row)].append(row)
        rows = []
        for key, subset in sorted(grouped.items()):
            labels = np.asarray(
                [int(row["truth_label"]) for row in subset], dtype=np.int8
            )
            scores = np.asarray([float(row["stage4_score"]) for row in subset])
            ranked = (
                auc_rank(labels, scores)
                if len(set(labels.tolist())) == 2
                else float("nan")
            )
            threshold = _threshold_metrics(labels, scores, stage4_threshold)
            rows.append(
                {
                    "stratum": key,
                    "n": len(subset),
                    "n_match": int(np.sum(labels == 1)),
                    "n_mismatch": int(np.sum(labels == 0)),
                    "stage4_auc_roc": ranked,
                    "recall_at_frozen_threshold": threshold["recall"],
                    "false_positive_rate_at_frozen_threshold": threshold[
                        "false_positive_rate"
                    ],
                }
            )
        output[name] = rows
    return output


def _plots(out_dir: Path, result, joined, calibration, negative_cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = result["paired_bootstrap"]["methods"]
    names = list(METHOD_COLUMNS)
    values = [methods[name]["auc"] for name in names]
    xerr = np.asarray(
        [
            [values[i] - methods[name]["ci_lo"] for i, name in enumerate(names)],
            [methods[name]["ci_hi"] - values[i] for i, name in enumerate(names)],
        ]
    )
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(names))
    axis.errorbar(values, y, xerr=xerr, fmt="o", capsize=4, color="#16697a")
    axis.axvline(0.5, linestyle="--", color="0.4", label="Chance")
    axis.axvline(
        result["success_criteria"]["target_auc"],
        linestyle=":",
        color="#2a9d55",
        label="Registered target",
    )
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    axis.set_xlabel("Candidate-union AUC-ROC (95% paired-bootstrap CI)")
    axis.set_title("Synthetic Test B: conditional post-BLISS ranking")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "01_test_b_endpoint_forest.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    denominators = result["pipeline_denominators"]
    counts = [
        denominators["n_positive_injected_events"],
        denominators["n_entering_bliss_union"],
        denominators["n_scored_by_stage4"],
        denominators["n_final_correct_match_decisions"],
    ]
    labels = [
        "Injected\npositive events",
        "Entered BLISS\nunion",
        "Stage-4\nscored",
        "Final correct\nmatch",
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    bars = axis.bar(labels, counts, color=["#457b9d", "#2a9d8f", "#f4a261", "#2a9d55"])
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Number of events")
    axis.set_title("End-to-end and conditional denominators remain distinct")
    fig.tight_layout()
    fig.savefig(
        out_dir / "02_test_b_pipeline_denominators.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    labels_array = np.asarray([int(row["truth_label"]) for row in joined])
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    for name, column in METHOD_COLUMNS.items():
        scores = np.asarray([float(row[column]) for row in joined])
        fpr, tpr, _ = roc_curve(labels_array, scores)
        axis.plot(
            fpr, tpr, label="%s (AUC %.3f)" % (name, auc_rank(labels_array, scores))
        )
    axis.set_xlim(0.0, min(0.20, max(0.02, axis.get_xlim()[1])))
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Low-false-positive operating region")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "03_test_b_low_fpr_roc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if negative_cases:
        fig, axis = plt.subplots(figsize=(9, max(3.5, 0.42 * len(negative_cases))))
        case_names = [row["negative_case"] for row in negative_cases]
        fprs = [
            row["false_positive_rate_at_frozen_threshold"] for row in negative_cases
        ]
        axis.barh(case_names, fprs, color="#e76f51")
        axis.set_xlabel("False-positive rate at frozen validation threshold")
        axis.set_title("Stage-4 rejection by union-negative case")
        axis.invert_yaxis()
        fig.tight_layout()
        fig.savefig(
            out_dir / "04_test_b_negative_cases.png", dpi=220, bbox_inches="tight"
        )
        plt.close(fig)

    bins = calibration["bins"]
    if bins:
        fig, axis = plt.subplots(figsize=(5.5, 5.2))
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        axis.plot(
            [row["mean_score"] for row in bins],
            [row["observed_match_frequency"] for row in bins],
            marker="o",
            color="#6a4c93",
        )
        axis.set_xlabel("Mean Stage-4 score")
        axis.set_ylabel("Observed match frequency")
        axis.set_title("Conditional calibration (natural Test-B prevalence)")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / "05_test_b_reliability.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    preregistration = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    if preregistration.get("status") != "frozen_before_test_b":
        raise ValueError("Test-B preregistration is not frozen")
    preregistered_population = preregistration.get("locked_inputs", {}).get(
        "population"
    )
    if preregistered_population != args.primary_population:
        raise ValueError(
            "primary population %r disagrees with preregistration %r"
            % (args.primary_population, preregistered_population)
        )
    predictions = read_json_records(args.predictions)
    labels = read_json_records(args.labels)
    events = read_json_records(args.events)
    joined, n_unscored_labels = _join(predictions, labels)
    if not joined:
        raise ValueError("no scored union entries")
    label_array = np.asarray([int(row["truth_label"]) for row in joined], dtype=np.int8)
    populations = sorted({str(row.get("truth_population", "")) for row in joined})
    if populations != [args.primary_population]:
        raise ValueError(
            "joined populations %s do not equal registered primary population %r"
            % (populations, args.primary_population)
        )
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError(
            "conditional evaluation requires both match and mismatch labels"
        )
    score_map = {
        name: np.asarray([float(row[column]) for row in joined], dtype=float)
        for name, column in METHOD_COLUMNS.items()
    }
    cases = [str(row["truth_case"]) for row in joined]
    states = [str(row["truth_detection_state"]) for row in joined]
    groups = [
        str(row.get("resampling_block_id") or row["simultaneous_group_id"])
        for row in joined
    ]
    bootstrap = paired_bootstrap(
        label_array,
        score_map,
        cases,
        states,
        groups,
        args.n_boot,
        args.ci_level,
        args.seed,
        args.bootstrap_unit,
    )
    ranking = {
        name: _ranking_metrics(label_array, scores)
        for name, scores in score_map.items()
    }
    stage4_thresholds = {float(row["stage4_validation_threshold"]) for row in joined}
    if len(stage4_thresholds) != 1:
        raise ValueError("predictions contain inconsistent Stage-4 thresholds")
    stage4_threshold = next(iter(stage4_thresholds))
    operating = {
        "Raw Stage 3": _threshold_metrics(
            label_array, score_map["Raw Stage 3"], -float(args.stage3_margin)
        ),
        "Stage 4": _threshold_metrics(
            label_array, score_map["Stage 4"], stage4_threshold
        ),
    }
    calibration = _ece(label_array, score_map["Stage 4"], args.calibration_bins)
    negative_cases = _negative_case_results(joined, stage4_threshold)
    conditional_subgroups = _conditional_subgroups(joined, stage4_threshold)
    denominators = _pipeline_denominators(
        joined, events, args.width_min, args.width_max, args.primary_population
    )
    primary = bootstrap["methods"]["Stage 4"]
    delta_raw = bootstrap["paired_deltas"]["stage4_minus_raw"]
    success = {
        "target_auc": float(args.target_auc),
        "stage4_auc_at_least_target": bool(primary["auc"] >= args.target_auc),
        "paired_delta_vs_raw_lower_ci_above_zero": bool(delta_raw["ci_lo"] > 0),
        "synthetic_test_b_conditional_success": bool(
            primary["auc"] >= args.target_auc and delta_raw["ci_lo"] > 0
        ),
        "stage4_minus_filter_is_supporting_not_required": True,
    }
    result = {
        "format_version": 1,
        "test_name": "Synthetic Test B: actual BLISS-recovered metadata on independent synthetic injections",
        "n_scored_union_entries": len(joined),
        "n_unscored_union_labels": n_unscored_labels,
        "natural_match_prevalence": float(np.mean(label_array)),
        "primary_population": args.primary_population,
        "paired_bootstrap": bootstrap,
        "ranking_metrics": ranking,
        "frozen_operating_points": operating,
        "stage4_calibration": calibration,
        "negative_case_results": negative_cases,
        "conditional_subgroup_results": conditional_subgroups,
        "prevalence_projections": _prevalence_projection(operating["Stage 4"]),
        "pipeline_denominators": denominators,
        "success_criteria": success,
        "inputs": {
            "predictions": str(Path(args.predictions).resolve()),
            "predictions_sha256": sha256_file(args.predictions),
            "labels": str(Path(args.labels).resolve()),
            "labels_sha256": sha256_file(args.labels),
            "events": str(Path(args.events).resolve()),
            "events_sha256": sha256_file(args.events),
            "preregistration": str(Path(args.preregistration).resolve()),
            "preregistration_sha256": sha256_file(args.preregistration),
        },
        "interpretation_boundary": (
            "This is an end-to-end synthetic injection through actual BLISS recovery and "
            "frozen Stage-4 coincidence on the supplied backgrounds. It is not real "
            "Ireland-Sweden candidate validation."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_dir / "synthetic_test_b_pair_level_results.jsonl"), joined)
    write_json(str(out_dir / "synthetic_test_b_evaluation.json"), result)
    _plots(out_dir, result, joined, calibration, negative_cases)
    report = (
        "Synthetic Test B — frozen BLISS-to-Stage-4 evaluation\n"
        "=====================================================\n"
        "Scored candidate-union entries: %d\n"
        "Stage-4 conditional AUC-ROC: %.6f [%.6f, %.6f]\n"
        "Paired ΔAUC vs raw Stage 3: %.6f [%.6f, %.6f]\n"
        "Paired ΔAUC vs model-free filter: %.6f [%.6f, %.6f]\n"
        "BLISS-union entry fraction (positive injected events): %.6f\n"
        "Post-union extraction/scoring fraction: %.6f\n"
        "Conditional Stage-4 event recall among scored events: %.6f\n"
        "Post-union pipeline recovery fraction: %.6f\n"
        "End-to-end final recovery fraction: %.6f\n"
        "Registered conditional criterion: %s\n\n"
        "%s\n"
        % (
            len(joined),
            primary["auc"],
            primary["ci_lo"],
            primary["ci_hi"],
            delta_raw["delta_auc"],
            delta_raw["ci_lo"],
            delta_raw["ci_hi"],
            bootstrap["paired_deltas"]["stage4_minus_filter"]["delta_auc"],
            bootstrap["paired_deltas"]["stage4_minus_filter"]["ci_lo"],
            bootstrap["paired_deltas"]["stage4_minus_filter"]["ci_hi"],
            denominators["bliss_union_entry_fraction"],
            denominators["post_union_extraction_and_scoring_fraction"],
            denominators["conditional_stage4_event_recall_among_scored"],
            denominators["post_union_pipeline_recovery_fraction"],
            denominators["end_to_end_final_recovery_fraction"],
            "PASS" if success["synthetic_test_b_conditional_success"] else "FAIL",
            result["interpretation_boundary"],
        )
    )
    atomic_write_text(str(out_dir / "synthetic_test_b_report.txt"), report)
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate locked Synthetic Test B with paired and pipeline-aware statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--bootstrap-unit",
        choices=("auto", "pair_stratified", "simultaneous_group"),
        default="auto",
    )
    parser.add_argument("--stage3-margin", type=float, default=0.1833)
    parser.add_argument("--target-auc", type=float, default=0.80)
    parser.add_argument("--width-min", type=float, default=10.0)
    parser.add_argument("--width-max", type=float, default=100.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--primary-population",
        choices=("detected_conditioned", "fixed_power"),
        default="detected_conditioned",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
