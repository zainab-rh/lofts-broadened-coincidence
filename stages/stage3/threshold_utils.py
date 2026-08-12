"""Decision-threshold utilities for Stage 3 coincidence scores.

Lower latent distances indicate a cross-station match.  These functions keep
threshold calibration separate from model training and operate on the
case-stratified distance arrays produced by :mod:`train`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def split_match_mismatch(
    raw_distances: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine case-level distances into matched and mismatched populations.

    Case names ending in ``"_match"`` are treated as positive examples; all
    other cases are treated as mismatches.  This follows the naming convention
    used by the Stage 3 data generator.
    """

    match_parts = []
    mismatch_parts = []
    for case, values in raw_distances.items():
        arr = np.asarray(values, dtype=float).reshape(-1)
        if case.endswith("_match"):
            match_parts.append(arr)
        else:
            mismatch_parts.append(arr)

    match = np.concatenate(match_parts) if match_parts else np.array([], dtype=float)
    mismatch = (
        np.concatenate(mismatch_parts) if mismatch_parts else np.array([], dtype=float)
    )
    return match, mismatch


def metrics_at_threshold(
    raw_distances: Mapping[str, Sequence[float]], threshold: float
) -> dict[str, float | int]:
    """Calculate binary classification metrics at one distance threshold."""

    match_d, mismatch_d = split_match_mismatch(raw_distances)
    tp = int(np.sum(match_d < threshold))
    fn = int(np.sum(match_d >= threshold))
    fp = int(np.sum(mismatch_d < threshold))
    tn = int(np.sum(mismatch_d >= threshold))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(len(match_d) + len(mismatch_d), 1)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_match": int(len(match_d)),
        "n_mismatch": int(len(mismatch_d)),
    }


def per_case_match_rate(
    raw_distances: Mapping[str, Sequence[float]], threshold: float
) -> dict[str, float]:
    """Return the fraction of each case classified as a match."""

    return {
        case: (
            float(np.mean(np.asarray(values) < threshold))
            if len(values)
            else float("nan")
        )
        for case, values in raw_distances.items()
    }


def sweep_thresholds(
    raw_distances: Mapping[str, Sequence[float]], thresholds: Sequence[float]
) -> list[dict[str, float | int]]:
    """Evaluate a sequence of candidate decision thresholds."""

    return [metrics_at_threshold(raw_distances, float(value)) for value in thresholds]


def best_f1_threshold(
    raw_distances: Mapping[str, Sequence[float]],
    candidates: Sequence[float] | None = None,
    n_candidates: int = 60,
    max_threshold: float = 1.0,
) -> dict[str, object]:
    """Select the threshold with the highest F1 score.

    Equal-F1 candidates are resolved in favour of the smaller threshold, which
    is the more conservative operating point for follow-up selection.
    """

    if candidates is None:
        if n_candidates <= 0 or max_threshold <= 0:
            raise ValueError("n_candidates and max_threshold must be positive")
        candidates = np.linspace(
            max_threshold / n_candidates, max_threshold, n_candidates
        )

    rows = sweep_thresholds(raw_distances, candidates)
    valid = [row for row in rows if not np.isnan(float(row["f1"]))]
    if valid:
        best = min(valid, key=lambda row: (-float(row["f1"]), float(row["threshold"])))
        result: dict[str, object] = dict(best)
    else:
        result = {
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }
    result["sweep"] = rows
    return result
