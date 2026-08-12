#!/usr/bin/env python3
"""Measure empirical BLISS recovery errors and draft association tolerances.

This program joins *recovered* candidate records to a separately stored
injection-truth registry.  Exact recovery identifiers are preferred.  If they
are unavailable, a broad, pre-declared Hungarian matching gate may be used.
The resulting policy is marked ``empirical_draft`` and must be reviewed and
frozen before a locked candidate-union evaluation.

Frequency residuals are evaluated at the injection record's declared
frequency epoch.  This prevents a drift error from being misreported as a
frequency error merely because the two tables use different reference times.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from lofts_bliss_schema import (
    CandidateRecord,
    InjectionTruthRecord,
    atomic_write_text,
    load_candidates,
    load_truth,
    read_json_records,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)
from scipy.optimize import linear_sum_assignment


def _load_recovery_links(path: Optional[str]) -> Dict[str, str]:
    if path is None:
        return {}
    result: Dict[str, str] = {}
    for row in read_json_records(path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        injection_id = str(row.get("recovery_link_id", "")).strip()
        if not candidate_id or not injection_id:
            raise ValueError(
                "recovery-link rows require candidate_id and recovery_link_id"
            )
        if candidate_id in result:
            raise ValueError("duplicate recovery link for candidate %s" % candidate_id)
        result[candidate_id] = injection_id
    return result


def _truth_frequency_at_candidate_epoch(
    truth: InjectionTruthRecord, candidate: CandidateRecord
) -> Tuple[float, float]:
    """Return candidate and truth frequencies at the truth reference epoch."""

    if truth.frequency_ref_mjd is not None:
        if candidate.frequency_ref_mjd is None:
            raise ValueError("candidate/truth time-reference modes disagree")
        target = float(truth.frequency_ref_mjd)
        return candidate.frequency_at_mjd(target), float(truth.frequency_hz)
    if candidate.frequency_ref_offset_s is None:
        raise ValueError("candidate/truth time-reference modes disagree")
    target = float(truth.frequency_ref_offset_s)
    return candidate.frequency_at_offset_s(target), float(truth.frequency_hz)


def _residuals(
    candidate: CandidateRecord, truth: InjectionTruthRecord
) -> Dict[str, float]:
    recovered_frequency, injected_frequency = _truth_frequency_at_candidate_epoch(
        truth, candidate
    )
    return {
        "frequency_residual_hz": recovered_frequency - injected_frequency,
        "drift_residual_hz_s": float(candidate.drift_hz_s) - float(truth.drift_hz_s),
        "log_width_residual": math.log(
            float(candidate.width_hz) / float(truth.width_hz)
        ),
        "snr_residual": float(candidate.snr) - float(truth.snr),
    }


def _matching_cost(
    candidate: CandidateRecord,
    truth: InjectionTruthRecord,
    gates: Mapping[str, float],
) -> float:
    try:
        residual = _residuals(candidate, truth)
    except ValueError:
        return float("inf")
    components = (
        abs(residual["frequency_residual_hz"]) / gates["frequency_hz"],
        abs(residual["drift_residual_hz_s"]) / gates["drift_hz_s"],
        abs(residual["log_width_residual"]) / gates["log_width"],
    )
    if any(value > 1.0 for value in components):
        return float("inf")
    return float(sum(value * value for value in components))


def _exact_link_matches(
    candidates: Sequence[CandidateRecord],
    truths: Sequence[InjectionTruthRecord],
    recovery_links: Mapping[str, str],
) -> Tuple[List[Tuple[CandidateRecord, InjectionTruthRecord]], List[str], List[str]]:
    truth_by_id = {item.injection_id: item for item in truths}
    grouped: Dict[str, List[CandidateRecord]] = defaultdict(list)
    unlinked_candidates: List[str] = []
    for candidate in candidates:
        link = str(recovery_links.get(candidate.candidate_id, "")).strip()
        if link:
            grouped[link].append(candidate)
        else:
            unlinked_candidates.append(candidate.candidate_id)
    matches: List[Tuple[CandidateRecord, InjectionTruthRecord]] = []
    used_candidates = set()
    for injection_id, choices in grouped.items():
        if injection_id not in truth_by_id:
            unlinked_candidates.extend(item.candidate_id for item in choices)
            continue
        truth = truth_by_id[injection_id]
        valid = []
        for candidate in choices:
            try:
                residual = _residuals(candidate, truth)
            except ValueError:
                continue
            cost = (
                abs(residual["frequency_residual_hz"])
                + abs(residual["drift_residual_hz_s"])
                + abs(residual["log_width_residual"])
            )
            valid.append((cost, -candidate.snr, candidate.candidate_id, candidate))
        if valid:
            candidate = min(valid)[-1]
            matches.append((candidate, truth))
            used_candidates.add(candidate.candidate_id)
        unlinked_candidates.extend(
            item.candidate_id
            for item in choices
            if item.candidate_id not in used_candidates
        )
    matched_truth = {truth.injection_id for _, truth in matches}
    missed_truth = [
        item.injection_id for item in truths if item.injection_id not in matched_truth
    ]
    return matches, sorted(set(unlinked_candidates)), missed_truth


def _hungarian_matches(
    candidates: Sequence[CandidateRecord],
    truths: Sequence[InjectionTruthRecord],
    gates: Mapping[str, float],
) -> Tuple[List[Tuple[CandidateRecord, InjectionTruthRecord]], List[str], List[str]]:
    matches: List[Tuple[CandidateRecord, InjectionTruthRecord]] = []
    used_candidates = set()
    used_truth = set()
    keys = sorted(
        set((item.observation_id, item.station_id) for item in candidates)
        | set((item.observation_id, item.station_id) for item in truths)
    )
    for key in keys:
        local_candidates = [
            item for item in candidates if (item.observation_id, item.station_id) == key
        ]
        local_truth = [
            item for item in truths if (item.observation_id, item.station_id) == key
        ]
        if not local_candidates or not local_truth:
            continue
        costs = np.full((len(local_candidates), len(local_truth)), np.inf, dtype=float)
        for i, candidate in enumerate(local_candidates):
            for j, truth in enumerate(local_truth):
                costs[i, j] = _matching_cost(candidate, truth, gates)
        finite = np.isfinite(costs)
        if not finite.any():
            continue
        safe = np.where(finite, costs, 1e12)
        rows, cols = linear_sum_assignment(safe)
        for i, j in zip(rows, cols):
            if not np.isfinite(costs[i, j]):
                continue
            candidate, truth = local_candidates[i], local_truth[j]
            matches.append((candidate, truth))
            used_candidates.add(candidate.candidate_id)
            used_truth.add(truth.injection_id)
    unmatched_candidates = [
        item.candidate_id
        for item in candidates
        if item.candidate_id not in used_candidates
    ]
    missed_truth = [
        item.injection_id for item in truths if item.injection_id not in used_truth
    ]
    return matches, unmatched_candidates, missed_truth


def _describe(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {
            key: float("nan")
            for key in (
                "n",
                "mean",
                "median",
                "std",
                "mad_sigma",
                "q68_abs",
                "q90_abs",
                "q95_abs",
                "q99_abs",
            )
        }
    median = float(np.median(x))
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": median,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "mad_sigma": 1.482602218505602 * float(np.median(np.abs(x - median))),
        "q68_abs": float(np.quantile(np.abs(x), 0.68)),
        "q90_abs": float(np.quantile(np.abs(x), 0.90)),
        "q95_abs": float(np.quantile(np.abs(x), 0.95)),
        "q99_abs": float(np.quantile(np.abs(x), 0.99)),
    }


def _nearest_bin(value: float, centres: Sequence[float]) -> float:
    return float(min(centres, key=lambda item: abs(float(item) - float(value))))


def _bin_summary(
    truths: Sequence[InjectionTruthRecord],
    matched_truth_ids: set,
    width_centres: Sequence[float],
    snr_edges: Sequence[float],
) -> List[Dict[str, Any]]:
    cells: Dict[Tuple[float, str], List[InjectionTruthRecord]] = defaultdict(list)
    for truth in truths:
        width_bin = _nearest_bin(truth.width_hz, width_centres)
        snr_index = int(
            np.searchsorted(np.asarray(snr_edges), truth.snr, side="right") - 1
        )
        if snr_index < 0:
            snr_label = "<%.3g" % snr_edges[0]
        elif snr_index >= len(snr_edges) - 1:
            snr_label = ">=%.3g" % snr_edges[-1]
        else:
            snr_label = "%.3g-%.3g" % (snr_edges[snr_index], snr_edges[snr_index + 1])
        cells[(width_bin, snr_label)].append(truth)
    rows = []
    for (width_bin, snr_label), items in sorted(cells.items()):
        recovered = sum(item.injection_id in matched_truth_ids for item in items)
        rows.append(
            {
                "width_bin_hz": width_bin,
                "snr_bin": snr_label,
                "n_injected": len(items),
                "n_recovered": int(recovered),
                "recovery_fraction": recovered / float(len(items)),
            }
        )
    return rows


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _policy_tolerance(
    values: Sequence[float], quantile: float, multiplier: float, floor: float
) -> float:
    if not values:
        raise ValueError("cannot derive a policy from zero matched recoveries")
    return max(float(floor), multiplier * float(np.quantile(np.abs(values), quantile)))


def main(args: argparse.Namespace) -> None:
    if not (0.5 < args.policy_quantile < 1.0):
        raise ValueError("policy_quantile must lie in (0.5, 1)")
    if args.policy_multiplier <= 0:
        raise ValueError("policy_multiplier must be positive")
    candidates = load_candidates(args.candidates)
    truths = load_truth(args.truth)
    recovery_links = _load_recovery_links(args.recovery_links)
    if not candidates or not truths:
        raise ValueError("candidate and truth inputs must both be non-empty")

    unknown_link_candidates = sorted(
        set(recovery_links) - {item.candidate_id for item in candidates}
    )
    if unknown_link_candidates:
        raise ValueError(
            "recovery-link sidecar references unknown candidate IDs: %s"
            % unknown_link_candidates[:5]
        )
    has_any_links = any(item.candidate_id in recovery_links for item in candidates)
    if args.matching == "exact" or (args.matching == "auto" and has_any_links):
        if not has_any_links:
            raise ValueError(
                "exact matching requested but the recovery-link sidecar is empty"
            )
        matches, false_hits, missed = _exact_link_matches(
            candidates, truths, recovery_links
        )
        method = "exact_recovery_link_id"
    else:
        gates = {
            "frequency_hz": float(args.match_frequency_hz),
            "drift_hz_s": float(args.match_drift_hz_s),
            "log_width": float(args.match_log_width),
        }
        if any(value <= 0 for value in gates.values()):
            raise ValueError("all Hungarian matching gates must be positive")
        matches, false_hits, missed = _hungarian_matches(candidates, truths, gates)
        method = "hungarian_predeclared_gate"

    if len(matches) < args.min_policy_matches and not args.allow_small_sample_policy:
        raise ValueError(
            "only %d recoveries were matched; at least %d are required to draft an "
            "empirical policy (use --allow-small-sample-policy only for smoke tests)"
            % (len(matches), args.min_policy_matches)
        )

    residual_rows: List[Dict[str, Any]] = []
    residual_vectors: Dict[str, List[float]] = defaultdict(list)
    for candidate, truth in matches:
        values = _residuals(candidate, truth)
        residual_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "injection_id": truth.injection_id,
                "observation_id": truth.observation_id,
                "station_id": truth.station_id,
                "shape": truth.shape,
                "injected_width_hz": truth.width_hz,
                "injected_snr": truth.snr,
                **values,
            }
        )
        for key, value in values.items():
            residual_vectors[key].append(float(value))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(str(out_dir / "bliss_recovery_residuals.csv"), residual_rows)
    write_jsonl(
        str(out_dir / "candidate_recovery_links.matched.jsonl"),
        [
            {
                "candidate_id": candidate.candidate_id,
                "recovery_link_id": truth.injection_id,
            }
            for candidate, truth in matches
        ],
    )
    width_centres = [float(item) for item in args.widths.split(",") if item.strip()]
    snr_edges = [float(item) for item in args.snr_edges.split(",") if item.strip()]
    if not width_centres or len(snr_edges) < 2:
        raise ValueError(
            "widths must be non-empty and snr_edges must contain >=2 values"
        )
    bins = _bin_summary(
        truths,
        {truth.injection_id for _, truth in matches},
        width_centres,
        snr_edges,
    )
    _write_csv(str(out_dir / "bliss_detection_completeness.csv"), bins)

    residual_summary = {
        key: _describe(values) for key, values in residual_vectors.items()
    }
    summary = {
        "format_version": 1,
        "matching_method": method,
        "n_injections": len(truths),
        "n_recovered": len(matches),
        "n_missed_injections": len(missed),
        "n_unmatched_candidate_hits": len(false_hits),
        "conditional_recovery_fraction": len(matches) / float(len(truths)),
        "residuals": residual_summary,
        "missed_injection_ids": missed,
        "unmatched_candidate_ids": false_hits,
        "warning": (
            "This uncertainty estimate is conditional on the supplied backgrounds, "
            "injection distribution and BLISS configuration."
        ),
        "inputs": {
            "candidate_file": str(Path(args.candidates).resolve()),
            "candidate_sha256": sha256_file(args.candidates),
            "truth_file": str(Path(args.truth).resolve()),
            "truth_sha256": sha256_file(args.truth),
            "recovery_links_file": (
                str(Path(args.recovery_links).resolve())
                if args.recovery_links
                else None
            ),
            "recovery_links_sha256": (
                sha256_file(args.recovery_links) if args.recovery_links else None
            ),
        },
    }
    write_json(str(out_dir / "bliss_recovery_summary.json"), summary)

    policy = {
        "format_version": 1,
        "policy_id": stable_id(
            "policy", args.candidates, args.truth, len(matches), args.policy_quantile
        ),
        "status": "empirical_draft",
        "derivation": {
            "matching_method": method,
            "n_matched_recoveries": len(matches),
            "absolute_residual_quantile": args.policy_quantile,
            "pair_multiplier": args.policy_multiplier,
            "candidate_file": str(Path(args.candidates).resolve()),
            "truth_file": str(Path(args.truth).resolve()),
            "dataset_role": args.dataset_role,
        },
        "association_tolerances": {
            "frequency_hz": _policy_tolerance(
                residual_vectors["frequency_residual_hz"],
                args.policy_quantile,
                args.policy_multiplier,
                args.frequency_floor_hz,
            ),
            "drift_hz_s": _policy_tolerance(
                residual_vectors["drift_residual_hz_s"],
                args.policy_quantile,
                args.policy_multiplier,
                args.drift_floor_hz_s,
            ),
            "log_width": _policy_tolerance(
                residual_vectors["log_width_residual"],
                args.policy_quantile,
                args.policy_multiplier,
                args.log_width_floor,
            ),
        },
        "recovery_link_tolerances": {
            "frequency_hz": _policy_tolerance(
                residual_vectors["frequency_residual_hz"],
                args.policy_quantile,
                1.0,
                args.frequency_floor_hz,
            ),
            "drift_hz_s": _policy_tolerance(
                residual_vectors["drift_residual_hz_s"],
                args.policy_quantile,
                1.0,
                args.drift_floor_hz_s,
            ),
            "log_width": _policy_tolerance(
                residual_vectors["log_width_residual"],
                args.policy_quantile,
                1.0,
                args.log_width_floor,
            ),
        },
        "deduplication_tolerances": {
            "frequency_hz": max(
                args.frequency_floor_hz,
                _policy_tolerance(
                    residual_vectors["frequency_residual_hz"],
                    args.policy_quantile,
                    1.0,
                    args.frequency_floor_hz,
                ),
            ),
            "drift_hz_s": max(
                args.drift_floor_hz_s,
                _policy_tolerance(
                    residual_vectors["drift_residual_hz_s"],
                    args.policy_quantile,
                    1.0,
                    args.drift_floor_hz_s,
                ),
            ),
            "log_width": max(
                args.log_width_floor,
                _policy_tolerance(
                    residual_vectors["log_width_residual"],
                    args.policy_quantile,
                    1.0,
                    args.log_width_floor,
                ),
            ),
        },
        "review_required": True,
        "note": (
            "Association uses pair_multiplier times the single-station absolute-error "
            "quantile, a conservative triangle-inequality bound for two recovered "
            "stations. Recovery-link gates use the unmultiplied calibration quantile. "
            "Review against project BLISS conventions and freeze before Synthetic Test B; "
            "never tune these values on locked Test-B labels."
        ),
    }
    write_json(str(out_dir / "association_policy.empirical_draft.json"), policy)
    print(
        "Matched %d/%d injections; wrote empirical recovery audit and DRAFT policy to %s"
        % (len(matches), len(truths), out_dir)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate empirical BLISS metadata errors from an injection-recovery run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--recovery-links",
        default=None,
        help="segregated candidate-to-injection link sidecar from the injection harness",
    )
    parser.add_argument(
        "--dataset-role",
        required=True,
        choices=("calibration",),
        help="policy tolerances may be derived only on a calibration injection set",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--matching", choices=("auto", "exact", "hungarian"), default="auto"
    )
    parser.add_argument("--match-frequency-hz", type=float, default=25.0)
    parser.add_argument("--match-drift-hz-s", type=float, default=0.5)
    parser.add_argument("--match-log-width", type=float, default=0.7)
    parser.add_argument("--widths", default="10,20,30,50,75,100")
    parser.add_argument("--snr-edges", default="8,10,12,16,20,30")
    parser.add_argument("--policy-quantile", type=float, default=0.99)
    parser.add_argument("--policy-multiplier", type=float, default=2.0)
    parser.add_argument("--frequency-floor-hz", type=float, default=4.0)
    parser.add_argument("--drift-floor-hz-s", type=float, default=0.2)
    parser.add_argument("--log-width-floor", type=float, default=math.log(1.25))
    parser.add_argument("--min-policy-matches", type=int, default=100)
    parser.add_argument("--allow-small-sample-policy", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
