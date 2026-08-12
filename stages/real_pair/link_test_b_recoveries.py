#!/usr/bin/env python3
"""Create Synthetic-Test-B candidate-to-injection links after inference.

The preferred input is an exact link sidecar produced by the independent
injection/recovery harness.  If that facility is unavailable, this program
can instead apply the recovery-link tolerances frozen on the *calibration*
injections.  In either case it refuses to run until the label-blind prediction
file exists, so injected truth cannot influence union construction,
extraction, preprocessing, or model scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from candidate_union import load_policy
from estimate_bliss_recovery import _hungarian_matches
from lofts_bliss_schema import (
    load_candidates,
    load_truth,
    read_json_records,
    sha256_file,
    write_json,
    write_jsonl,
)

REQUIRED_TOLERANCES = {"frequency_hz", "drift_hz_s", "log_width"}


def _load_exact_links(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return []
    rows: List[Dict[str, str]] = []
    seen = set()
    for row in read_json_records(path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        injection_id = str(row.get("recovery_link_id", "")).strip()
        if not candidate_id or not injection_id:
            raise ValueError(
                "exact link rows require candidate_id and recovery_link_id"
            )
        key = (candidate_id, injection_id)
        if key in seen:
            raise ValueError("duplicate exact recovery link %r" % (key,))
        seen.add(key)
        rows.append({"candidate_id": candidate_id, "recovery_link_id": injection_id})
    return rows


def _validate_exact_links(rows, candidates, truths) -> Dict[str, Any]:
    candidate_by_id = {item.candidate_id: item for item in candidates}
    truth_by_id = {item.injection_id: item for item in truths}
    unknown_candidates = sorted(
        {row["candidate_id"] for row in rows} - set(candidate_by_id)
    )
    unknown_truth = sorted({row["recovery_link_id"] for row in rows} - set(truth_by_id))
    if unknown_candidates or unknown_truth:
        raise ValueError(
            "exact links contain unknown IDs (candidates=%s, injections=%s)"
            % (unknown_candidates[:5], unknown_truth[:5])
        )
    candidate_to_truth: Dict[str, str] = {}
    for row in rows:
        candidate_id, injection_id = row["candidate_id"], row["recovery_link_id"]
        prior = candidate_to_truth.get(candidate_id)
        if prior is not None and prior != injection_id:
            raise ValueError("candidate %s links to multiple injections" % candidate_id)
        candidate, truth = candidate_by_id[candidate_id], truth_by_id[injection_id]
        if (candidate.observation_id, candidate.station_id) != (
            truth.observation_id,
            truth.station_id,
        ):
            raise ValueError(
                "exact link %s -> %s crosses observation/station boundaries"
                % (candidate_id, injection_id)
            )
        candidate_to_truth[candidate_id] = injection_id
    linked_truth = set(candidate_to_truth.values())
    return {
        "rows": [
            {"candidate_id": key, "recovery_link_id": value}
            for key, value in sorted(candidate_to_truth.items())
        ],
        "n_unlinked_candidates": len(candidates) - len(candidate_to_truth),
        "n_missed_injections": len(truths) - len(linked_truth),
    }


def main(args: argparse.Namespace) -> None:
    if not args.acknowledge_post_inference:
        raise ValueError("refusing truth linkage without --acknowledge-post-inference")
    prediction_path = Path(args.predictions)
    if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
        raise FileNotFoundError(
            "label-blind predictions must exist before recovery linkage: %s"
            % prediction_path
        )
    candidates = load_candidates(args.candidates)
    truths = load_truth(args.truth)
    policy = load_policy(args.policy, allow_draft=False)
    exact_rows = _load_exact_links(args.exact_links)
    mode = args.mode
    if mode == "auto":
        mode = "exact" if exact_rows else "frozen_hungarian"

    if mode == "exact":
        if not exact_rows:
            raise ValueError("exact mode requested but the exact-link sidecar is empty")
        linked = _validate_exact_links(exact_rows, candidates, truths)
        rows = linked["rows"]
        n_unlinked_candidates = linked["n_unlinked_candidates"]
        n_missed_injections = linked["n_missed_injections"]
        tolerances = None
    else:
        tolerances = policy.get("recovery_link_tolerances", {})
        if set(tolerances) != REQUIRED_TOLERANCES:
            raise ValueError(
                "frozen policy lacks recovery_link_tolerances=%s"
                % sorted(REQUIRED_TOLERANCES)
            )
        matches, false_hits, missed = _hungarian_matches(
            candidates,
            truths,
            {key: float(tolerances[key]) for key in REQUIRED_TOLERANCES},
        )
        rows = [
            {
                "candidate_id": candidate.candidate_id,
                "recovery_link_id": truth.injection_id,
            }
            for candidate, truth in matches
        ]
        n_unlinked_candidates = len(false_hits)
        n_missed_injections = len(missed)

    write_jsonl(args.output, rows)
    audit = {
        "format_version": 1,
        "dataset_role": "locked_synthetic_test_b",
        "linkage_mode": mode,
        "n_candidates": len(candidates),
        "n_injections": len(truths),
        "n_candidate_links": len(rows),
        "n_unique_linked_injections": len({row["recovery_link_id"] for row in rows}),
        "n_unlinked_candidates": n_unlinked_candidates,
        "n_missed_injections": n_missed_injections,
        "recovery_link_tolerances": tolerances,
        "policy": str(Path(args.policy).resolve()),
        "policy_sha256": sha256_file(args.policy),
        "candidates": str(Path(args.candidates).resolve()),
        "candidates_sha256": sha256_file(args.candidates),
        "truth": str(Path(args.truth).resolve()),
        "truth_sha256": sha256_file(args.truth),
        "exact_links": (
            str(Path(args.exact_links).resolve()) if args.exact_links else None
        ),
        "exact_links_sha256": (
            sha256_file(args.exact_links) if args.exact_links else None
        ),
        "predictions_exist_before_linkage": True,
        "predictions": str(prediction_path.resolve()),
        "predictions_sha256": sha256_file(str(prediction_path)),
        "operational_inputs_truth_free": True,
    }
    write_json(str(Path(args.output).with_suffix(".audit.json")), audit)
    print(
        "Wrote %d post-inference recovery links using %s mode to %s"
        % (len(rows), mode, args.output)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Link locked Test-B BLISS hits to truth only after inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--exact-links", default=None)
    parser.add_argument(
        "--mode", choices=("auto", "exact", "frozen_hungarian"), default="auto"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--acknowledge-post-inference", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
