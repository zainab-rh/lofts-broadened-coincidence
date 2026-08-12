#!/usr/bin/env python3
"""Freeze the Synthetic-Test-B design before generating test predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from lofts_bliss_schema import sha256_file, stable_id, write_json


def _git_commit(repository: str):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _reject_placeholders(value, location="design"):
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_placeholders(nested, "%s.%s" % (location, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, "%s[%d]" % (location, index))
    elif isinstance(value, str) and value.strip().upper().startswith("EDIT_"):
        raise ValueError("unresolved Test-B design placeholder at %s" % location)


def _validate_design(path: str, expected_seed: int):
    try:
        design = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Test-B design must be auditable JSON") from exc
    if not isinstance(design, dict):
        raise ValueError("Test-B design root must be a JSON object")
    _reject_placeholders(design)
    if design.get("dataset_role") != "locked_synthetic_test_b":
        raise ValueError("design.dataset_role must be locked_synthetic_test_b")
    if design.get("population") not in {"detected_conditioned", "fixed_power"}:
        raise ValueError(
            "design.population must be detected_conditioned or fixed_power"
        )
    if int(design.get("random_seed")) != int(expected_seed):
        raise ValueError("--test-b-seed disagrees with design.random_seed")
    widths = [float(value) for value in design.get("widths_fwhm_hz", [])]
    if widths != [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
        raise ValueError("primary Test-B widths must be exactly 10,20,30,50,75,100 Hz")
    if set(design.get("profile_families", [])) != {"lorentzian", "gaussian", "box"}:
        raise ValueError(
            "primary Test-B profile families must be Lorentzian, Gaussian and box"
        )
    separation = design.get("separation", {})
    if not all(
        separation.get(key) is True
        for key in (
            "calibration_seeds_disjoint",
            "calibration_frequency_blocks_disjoint",
            "test_a_not_reused_for_selection",
        )
    ):
        raise ValueError("all calibration/Test-B separation assertions must be true")
    evaluation = design.get("evaluation", {})
    if evaluation.get("candidate_set") != "station_union_not_intersection":
        raise ValueError("design must register the station union, not intersection")
    if evaluation.get("one_station_detections_retained") is not True:
        raise ValueError("design must retain one-station detections")
    return design


def main(args: argparse.Namespace) -> None:
    if not args.acknowledge_before_test:
        raise ValueError(
            "use --acknowledge-before-test only before Test-B predictions or labels exist"
        )
    if not str(args.reviewer).strip() or not str(args.notes).strip():
        raise ValueError("reviewer and notes must be non-empty")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite frozen preregistration %s" % output
        )
    template = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not str(template.get("status", "")).startswith("template"):
        raise ValueError("input preregistration must still have template status")
    required = (
        args.input,
        args.test_b_design,
        args.association_policy,
        args.stage3_checkpoint,
        args.stage4_checkpoint,
    )
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    design = _validate_design(args.test_b_design, args.test_b_seed)
    policy = json.loads(Path(args.association_policy).read_text(encoding="utf-8"))
    if policy.get("status") != "frozen":
        raise ValueError("association policy must be frozen before preregistration")
    template["status"] = "frozen_before_test_b"
    template["preregistration_id"] = stable_id(
        "test_b_preregistration",
        sha256_file(args.input),
        sha256_file(args.test_b_design),
        sha256_file(args.association_policy),
        sha256_file(args.stage4_checkpoint),
        args.test_b_seed,
    )
    template["frozen_review"] = {
        "reviewer": str(args.reviewer).strip(),
        "notes": str(args.notes).strip(),
        "utc_datetime": dt.datetime.now(dt.timezone.utc).isoformat(),
        "test_b_predictions_or_labels_inspected": False,
    }
    template["locked_inputs"] = {
        "test_b_seed": int(args.test_b_seed),
        "population": design["population"],
        "test_b_design": str(Path(args.test_b_design).resolve()),
        "test_b_design_sha256": sha256_file(args.test_b_design),
        "association_policy": str(Path(args.association_policy).resolve()),
        "association_policy_sha256": sha256_file(args.association_policy),
        "stage3_checkpoint": str(Path(args.stage3_checkpoint).resolve()),
        "stage3_checkpoint_sha256": sha256_file(args.stage3_checkpoint),
        "stage4_checkpoint": str(Path(args.stage4_checkpoint).resolve()),
        "stage4_checkpoint_sha256": sha256_file(args.stage4_checkpoint),
        "repository": str(Path(args.repository).resolve()),
        "git_commit": _git_commit(str(Path(args.repository).resolve())),
    }
    write_json(str(output), template)
    print("Frozen Synthetic-Test-B preregistration at %s" % output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the Synthetic-Test-B design before test generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-b-design", required=True)
    parser.add_argument("--test-b-seed", type=int, required=True)
    parser.add_argument("--association-policy", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--stage4-checkpoint", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--acknowledge-before-test", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
