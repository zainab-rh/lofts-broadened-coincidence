#!/usr/bin/env python3
"""Checksum and freeze a completed unlabeled LOFTS0050 real-pair pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from lofts_bliss_schema import (
    atomic_write_text,
    read_json_records,
    sha256_file,
    write_json,
)

TRANSIENT_DIRECTORY_NAMES = {"__pycache__", ".matplotlib", ".pytest_cache"}
FORBIDDEN_OPERATIONAL_KEYS = {
    "event_id",
    "injection_id",
    "label",
    "pair_label",
    "recovery_link_id",
    "truth",
}


def command(arguments, cwd=None):
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def forbidden_keys(value: Any) -> List[str]:
    """Return truth-adjacent key paths from a nested operational record."""

    found: List[str] = []

    def visit(item: Any, prefix: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                path = "%s.%s" % (prefix, key) if prefix else str(key)
                if str(key) in FORBIDDEN_OPERATIONAL_KEYS:
                    found.append(path)
                visit(child, path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, "%s[%d]" % (prefix, index))

    visit(value, "")
    return found


def inventory_files(root: Path, excluded_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    excluded = {path.resolve() for path in excluded_paths}
    inventory: List[Dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        resolved = path.resolve()
        if resolved in excluded:
            continue
        relative = resolved.relative_to(root)
        if any(part in TRANSIENT_DIRECTORY_NAMES for part in relative.parts):
            continue
        inventory.append(
            {
                "relative_path": str(relative),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(str(resolved)),
            }
        )
    return inventory


def main(args: argparse.Namespace) -> None:
    if not args.acknowledge_unlabeled_external_pilot:
        raise ValueError("freeze requires --acknowledge-unlabeled-external-pilot")
    root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            "refusing to overwrite an existing frozen result: %s" % output
        )
    analysis_path = root / "analysis" / "real_pair_analysis.json"
    required_paths = (
        root / "observations.jsonl",
        root / "policy" / "real_pair_policy.frozen.json",
        root / "union" / "real_candidate_union.jsonl",
        root / "extracted" / "primary" / "pair_manifest.jsonl",
        root / "extracted" / "primary" / "extraction_summary.json",
        root / "extracted" / "controls" / "control_pair_manifest.jsonl",
        root / "extracted" / "controls" / "control_extraction_summary.json",
        root / "inference" / "primary" / "stage4_bliss_predictions.jsonl",
        root / "inference" / "primary" / "stage4_bliss_inference_summary.json",
        root / "inference" / "controls" / "stage4_bliss_predictions.jsonl",
        root / "inference" / "controls" / "stage4_bliss_inference_summary.json",
        analysis_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "real-pair freeze is missing required pipeline artifacts: %s" % missing
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("dataset_role") != "unlabeled_real_barycentric_pair":
        raise ValueError("analysis has the wrong dataset role")
    if analysis.get("labels_used") is not False:
        raise ValueError("real-pair freeze requires a label-free analysis")
    prediction_paths = (
        root / "inference" / "primary" / "stage4_bliss_predictions.jsonl",
        root / "inference" / "controls" / "stage4_bliss_predictions.jsonl",
    )
    for prediction_path in prediction_paths:
        for record in read_json_records(str(prediction_path)):
            forbidden = forbidden_keys(record)
            if forbidden:
                raise ValueError(
                    "real operational predictions contain truth-adjacent keys %s in %s"
                    % (forbidden[:5], prediction_path)
                )
            if record.get("stage4_predicted_match") is not None:
                raise ValueError(
                    "real predictions must not contain a binary deployment decision: %s"
                    % prediction_path
                )
    inference_summaries = []
    for directory in ("primary", "controls"):
        summary_path = (
            root / "inference" / directory / "stage4_bliss_inference_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset_role") != "unlabeled_real_pair":
            raise ValueError("%s has the wrong dataset role" % summary_path)
        if summary.get("labels_used") is not False:
            raise ValueError("%s does not assert labels_used=false" % summary_path)
        if int(summary.get("n_scored_union_pairs", 0)) <= 0:
            raise ValueError("%s contains no scored pairs" % summary_path)
        if float(
            summary.get("station_order_symmetry_max_abs_score_difference", 1.0)
        ) > float(args.symmetry_tolerance):
            raise ValueError("%s failed the station-order symmetry gate" % summary_path)
        inference_summaries.append(summary)
    for key in ("stage3_checkpoint_sha256", "stage4_checkpoint_sha256"):
        values = {str(summary.get(key, "")) for summary in inference_summaries}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(
                "primary/control inference summaries disagree on %s: %s"
                % (key, sorted(values))
            )

    environment = command([sys.executable, "-m", "pip", "freeze"])
    if environment is None:
        raise RuntimeError("python -m pip freeze failed; refusing an incomplete freeze")
    environment_path = output.with_suffix(".environment.txt")
    atomic_write_text(str(environment_path), environment + "\n")
    inventory = inventory_files(root, excluded_paths=(output,))
    pipeline_root = Path(__file__).resolve().parent
    pipeline_source_inventory = inventory_files(
        pipeline_root,
        excluded_paths=(),
    )
    repository = str(Path(args.repository).resolve())
    payload = {
        "format_version": 1,
        "experiment": "LOFTS0050 real Ireland-Sweden BLISS-to-Stage4 pilot",
        "status": "frozen_unlabeled_external_pipeline_pilot",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_root": str(root),
        "n_files": len(inventory),
        "files": inventory,
        "analysis_sha256": sha256_file(str(analysis_path)),
        "python": sys.version,
        "platform": platform.platform(),
        "environment_freeze": str(environment_path),
        "environment_freeze_sha256": sha256_file(str(environment_path)),
        "transient_directories_excluded": sorted(TRANSIENT_DIRECTORY_NAMES),
        "pipeline_source_root": str(pipeline_root),
        "pipeline_source_files": pipeline_source_inventory,
        "primary_control_checkpoint_identity_verified": True,
        "station_order_symmetry_tolerance": float(args.symmetry_tolerance),
        "stage4_repository": repository,
        "stage4_git_commit": command(["git", "rev-parse", "HEAD"], cwd=repository),
        "stage4_git_status_short": command(
            ["git", "status", "--short"], cwd=repository
        ),
        "scientific_boundary": (
            "This freezes an unlabeled real simultaneous barycentric processing "
            "pilot. It does not freeze a real-data accuracy, sensitivity, or "
            "technosignature-validation result."
        ),
    }
    write_json(str(output), payload)
    print("Frozen %d real-pair artifacts in %s" % (len(inventory), output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the unlabeled real-pair pilot artifact tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-6)
    parser.add_argument("--acknowledge-unlabeled-external-pilot", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
