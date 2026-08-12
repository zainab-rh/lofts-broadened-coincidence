#!/usr/bin/env python3
"""Freeze the unlabeled real-pair pilot policy before union/scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from lofts_bliss_schema import sha256_file, stable_id, write_json


def main(args: argparse.Namespace) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "resolution_derived_draft":
        raise ValueError("input policy must have status=resolution_derived_draft")
    if not args.acknowledge_before_union_and_scores:
        raise ValueError(
            "refusing to freeze without --acknowledge-before-union-and-scores"
        )
    if not args.reviewer.strip() or not args.notes.strip():
        raise ValueError("reviewer and notes must be non-empty")
    payload["status"] = "frozen_real_pair_pilot"
    payload["frozen_review"] = {
        "reviewer": args.reviewer.strip(),
        "notes": args.notes.strip(),
        "utc_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_draft": str(source.resolve()),
        "source_draft_sha256": sha256_file(str(source)),
        "labels_used": False,
        "stage4_scores_used": False,
        "association_results_used": False,
    }
    payload["policy_id"] = stable_id(
        "real_policy_frozen", payload.get("policy_id"), payload["frozen_review"]
    )
    write_json(args.output, payload)
    print("Frozen real-pair pilot policy written to %s" % args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a resolution-derived real-pair policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--acknowledge-before-union-and-scores", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
