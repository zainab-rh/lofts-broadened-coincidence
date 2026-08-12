#!/usr/bin/env python3
"""Review-gate an empirical BLISS association policy for a locked run.

The command does not change any tolerance.  It records the reviewer, date,
notes and source checksum, then changes the status from ``empirical_draft`` to
``frozen``.  If tolerances need changing, edit/regenerate the draft first and
document the reason before using this gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from lofts_bliss_schema import sha256_file, stable_id, write_json


def main(args: argparse.Namespace) -> None:
    if not str(args.reviewer).strip() or not str(args.notes).strip():
        raise ValueError("reviewer and review notes must be non-empty")
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "empirical_draft":
        raise ValueError("input policy must have status=empirical_draft")
    required = {"frequency_hz", "drift_hz_s", "log_width"}
    for section in (
        "association_tolerances",
        "deduplication_tolerances",
        "recovery_link_tolerances",
    ):
        tolerances = payload.get(section, {})
        if set(tolerances) != required:
            raise ValueError("%s must contain exactly %s" % (section, sorted(required)))
        if any(float(tolerances[key]) <= 0 for key in required):
            raise ValueError("%s tolerances must be positive" % section)
    if not args.acknowledge_locked_test:
        raise ValueError(
            "refusing to freeze without --acknowledge-locked-test; this confirms "
            "that Test-B labels have not been used to choose the tolerances"
        )
    payload["status"] = "frozen"
    payload["review_required"] = False
    payload["frozen_review"] = {
        "reviewer": args.reviewer,
        "utc_date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": args.notes,
        "source_draft": str(source.resolve()),
        "source_draft_sha256": sha256_file(str(source)),
        "test_b_labels_used_for_selection": False,
    }
    payload["policy_id"] = stable_id(
        "policy_frozen", payload.get("policy_id"), payload["frozen_review"]
    )
    write_json(args.output, payload)
    print("Frozen association policy written to %s" % args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a reviewed empirical association policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--acknowledge-locked-test", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
