#!/usr/bin/env python3
"""Combine already-adapted station JSONL files with global-ID validation."""

from __future__ import annotations

import argparse

from lofts_bliss_schema import (
    CandidateRecord,
    InjectionTruthRecord,
    read_json_records,
    write_jsonl,
)


def main(args: argparse.Namespace) -> None:
    rows = []
    for path in args.inputs:
        rows.extend(read_json_records(path))
    if args.kind == "candidate":
        parsed = [CandidateRecord.from_dict(row) for row in rows]
        identifiers = [item.candidate_id for item in parsed]
        output = [item.to_dict(include_truth=False) for item in parsed]
    elif args.kind == "truth":
        parsed = [InjectionTruthRecord.from_dict(row) for row in rows]
        identifiers = [item.injection_id for item in parsed]
        output = [item.to_dict() for item in parsed]
    else:
        output = rows
        identifiers = []
        for row in rows:
            candidate_id = str(row.get("candidate_id", "")).strip()
            recovery_link_id = str(row.get("recovery_link_id", "")).strip()
            if not candidate_id or not recovery_link_id:
                raise ValueError(
                    "recovery-link rows require candidate_id and recovery_link_id"
                )
            identifiers.append(candidate_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("record identifiers must be globally unique across inputs")
    write_jsonl(args.output, output)
    print("Combined %d %s records into %s" % (len(output), args.kind, args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine canonical per-station JSONL files safely",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kind", choices=("candidate", "truth", "recovery_links"), required=True
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
