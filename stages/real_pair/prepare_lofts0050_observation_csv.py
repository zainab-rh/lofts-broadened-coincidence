#!/usr/bin/env python3
"""Create the exact two-row LOFTS0050 observation input from frozen provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IRL_SHA256 = "ce55b78e99d778576359cef22507ba4bbf9584c263dce61e92daa942fbfd8207"
SWE_SHA256 = "778414d498396ead18a641ca55821751ce674403f56381b0f5c7b59e7cdbb8ad"


def is_placeholder(value: str) -> bool:
    text = str(value).strip().lower()
    return not text or any(
        text.startswith(prefix)
        for prefix in ("edit", "todo", "tbd", "unknown", "pending", "unverified")
    )


def main(args: argparse.Namespace) -> None:
    provenance_path = Path(args.naoise_provenance).resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("git_commit") != args.expected_commit:
        raise ValueError("Naoise provenance does not contain the registered commit")
    if not provenance.get("script_sha256"):
        raise ValueError("Naoise provenance lacks the search-script digest")
    if (
        is_placeholder(args.barycentric_tool)
        or is_placeholder(args.barycentric_version)
    ) and not args.allow_unverified_barycentric_provenance:
        raise ValueError(
            "barycentric tool/version are still unverified; ask Naoise, or use "
            "--allow-unverified-barycentric-provenance only for an engineering pilot"
        )
    rows = []
    for station, observation_id, path, digest in (
        ("IRL", "LOFTS0050_IRL_part000", args.irl_h5, args.irl_sha256),
        ("SWE", "LOFTS0050_SWE_part000", args.swe_h5, args.swe_sha256),
    ):
        rows.append(
            {
                "observation_id": observation_id,
                "simultaneous_group_id": args.group_id,
                "station_id": station,
                "filterbank_path": str(Path(path).expanduser().resolve()),
                "barycentric_status": "barycentric",
                "time_alignment": "absolute_mjd",
                "barycentric_tool": args.barycentric_tool,
                "barycentric_version": args.barycentric_version,
                "expected_file_sha256": digest,
                "expected_signed_foff_hz": "",
                "expected_tsamp_s": "",
                "search_fine_channels_per_coarse": 262144,
                "search_rolloff_fraction": 0.2,
                "search_drift_min_hz_s": -0.2,
                "search_drift_max_hz_s": 0.2,
                "search_floor": 20.0,
                "search_bank_width_channels": "1,3,8,20,50,120",
                "search_git_commit": provenance["git_commit"],
                "search_code_sha256": provenance["script_sha256"],
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote LOFTS0050 two-station observation input to %s" % output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the LOFTS0050 real-pair observation CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--irl-h5", required=True)
    parser.add_argument("--swe-h5", required=True)
    parser.add_argument("--naoise-provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-id", default="LOFTS0050_part000")
    parser.add_argument("--irl-sha256", default=IRL_SHA256)
    parser.add_argument("--swe-sha256", default=SWE_SHA256)
    parser.add_argument("--barycentric-tool", required=True)
    parser.add_argument("--barycentric-version", required=True)
    parser.add_argument(
        "--expected-commit",
        default="dee329949384f0a0ddb6306d8bbbc2b0db74011a",
    )
    parser.add_argument(
        "--allow-unverified-barycentric-provenance", action="store_true"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
