#!/usr/bin/env python3
"""Audit how Naoise's discrete width bank covers Stage-4's 10--100 Hz scope."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from lofts_bliss_schema import (
    atomic_write_text,
    load_observations,
    sha256_file,
    write_json,
)


def main(args: argparse.Namespace) -> None:
    observations = list(
        load_observations(args.observations, require_files=False).values()
    )
    targets = [
        float(value) for value in args.target_widths_hz.split(",") if value.strip()
    ]
    if not targets or any(value <= 0 for value in targets):
        raise ValueError("target widths must be positive")
    station_rows = {}
    proposed = set()
    for observation in observations:
        channel_hz = abs(float(observation.signed_foff_hz))
        bank = list(observation.search_bank_width_channels)
        physical = {width: width * channel_hz for width in bank}
        rows = []
        for target in targets:
            nearest = min(
                bank, key=lambda value: abs(math.log(physical[value] / target))
            )
            ratio = max(physical[nearest] / target, target / physical[nearest])
            rounded = max(1, int(round(target / channel_hz)))
            proposed.add(rounded)
            rows.append(
                {
                    "target_width_hz": target,
                    "nearest_native_template_channels": nearest,
                    "nearest_native_template_hz": physical[nearest],
                    "multiplicative_mismatch": ratio,
                    "target_rounded_template_channels": rounded,
                    "target_rounded_template_hz": rounded * channel_hz,
                    "nearest_is_inside_stage4_nominal_range": (
                        args.stage4_width_min_hz
                        <= physical[nearest]
                        <= args.stage4_width_max_hz
                    ),
                }
            )
        station_rows[observation.station_id] = {
            "channel_bandwidth_hz": channel_hz,
            "native_bank_channels": bank,
            "native_bank_hz": [physical[value] for value in bank],
            "targets": rows,
        }
    payload = {
        "format_version": 1,
        "observations": str(Path(args.observations).resolve()),
        "observations_sha256": sha256_file(args.observations),
        "stage4_nominal_width_range_hz": [
            args.stage4_width_min_hz,
            args.stage4_width_max_hz,
        ],
        "target_widths_hz": targets,
        "stations": station_rows,
        "target_aligned_integer_template_proposal_channels": sorted(proposed),
        "proposal_boundary": (
            "This is a geometry audit, not permission to change the locked first "
            "comparison. A new bank changes trials, false-alarm floor, compute cost, "
            "candidate multiplicity and recovery errors; it requires fresh noise-floor "
            "calibration and blind-injection validation."
        ),
    }
    write_json(args.output_json, payload)
    lines = [
        "# Naoise-bank coverage of the Stage-4 width scope",
        "",
        "| Station | Target (Hz) | Nearest template (channels) | Template (Hz) | Mismatch factor | Native width inside 10–100 Hz? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for station, values in sorted(station_rows.items()):
        for row in values["targets"]:
            lines.append(
                "| %s | %.1f | %d | %.3f | %.3f | %s |"
                % (
                    station,
                    row["target_width_hz"],
                    row["nearest_native_template_channels"],
                    row["nearest_native_template_hz"],
                    row["multiplicative_mismatch"],
                    "yes" if row["nearest_is_inside_stage4_nominal_range"] else "no",
                )
            )
    lines.extend(
        [
            "",
            "Target-aligned rounded channel widths: `%s`."
            % ",".join(str(value) for value in sorted(proposed)),
            "",
            payload["proposal_boundary"],
        ]
    )
    atomic_write_text(args.output_markdown, "\n".join(lines) + "\n")
    print("Wrote bank-coverage audit to %s" % args.output_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit BLISS width-template coverage of Stage-4's target range",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--target-widths-hz", default="10,20,30,50,75,100")
    parser.add_argument("--stage4-width-min-hz", type=float, default=10.0)
    parser.add_argument("--stage4-width-max-hz", type=float, default=100.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
