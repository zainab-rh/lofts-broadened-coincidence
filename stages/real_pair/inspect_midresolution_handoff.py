#!/usr/bin/env python3
"""Verify the real high-/mid-resolution headers and document width routing.

This is a hand-off audit, not a sensitivity experiment.  It deliberately uses
the actual SIGPROC headers rather than the approximate resolutions quoted in a
meeting.  It does not resample data and does not claim that one channel equals
one statistically complete detection trial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lofts_bliss_schema import atomic_write_text, write_json
from make_observation_manifest import inspect_filterbank


def _summary(path: str, widths):
    header = inspect_filterbank(path)
    resolution = abs(float(header["foff_mhz"]) * 1e6)
    return {
        "path": str(Path(path).resolve()),
        "signed_foff_hz": float(header["foff_mhz"]) * 1e6,
        "absolute_channel_width_hz": resolution,
        "tsamp_s": float(header["tsamp_s"]),
        "n_time": int(header["n_time"]),
        "n_channels": int(header["n_channels"]),
        "profile_widths_in_native_channels": {
            "%.12g" % width: float(width) / resolution for width in widths
        },
        "header_fingerprint": header["header_fingerprint"],
    }


def main(args: argparse.Namespace) -> None:
    widths = [float(item) for item in args.widths.split(",") if item.strip()]
    if not widths or any(item <= 0 for item in widths):
        raise ValueError("--widths must contain positive comma-separated values")
    if not (0 < args.high_resolution_limit_hz):
        raise ValueError("high-resolution limit must be positive")
    high = _summary(args.high_resolution_file, widths)
    mid = _summary(args.mid_resolution_file, widths)
    routes = [
        {
            "width_hz": width,
            "route": (
                "high_resolution_stage4"
                if width <= args.high_resolution_limit_hz
                else "mid_resolution_search_handoff"
            ),
            "high_resolution_native_channels": width
            / high["absolute_channel_width_hz"],
            "mid_resolution_native_channels": width / mid["absolute_channel_width_hz"],
        }
        for width in widths
    ]
    result = {
        "format_version": 1,
        "high_resolution_product": high,
        "mid_resolution_product": mid,
        "configured_high_resolution_limit_hz": float(args.high_resolution_limit_hz),
        "routing_demonstration": routes,
        "interpretation_boundary": (
            "Channel occupancy documents representation and routing only. Detection "
            "completeness must be measured with the established search on each product."
        ),
    }
    write_json(args.output_json, result)
    lines = [
        "LOFTS resolution hand-off audit",
        "===============================",
        "High-resolution signed foff: %.12g Hz/channel" % high["signed_foff_hz"],
        "Mid-resolution signed foff: %.12g Hz/channel" % mid["signed_foff_hz"],
        "Configured Stage-4 upper width: %.12g Hz" % args.high_resolution_limit_hz,
        "",
    ]
    for row in routes:
        lines.append(
            "%g Hz -> %s (high %.3f channels; mid %.3f channels)"
            % (
                row["width_hz"],
                row["route"],
                row["high_resolution_native_channels"],
                row["mid_resolution_native_channels"],
            )
        )
    lines.extend(("", result["interpretation_boundary"], ""))
    atomic_write_text(str(Path(args.output_json).with_suffix(".txt")), "\n".join(lines))
    print("Wrote verified resolution hand-off audit to %s" % args.output_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect actual high-/mid-resolution headers and document routing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--high-resolution-file", required=True)
    parser.add_argument("--mid-resolution-file", required=True)
    parser.add_argument("--high-resolution-limit-hz", type=float, default=100.0)
    parser.add_argument("--widths", default="10,30,50,75,100,300,1000,3000")
    parser.add_argument("--output-json", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
