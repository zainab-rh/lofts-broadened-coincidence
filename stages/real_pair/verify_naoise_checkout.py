#!/usr/bin/env python3
"""Fail-fast provenance gate for Naoise's pinned blind-hit-finder checkout."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from lofts_bliss_schema import sha256_file, write_json


def command(arguments, cwd: Path) -> str:
    completed = subprocess.run(
        arguments,
        cwd=str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed (%s): %s" % (" ".join(arguments), completed.stderr.strip())
        )
    return completed.stdout.strip()


def main(args: argparse.Namespace) -> None:
    repository = Path(args.repository).expanduser().resolve()
    script = Path(args.script).expanduser().resolve()
    if not repository.is_dir():
        raise NotADirectoryError(repository)
    if not script.is_file():
        raise FileNotFoundError(script)
    commit = command(["git", "rev-parse", "HEAD"], repository)
    if commit != args.expected_commit:
        raise ValueError(
            "Naoise checkout is at %s; the preregistered first comparison requires %s"
            % (commit, args.expected_commit)
        )
    status = command(["git", "status", "--short"], repository)
    if status and not args.allow_dirty:
        raise ValueError(
            "Naoise checkout is dirty. Commit/stash changes or pass --allow-dirty "
            "only for a clearly labelled engineering run:\n%s" % status
        )
    source = script.read_text(encoding="utf-8")
    match = re.search(r"^#\s*VERSION_TAG:\s*(\S.*?)\s*$", source, re.MULTILINE)
    payload = {
        "format_version": 1,
        "repository": str(repository),
        "script": str(script),
        "git_commit": commit,
        "expected_git_commit": args.expected_commit,
        "git_status_short": status,
        "dirty_checkout_allowed": bool(args.allow_dirty),
        "script_sha256": sha256_file(str(script)),
        "version_tag": None if match is None else match.group(1),
        "registered_search": {
            "drift_min_hz_s": -0.2,
            "drift_max_hz_s": 0.2,
            "bank_floor": 20.0,
            "bank_width_channels": [1, 3, 8, 20, 50, 120],
            "fine_channels_per_coarse": 262144,
            "drift_resolution_factor": 1,
            "rolloff_fraction": 0.2,
            "width_tolerance": 0.10,
            "raw_uncollapsed_catalog_required": True,
            "per_template_export_required": True,
        },
    }
    write_json(args.output, payload)
    print(
        "Verified Naoise checkout %s; script SHA-256 %s"
        % (commit, payload["script_sha256"])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the exact Naoise BLISS search checkout and script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument(
        "--expected-commit",
        default="dee329949384f0a0ddb6306d8bbbc2b0db74011a",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
