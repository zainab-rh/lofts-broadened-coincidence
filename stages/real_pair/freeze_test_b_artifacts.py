#!/usr/bin/env python3
"""Create a checksum inventory for a completed, read-only Synthetic Test B."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from lofts_bliss_schema import atomic_write_text, sha256_file, write_json


def _command(arguments, cwd=None):
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main(args: argparse.Namespace) -> None:
    if not args.acknowledge_no_further_tuning:
        raise ValueError(
            "use --acknowledge-no-further-tuning only after the locked evaluation is complete"
        )
    root = Path(args.run_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = Path(args.output).resolve()
    inventory = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == output:
            continue
        inventory.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(str(path)),
            }
        )
    repository = Path(args.repository).resolve() if args.repository else root
    git_commit = _command(["git", "rev-parse", "HEAD"], cwd=str(repository))
    git_status = _command(["git", "status", "--short"], cwd=str(repository))
    pip_freeze = _command([sys.executable, "-m", "pip", "freeze"])
    environment_path = output.with_name(output.stem + ".environment.txt")
    atomic_write_text(str(environment_path), (pip_freeze or "unavailable") + "\n")
    payload = {
        "format_version": 1,
        "test_name": "Synthetic Test B",
        "status": "frozen",
        "run_root": str(root),
        "files": inventory,
        "n_files": len(inventory),
        "python": sys.version,
        "platform": platform.platform(),
        "git_repository": str(repository),
        "git_commit": git_commit,
        "git_status_at_freeze": git_status,
        "environment_freeze": str(environment_path),
        "environment_freeze_sha256": sha256_file(str(environment_path)),
        "acknowledgement": (
            "The checkpoint, policy and Test-B design will not be tuned using these labels."
        ),
    }
    write_json(str(output), payload)
    print("Frozen %d artifacts in %s" % (len(inventory), output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checksum and freeze a completed Synthetic-Test-B artifact tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=None)
    parser.add_argument("--acknowledge-no-further-tuning", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
