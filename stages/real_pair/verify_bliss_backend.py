#!/usr/bin/env python3
"""Fail-fast provenance gate for the BLISS backend resolved by the search.

The broadened-signal driver and the compiled ``blissdedrift`` backend are two
separate provenance surfaces.  This verifier records both the Git checkout and
the exact module path resolved by the Python interpreter that will run the
blind search.  It deliberately fails if that path is outside the registered
checkout.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Optional

from lofts_bliss_schema import write_json

PROBE_PREFIX = "LOFTS_BLISS_PROBE_JSON="


def run_command(arguments, cwd: Optional[Path] = None):
    completed = subprocess.run(
        [str(value) for value in arguments],
        cwd=None if cwd is None else str(cwd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed (%s): %s"
            % (" ".join(str(value) for value in arguments), completed.stderr.strip())
        )
    return completed


def command(arguments, cwd: Optional[Path] = None) -> str:
    return run_command(arguments, cwd).stdout.strip()


def parse_probe_stdout(stdout: str, stderr: str = ""):
    """Extract the sentinel-prefixed JSON record from noisy module output.

    Some BLISS builds and Python startup hooks print CUDA or
    optional-dependency diagnostics to stdout.  Treating the complete stdout
    stream as JSON caused the original verifier to fail even though module
    resolution succeeded.  A unique protocol prefix makes the parser
    independent of unrelated output.
    """

    for line in reversed(stdout.splitlines()):
        if not line.startswith(PROBE_PREFIX):
            continue
        payload = line[len(PROBE_PREFIX) :]
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "BLISS probe emitted malformed protocol JSON: %r" % payload
            ) from exc
        if not isinstance(value, dict):
            raise ValueError("BLISS probe protocol payload is not a JSON object")
        return value
    raise ValueError(
        "BLISS probe did not emit its protocol record. stdout=%r stderr=%r"
        % (stdout[-2000:], stderr[-2000:])
    )


def probe_module_resolution(python: Path):
    """Resolve ``blissdedrift`` without executing its package initializer.

    Provenance verification only needs to prove which checkout the selected
    Python interpreter would resolve. Importing the package during this gate
    unnecessarily initialises optional devices and produces diagnostics. The
    actual search imports it later on the explicitly selected CPU/CUDA backend.
    """

    probe_code = (
        "import importlib.util,json,sys; "
        "spec=importlib.util.find_spec('blissdedrift'); "
        "locations=[] if spec is None or spec.submodule_search_locations is None "
        "else list(spec.submodule_search_locations); "
        "version=None; "
        'exec("try:\\n import importlib.metadata as m\\n '
        "version=m.version('blissdedrift')\\nexcept Exception:\\n pass\"); "
        "payload={'module_file': None if spec is None else spec.origin, "
        "'module_search_locations': locations, 'module_version': version, "
        "'python_version': sys.version, 'python_executable': sys.executable}; "
        "print(%r + json.dumps(payload, sort_keys=True))" % PROBE_PREFIX
    )
    completed = run_command([python, "-c", probe_code])
    imported = parse_probe_stdout(completed.stdout, completed.stderr)
    protocol_lines = {
        line for line in completed.stdout.splitlines() if line.startswith(PROBE_PREFIX)
    }
    stdout_noise = [
        line for line in completed.stdout.splitlines() if line not in protocol_lines
    ]
    return imported, stdout_noise, completed.stderr.splitlines()


def source_literal_version(module_file: Path):
    """Read a literal ``__version__`` assignment without importing the module."""

    if module_file.suffix != ".py":
        return None
    try:
        tree = ast.parse(module_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, TypeError):
                    return None
                return str(literal) if isinstance(literal, str) else None
    return None


def main(args: argparse.Namespace) -> None:
    repository = Path(args.repository).expanduser().resolve()
    python = Path(args.python).expanduser().resolve()
    if not repository.is_dir():
        raise NotADirectoryError(repository)
    if not python.is_file():
        raise FileNotFoundError(python)

    commit = command(["git", "rev-parse", "HEAD"], repository)
    if commit != args.expected_commit:
        raise ValueError(
            "BLISS checkout is at %s; the registered pilot requires %s"
            % (commit, args.expected_commit)
        )
    status = command(["git", "status", "--short"], repository)
    if status and not args.allow_dirty:
        raise ValueError(
            "BLISS checkout is dirty. Commit/stash changes or use --allow-dirty "
            "only for a separately labelled engineering run:\n%s" % status
        )

    imported, probe_stdout_noise, probe_stderr = probe_module_resolution(python)
    module_file_text = imported.get("module_file")
    if not module_file_text:
        raise ValueError(
            "the selected BLISS Python cannot resolve the blissdedrift package"
        )
    module_file = Path(module_file_text).expanduser().resolve()
    if not module_file.is_file():
        raise FileNotFoundError(
            "resolved blissdedrift module file does not exist: %s" % module_file
        )
    try:
        module_file.relative_to(repository)
    except ValueError as exc:
        raise ValueError(
            "blissdedrift imported from %s, outside registered repository %s"
            % (module_file, repository)
        ) from exc

    module_search_locations = []
    for value in imported.get("module_search_locations", []):
        location = Path(value).expanduser().resolve()
        try:
            location.relative_to(repository)
        except ValueError as exc:
            raise ValueError(
                "blissdedrift search location %s is outside registered repository %s"
                % (location, repository)
            ) from exc
        module_search_locations.append(str(location))

    module_version = imported.get("module_version")
    if module_version in (None, ""):
        module_version = source_literal_version(module_file)

    payload = {
        "format_version": 1,
        "repository": str(repository),
        "git_commit": commit,
        "expected_git_commit": args.expected_commit,
        "git_status_short": status,
        "dirty_checkout_allowed": bool(args.allow_dirty),
        "python_executable": imported["python_executable"],
        "python_version": imported["python_version"],
        "blissdedrift_module_file": str(module_file),
        "blissdedrift_module_search_locations": module_search_locations,
        "blissdedrift_module_version": module_version,
        "probe_mode": "importlib_find_spec_without_package_import",
        "runtime_import_deferred_to_search": True,
        "registered_search_backend_default": "cpu",
        "probe_stdout_nonprotocol_lines": probe_stdout_noise,
        "probe_stderr_lines": probe_stderr,
    }
    write_json(args.output, payload)
    print(
        "Verified BLISS backend %s; blissdedrift resolves from %s"
        % (commit, module_file)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the exact BLISS backend checkout and resolved module path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument(
        "--expected-commit",
        default="2b98afe960f13ee7e467aca499576d87ee7502f5",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
