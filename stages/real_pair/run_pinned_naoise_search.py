#!/usr/bin/env python3
"""Run the pinned Naoise search with an explicit, auditable array backend.

The upstream ``blind_hit_finder.py`` chooses CuPy whenever it can be imported.
On some login/compute nodes CuPy imports successfully even though CUDA device
initialisation fails.  That makes an import test an unsafe device-selection
test.  This launcher keeps the pinned upstream script byte-for-byte unchanged
while making backend selection explicit:

* ``cpu`` blocks imports of CuPy inside the pinned script, selecting its
  existing NumPy/BLISS CPU path;
* ``cuda`` requires a successful CuPy device probe before execution; and
* ``auto`` uses CUDA only when that probe succeeds, otherwise it uses CPU.

The selected backend, exact script hash, Python interpreter, command arguments,
thread settings, and exit status are written to a JSON provenance sidecar.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import json
import os
import platform
import runpy
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def probe_cuda() -> Tuple[bool, Optional[Dict[str, object]], Optional[str]]:
    """Return whether CuPy can initialise at least one usable CUDA device."""

    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        if count < 1:
            return False, None, "CuPy reported zero CUDA devices"
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        # Force a minimal allocation and synchronisation.  Device enumeration
        # alone can succeed on a node whose CUDA context is unusable.
        probe = cp.asarray([1.0], dtype=cp.float32)
        value = float(cp.asnumpy(probe)[0])
        cp.cuda.Stream.null.synchronize()
        if value != 1.0:
            return False, None, "CUDA round-trip probe returned an invalid value"
        return (
            True,
            {
                "device_count": count,
                "device_index": 0,
                "device_name": str(name),
                "cupy_version": getattr(cp, "__version__", None),
            },
            None,
        )
    except BaseException as exc:
        return False, None, "%s: %s" % (type(exc).__name__, exc)


def select_backend(
    requested: str,
) -> Tuple[str, Optional[Dict[str, object]], Optional[str]]:
    if requested == "cpu":
        return "cpu", None, None
    available, cuda, error = probe_cuda()
    if requested == "cuda":
        if not available:
            raise RuntimeError(
                "CUDA was requested but its runtime probe failed: %s" % error
            )
        return "cuda", cuda, None
    if available:
        return "cuda", cuda, None
    return "cpu", None, error


@contextlib.contextmanager
def block_cupy_imports() -> Iterator[None]:
    """Make the pinned script's optional CuPy import follow its CPU fallback."""

    original_import = builtins.__import__
    removed = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "cupy"
        or name.startswith("cupy.")
        or name == "cupyx"
        or name.startswith("cupyx.")
    }
    for name in removed:
        sys.modules.pop(name, None)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "cupy"
            or name.startswith("cupy.")
            or name == "cupyx"
            or name.startswith("cupyx.")
        ):
            raise ImportError("CuPy intentionally disabled by LOFTS CPU launcher")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = original_import
        for name in list(sys.modules):
            if (
                name == "cupy"
                or name.startswith("cupy.")
                or name == "cupyx"
                or name.startswith("cupyx.")
            ):
                sys.modules.pop(name, None)
        sys.modules.update(removed)


def wrapper_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def runtime_payload(
    args: argparse.Namespace,
    script: Path,
    script_sha256: str,
    effective_backend: str,
    cuda: Optional[Dict[str, object]],
    fallback_reason: Optional[str],
    script_arguments: List[str],
) -> Dict[str, object]:
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "format_version": 1,
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": wrapper_sha256(),
        "requested_backend": args.backend,
        "effective_backend": effective_backend,
        "cuda_probe": cuda,
        "auto_cpu_fallback_reason": fallback_reason,
        "pinned_script": str(script),
        "pinned_script_sha256": script_sha256,
        "expected_script_sha256": args.expected_script_sha256,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "cpu_count": os.cpu_count(),
        "thread_environment": thread_variables,
        "script_arguments": script_arguments,
        "started_unix_seconds": time.time(),
        "completed": False,
        "exit_status": None,
    }


def execute_script(script: Path, arguments: List[str], backend: str) -> None:
    old_argv = sys.argv[:]
    old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    sys.argv = [str(script)] + arguments
    try:
        if backend == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            with block_cupy_imports():
                runpy.run_path(str(script), run_name="__main__")
        else:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_cuda_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible


def main(args: argparse.Namespace, script_arguments: List[str]) -> int:
    script = Path(args.script).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    actual_sha256 = sha256_file(script)
    if actual_sha256 != args.expected_script_sha256:
        raise ValueError(
            "pinned Naoise script SHA-256 is %s; expected %s"
            % (actual_sha256, args.expected_script_sha256)
        )

    effective, cuda, fallback_reason = select_backend(args.backend)
    payload = runtime_payload(
        args,
        script,
        actual_sha256,
        effective,
        cuda,
        fallback_reason,
        script_arguments,
    )
    print(
        "# LOFTS pinned search launcher  requested=%s  effective=%s"
        % (args.backend, effective),
        flush=True,
    )
    if fallback_reason:
        print(
            "# CUDA probe unavailable; CPU fallback reason: %s" % fallback_reason,
            flush=True,
        )

    if args.probe_only:
        payload["completed"] = True
        payload["exit_status"] = 0
        payload["completed_unix_seconds"] = time.time()
        if args.runtime_provenance:
            atomic_write_json(Path(args.runtime_provenance), payload)
        print(json.dumps(payload, sort_keys=True))
        return 0

    if not script_arguments:
        raise ValueError("the pinned script arguments are missing; place them after --")

    status = 0
    try:
        execute_script(script, script_arguments, effective)
    except SystemExit as exc:
        if exc.code is None:
            status = 0
        elif isinstance(exc.code, int):
            status = exc.code
        else:
            status = 1
            print(str(exc.code), file=sys.stderr)
    except BaseException as exc:
        status = 130 if isinstance(exc, KeyboardInterrupt) else 1
        payload["exception_type"] = type(exc).__name__
        payload["exception_message"] = str(exc)
        raise
    finally:
        payload["completed"] = status == 0
        payload["exit_status"] = status
        payload["completed_unix_seconds"] = time.time()
        payload["elapsed_seconds"] = (
            payload["completed_unix_seconds"] - payload["started_unix_seconds"]
        )
        if args.runtime_provenance:
            atomic_write_json(Path(args.runtime_provenance), payload)
    return status


def parse_arguments(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run a pinned Naoise blind search on an explicit CPU/CUDA backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--script", required=True)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--runtime-provenance")
    parser.add_argument("--probe-only", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    return args, remaining


if __name__ == "__main__":
    parsed, downstream = parse_arguments()
    raise SystemExit(main(parsed, downstream))
