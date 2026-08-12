#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

OUTPUT_DIR=$1
PYTHON_BIN=${PYTHON_BIN:-python}
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" --version > "$OUTPUT_DIR/python_version.txt" 2>&1
"$PYTHON_BIN" -m pip freeze --all > "$OUTPUT_DIR/pip_freeze.txt"
uname -a > "$OUTPUT_DIR/system.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$OUTPUT_DIR/nvidia_smi.txt"
fi

if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
  {
    git rev-parse HEAD
    git status --short
  } > "$OUTPUT_DIR/git_state.txt"
fi

"$PYTHON_BIN" - <<'PY' > "$OUTPUT_DIR/runtime.txt"
import json
import platform

payload = {
    "platform": platform.platform(),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
}

try:
    import torch
    payload.update({
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    })
except Exception as exc:
    payload["torch_error"] = repr(exc)

print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "Environment captured in $OUTPUT_DIR"
