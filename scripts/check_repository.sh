#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$REPO_ROOT/stages/stage3:$REPO_ROOT/stages/stage4:$REPO_ROOT/stages/real_pair${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/5] Compiling Python sources"
"$PYTHON_BIN" -m compileall -q "$REPO_ROOT/stages" "$REPO_ROOT/tests"

echo "[2/5] Running Stage-3 core tests"
"$PYTHON_BIN" "$REPO_ROOT/tests/test_stage3_core.py"

echo "[3/5] Running Stage-4 core tests"
(cd "$REPO_ROOT/stages/stage4" && "$PYTHON_BIN" -m unittest -v test_stage4_core.py)

echo "[4/5] Running real-pair regression tests"
(cd "$REPO_ROOT/stages/real_pair" && "$PYTHON_BIN" -m unittest -v \
  test_bliss_stage4_integration.py test_real_pair_v3.py test_cpu_and_plots_v3_2.py)

echo "[5/5] Checking shell syntax"
bash -n "$REPO_ROOT/stages/stage4/run_stage4_experiment.sh"
bash -n "$REPO_ROOT/stages/real_pair/run_bliss_stage4_pipeline.sh"
bash -n "$REPO_ROOT/stages/real_pair/run_lofts0050_real_pair.sh"

echo "Repository checks passed."
