#!/usr/bin/env bash
set -euo pipefail

# Repository-aware Stage-4 runner. Override data and checkpoint paths with
# the same-named environment variables before launching a compute job.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
STAGE3_CODE_DIR=${STAGE3_CODE_DIR:-$REPO_ROOT/stages/stage3}
export PYTHONPATH="$SCRIPT_DIR:$STAGE3_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

ACTION="help"
if [[ $# -ge 1 ]]; then
  ACTION="$1"
fi

PYTHON_BIN="$(printenv PYTHON_BIN 2>/dev/null || true)"
STATION_A="$(printenv STATION_A 2>/dev/null || true)"
STATION_B="$(printenv STATION_B 2>/dev/null || true)"
STAGE3_CHECKPOINT="$(printenv STAGE3_CHECKPOINT 2>/dev/null || true)"
STAGE4_OUT="$(printenv STAGE4_OUT 2>/dev/null || true)"
EVAL_OUT="$(printenv EVAL_OUT 2>/dev/null || true)"
SAFE_STATION_A_PATH="$(printenv SAFE_STATION_A_PATH 2>/dev/null || true)"
SAFE_STATION_B_PATH="$(printenv SAFE_STATION_B_PATH 2>/dev/null || true)"

[[ -n "$PYTHON_BIN" ]] || PYTHON_BIN="python"
[[ -n "$STATION_A" ]] || STATION_A="$SAFE_STATION_A_PATH"
[[ -n "$STATION_B" ]] || STATION_B="$SAFE_STATION_B_PATH"
[[ -n "$STATION_A" ]] || STATION_A="/datax2/projects/LOFTS/2025-05-14/LOFTS0192/LOFTS0192.rawspec.0000.fil"
[[ -n "$STATION_B" ]] || STATION_B="/datax2/projects/LOFTS/2025-05-21/LOFTS0199/LOFTS0199.rawspec.0000.fil"
[[ -n "$STAGE3_CHECKPOINT" ]] || STAGE3_CHECKPOINT="$REPO_ROOT/checkpoints/stage3/model_high_freq_broadened.pth"
[[ -n "$STAGE4_OUT" ]] || STAGE4_OUT="$REPO_ROOT/results/test_a/training"
[[ -n "$EVAL_OUT" ]] || EVAL_OUT="$REPO_ROOT/results/test_a/evaluation"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 2
  fi
}

preflight() {
  require_file "$STAGE3_CODE_DIR/train.py"
  require_file "$SCRIPT_DIR/candidate_preprocessing.py"
  require_file "$SCRIPT_DIR/channelized_injection.py"
  require_file "$SCRIPT_DIR/stage4_data.py"
  require_file "$SCRIPT_DIR/stage4_model.py"
  require_file "$SCRIPT_DIR/train_stage4.py"
  require_file "$SCRIPT_DIR/evaluate_stage4.py"
  require_file "$STATION_A"
  require_file "$STATION_B"
  require_file "$STAGE3_CHECKPOINT"
  "$PYTHON_BIN" -m py_compile \
    "$SCRIPT_DIR/candidate_preprocessing.py" "$SCRIPT_DIR/channelized_injection.py" \
    "$SCRIPT_DIR/stage4_data.py" "$SCRIPT_DIR/stage4_model.py" \
    "$SCRIPT_DIR/train_stage4.py" "$SCRIPT_DIR/evaluate_stage4.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/candidate_preprocessing.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/channelized_injection.py"
  (cd "$SCRIPT_DIR" && "$PYTHON_BIN" -m unittest -v test_stage4_core.py)
}

AMP_FLAG=""
if "$PYTHON_BIN" -c \
  'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  >/dev/null 2>&1; then
  AMP_FLAG="--amp"
fi

smoke_train() {
  "$PYTHON_BIN" "$SCRIPT_DIR/train_stage4.py" \
    --mode high_freq \
    --station_a "$STATION_A" \
    --station_b "$STATION_B" \
    --stage3_checkpoint "$STAGE3_CHECKPOINT" \
    --out_dir "$STAGE4_OUT"_smoke \
    --snr_mode detected \
    --integration boxcar \
    --n_train 96 --n_val 96 --epochs 2 \
    --batch_size 8 --num_workers 0 \
    --head_only_epochs 1 \
    $AMP_FLAG
}

full_train() {
  "$PYTHON_BIN" "$SCRIPT_DIR/train_stage4.py" \
    --mode high_freq \
    --station_a "$STATION_A" \
    --station_b "$STATION_B" \
    --stage3_checkpoint "$STAGE3_CHECKPOINT" \
    --out_dir "$STAGE4_OUT" \
    --snr_mode detected \
    --integration boxcar \
    --width_min 10 --width_max 100 \
    --snr_min 8 --snr_max 30 \
    --n_train 8000 --n_val 2000 --epochs 10 \
    --batch_size 32 --num_workers 0 \
    --head_only_epochs 1 \
    --head_lr 3e-4 --encoder_lr 3e-5 \
    --contrastive_weight 0.25 \
    $AMP_FLAG
}

smoke_evaluate() {
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_stage4.py" \
    --mode high_freq \
    --station_a "$STATION_A" \
    --station_b "$STATION_B" \
    --stage3_checkpoint "$STAGE3_CHECKPOINT" \
    --stage4_checkpoint \
      "$STAGE4_OUT"_smoke/model_stage4_candidate_conditioned.pt \
    --out_dir "$EVAL_OUT"_smoke \
    --widths 10,30 \
    --shapes lorentzian \
    --snr_modes detected \
    --n_per_class 20 --n_boot 200 \
    --batch_size 8 \
    $AMP_FLAG
}

full_evaluate() {
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_stage4.py" \
    --mode high_freq \
    --station_a "$STATION_A" \
    --station_b "$STATION_B" \
    --stage3_checkpoint "$STAGE3_CHECKPOINT" \
    --stage4_checkpoint \
      "$STAGE4_OUT"/model_stage4_candidate_conditioned.pt \
    --out_dir "$EVAL_OUT" \
    --widths 3,10,20,30,50,75,100 \
    --shapes lorentzian,box,gaussian \
    --snr_modes detected,power \
    --negative_mix onesided:0.38,independent:0.38,noise:0.24 \
    --n_per_class 300 --n_boot 2000 \
    --ci_level 0.95 --batch_size 32 \
    $AMP_FLAG
}

case "$ACTION" in
  preflight)
    preflight
    ;;
  smoke-train)
    preflight
    smoke_train
    ;;
  train)
    preflight
    full_train
    ;;
  smoke-evaluate)
    smoke_evaluate
    ;;
  evaluate)
    full_evaluate
    ;;
  all)
    preflight
    full_train
    full_evaluate
    ;;
  *)
    echo "Usage: $0 {preflight|smoke-train|train|smoke-evaluate|evaluate|all}"
    echo
    echo "Run train/evaluate/all only after obtaining a Sweden compute-node"
    echo "allocation; do not run them on the Ireland head node."
    exit 2
    ;;
esac
