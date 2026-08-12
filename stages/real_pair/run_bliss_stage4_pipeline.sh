#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible orchestration for Synthetic Test B.  BLISS itself is run by the
# established project command outside this wrapper because its broadened
# branch and column contract are project-specific.  This script starts from
# those exported hit/truth tables and never alters the frozen Stage-4 model.

# Input HDF5 products are opened read-only.  Shared project filesystems can
# lack POSIX lock service support (ENOLCK), so use the standard HDF5 read-only
# workaround unless the site explicitly overrides it.
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
STAGE3_CODE_DIR=${STAGE3_CODE_DIR:-$REPO_ROOT/stages/stage3}
STAGE4_CODE_DIR=${STAGE4_CODE_DIR:-$REPO_ROOT/stages/stage4}
INTEGRATION_OUT=${INTEGRATION_OUT:-$REPO_ROOT/results/test_b}
OBSERVATIONS=${OBSERVATIONS:-$INTEGRATION_OUT/observations.jsonl}
CAL_DIR=${CAL_DIR:-$INTEGRATION_OUT/calibration}
TESTB_DIR=${TESTB_DIR:-$INTEGRATION_OUT/locked_test_b}
ASSOCIATION_POLICY=${ASSOCIATION_POLICY:-$CAL_DIR/association_policy.frozen.json}
STAGE3_CHECKPOINT=${STAGE3_CHECKPOINT:-$REPO_ROOT/checkpoints/stage3/model_high_freq_broadened.pth}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-$REPO_ROOT/checkpoints/stage4/model_stage4_candidate_conditioned.pt}

mkdir -p "$CAL_DIR" "$TESTB_DIR"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$INTEGRATION_OUT/.matplotlib}
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$SCRIPT_DIR:$STAGE4_CODE_DIR:$STAGE3_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

require_vars() {
  local missing=0
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "ERROR: required environment variable $name is unset" >&2
      missing=1
    fi
  done
  [[ $missing -eq 0 ]]
}

adapt_two_station_set() {
  local role=$1
  local output_dir=$2
  local cand_a=$3
  local cand_map_a=$4
  local cand_b=$5
  local cand_map_b=$6
  local truth_a=$7
  local truth_map_a=$8
  local truth_b=$9
  local truth_map_b=${10}

  mkdir -p "$output_dir"
  "$PYTHON_BIN" "$SCRIPT_DIR/bliss_candidate_adapter.py" \
    --input "$cand_a" --mapping "$cand_map_a" --observations "$OBSERVATIONS" \
    --kind candidate --output "$output_dir/candidates_a.jsonl" \
    --recovery-links-output "$output_dir/recovery_links_a.jsonl"
  "$PYTHON_BIN" "$SCRIPT_DIR/bliss_candidate_adapter.py" \
    --input "$cand_b" --mapping "$cand_map_b" --observations "$OBSERVATIONS" \
    --kind candidate --output "$output_dir/candidates_b.jsonl" \
    --recovery-links-output "$output_dir/recovery_links_b.jsonl"
  "$PYTHON_BIN" "$SCRIPT_DIR/bliss_candidate_adapter.py" \
    --input "$truth_a" --mapping "$truth_map_a" --observations "$OBSERVATIONS" \
    --kind truth --output "$output_dir/truth_a.jsonl"
  "$PYTHON_BIN" "$SCRIPT_DIR/bliss_candidate_adapter.py" \
    --input "$truth_b" --mapping "$truth_map_b" --observations "$OBSERVATIONS" \
    --kind truth --output "$output_dir/truth_b.jsonl"

  "$PYTHON_BIN" "$SCRIPT_DIR/combine_canonical_records.py" --kind candidate \
    --inputs "$output_dir/candidates_a.jsonl" "$output_dir/candidates_b.jsonl" \
    --output "$output_dir/candidates.jsonl"
  "$PYTHON_BIN" "$SCRIPT_DIR/combine_canonical_records.py" --kind truth \
    --inputs "$output_dir/truth_a.jsonl" "$output_dir/truth_b.jsonl" \
    --output "$output_dir/truth.jsonl"
  "$PYTHON_BIN" "$SCRIPT_DIR/combine_canonical_records.py" --kind recovery_links \
    --inputs "$output_dir/recovery_links_a.jsonl" "$output_dir/recovery_links_b.jsonl" \
    --output "$output_dir/recovery_links.jsonl"
  echo "Adapted $role tables into $output_dir"
}

command=${1:-help}
case "$command" in
  preflight)
    "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR"/*.py
    (cd "$SCRIPT_DIR" && "$PYTHON_BIN" -m unittest -v test_bliss_stage4_integration.py)
    "$PYTHON_BIN" - <<'PY'
import numpy, scipy, sklearn, matplotlib
print("Integration dependencies import successfully")
PY
    if [[ -f "$STAGE4_CODE_DIR/test_stage4_core.py" ]]; then
      (cd "$STAGE4_CODE_DIR" && "$PYTHON_BIN" -m unittest -v test_stage4_core.py)
    fi
    ;;

  observations)
    require_vars OBSERVATION_CSV
    extra=()
    if [[ "${ALLOW_NORMALIZED_PROXY:-0}" == "1" ]]; then
      extra+=(--allow-normalized-proxy)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/make_observation_manifest.py" \
      --input-csv "$OBSERVATION_CSV" --output "$OBSERVATIONS" \
      ${extra[@]+"${extra[@]}"}
    ;;

  adapt-calibration)
    require_vars CAL_CANDIDATES_A CAL_CANDIDATE_MAP_A CAL_CANDIDATES_B \
      CAL_CANDIDATE_MAP_B CAL_TRUTH_A CAL_TRUTH_MAP_A CAL_TRUTH_B CAL_TRUTH_MAP_B
    adapt_two_station_set calibration "$CAL_DIR" \
      "$CAL_CANDIDATES_A" "$CAL_CANDIDATE_MAP_A" \
      "$CAL_CANDIDATES_B" "$CAL_CANDIDATE_MAP_B" \
      "$CAL_TRUTH_A" "$CAL_TRUTH_MAP_A" \
      "$CAL_TRUTH_B" "$CAL_TRUTH_MAP_B"
    ;;

  estimate-calibration)
    "$PYTHON_BIN" "$SCRIPT_DIR/estimate_bliss_recovery.py" \
      --candidates "$CAL_DIR/candidates.jsonl" \
      --truth "$CAL_DIR/truth.jsonl" \
      --recovery-links "$CAL_DIR/recovery_links.jsonl" \
      --dataset-role calibration --out-dir "$CAL_DIR/recovery_audit" \
      --matching auto --widths 10,20,30,50,75,100 \
      --snr-edges 8,10,12,16,20,30 --min-policy-matches "${MIN_POLICY_MATCHES:-100}"
    ;;

  freeze-policy)
    require_vars POLICY_REVIEWER POLICY_NOTES
    "$PYTHON_BIN" "$SCRIPT_DIR/freeze_association_policy.py" \
      --input "$CAL_DIR/recovery_audit/association_policy.empirical_draft.json" \
      --output "$ASSOCIATION_POLICY" --reviewer "$POLICY_REVIEWER" \
      --notes "$POLICY_NOTES" --acknowledge-locked-test
    ;;

  freeze-preregistration)
    require_vars PREREG_REVIEWER PREREG_NOTES TESTB_DESIGN TESTB_SEED
    test -s "$STAGE3_CHECKPOINT"
    test -s "$STAGE4_CHECKPOINT"
    "$PYTHON_BIN" "$SCRIPT_DIR/freeze_test_b_preregistration.py" \
      --input "$SCRIPT_DIR/config/synthetic_test_b_preregistration.json" \
      --output "$TESTB_DIR/SYNTHETIC_TEST_B_PREREGISTERED.json" \
      --test-b-design "$TESTB_DESIGN" --test-b-seed "$TESTB_SEED" \
      --association-policy "$ASSOCIATION_POLICY" \
      --stage3-checkpoint "$STAGE3_CHECKPOINT" \
      --stage4-checkpoint "$STAGE4_CHECKPOINT" \
      --repository "$STAGE4_CODE_DIR" --reviewer "$PREREG_REVIEWER" \
      --notes "$PREREG_NOTES" --acknowledge-before-test
    ;;

  adapt-test-b)
    require_vars TESTB_CANDIDATES_A TESTB_CANDIDATE_MAP_A TESTB_CANDIDATES_B \
      TESTB_CANDIDATE_MAP_B TESTB_TRUTH_A TESTB_TRUTH_MAP_A TESTB_TRUTH_B TESTB_TRUTH_MAP_B
    adapt_two_station_set locked_test_b "$TESTB_DIR/canonical" \
      "$TESTB_CANDIDATES_A" "$TESTB_CANDIDATE_MAP_A" \
      "$TESTB_CANDIDATES_B" "$TESTB_CANDIDATE_MAP_B" \
      "$TESTB_TRUTH_A" "$TESTB_TRUTH_MAP_A" \
      "$TESTB_TRUTH_B" "$TESTB_TRUTH_MAP_B"
    ;;

  union)
    test -s "$TESTB_DIR/SYNTHETIC_TEST_B_PREREGISTERED.json"
    extra=()
    if [[ "${ALLOW_NORMALIZED_PROXY:-0}" == "1" ]]; then
      extra+=(--allow-normalized-proxy)
    fi
    if [[ "${FAIL_ON_BLOCK_DISAGREEMENT:-1}" == "1" ]]; then
      extra+=(--fail-on-block-disagreement)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/candidate_union.py" \
      --observations "$OBSERVATIONS" \
      --candidate-files "$TESTB_DIR/canonical/candidates.jsonl" \
      --policy "$ASSOCIATION_POLICY" --output "$TESTB_DIR/candidate_union.jsonl" \
      --route-low-hz 10 --route-high-hz 100 ${extra[@]+"${extra[@]}"}
    ;;

  extract)
    extra=()
    if [[ "${STRICT_EXTRACTION:-1}" == "1" ]]; then
      extra+=(--fail-on-exclusion)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/extract_candidate_pairs.py" \
      --observations "$OBSERVATIONS" --union "$TESTB_DIR/candidate_union.jsonl" \
      --out-dir "$TESTB_DIR/extracted" --n-rows 16 --n-cols 1024 \
      --edge-guard-widths 4 --qa-count "${QA_COUNT:-20}" \
      ${extra[@]+"${extra[@]}"}
    ;;

  infer)
    test -s "$STAGE3_CHECKPOINT"
    test -s "$STAGE4_CHECKPOINT"
    extra=()
    if [[ "${USE_AMP:-0}" == "1" ]]; then
      extra+=(--amp)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/run_stage4_on_bliss.py" \
      --pair-manifest "$TESTB_DIR/extracted/pair_manifest.jsonl" \
      --stage3-checkpoint "$STAGE3_CHECKPOINT" --stage4-checkpoint "$STAGE4_CHECKPOINT" \
      --stage4-code-dir "$STAGE4_CODE_DIR" --out-dir "$TESTB_DIR/inference" \
      --batch-size "${BATCH_SIZE:-32}" --qa-count "${QA_COUNT:-20}" \
      ${extra[@]+"${extra[@]}"}
    ;;

  reveal-labels)
    test -s "$TESTB_DIR/inference/stage4_bliss_predictions.jsonl"
    mkdir -p "$TESTB_DIR/labeling"
    "$PYTHON_BIN" "$SCRIPT_DIR/link_test_b_recoveries.py" \
      --candidates "$TESTB_DIR/canonical/candidates.jsonl" \
      --truth "$TESTB_DIR/canonical/truth.jsonl" \
      --policy "$ASSOCIATION_POLICY" \
      --predictions "$TESTB_DIR/inference/stage4_bliss_predictions.jsonl" \
      --exact-links "$TESTB_DIR/canonical/recovery_links.jsonl" \
      --mode "${RECOVERY_LINK_MODE:-auto}" \
      --output "$TESTB_DIR/labeling/recovery_links.final.jsonl" \
      --acknowledge-post-inference
    "$PYTHON_BIN" "$SCRIPT_DIR/build_synthetic_test_b_labels.py" \
      --union "$TESTB_DIR/candidate_union.jsonl" \
      --truth "$TESTB_DIR/canonical/truth.jsonl" \
      --recovery-links "$TESTB_DIR/labeling/recovery_links.final.jsonl" \
      --preregistration "$TESTB_DIR/SYNTHETIC_TEST_B_PREREGISTERED.json" \
      --output "$TESTB_DIR/synthetic_test_b_labels.jsonl" \
      --expected-population "${TESTB_POPULATION:-detected_conditioned}"
    ;;

  evaluate)
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_bliss_stage4.py" \
      --predictions "$TESTB_DIR/inference/stage4_bliss_predictions.jsonl" \
      --labels "$TESTB_DIR/synthetic_test_b_labels.jsonl" \
      --events "$TESTB_DIR/synthetic_test_b_events.jsonl" \
      --preregistration "$TESTB_DIR/SYNTHETIC_TEST_B_PREREGISTERED.json" \
      --out-dir "$TESTB_DIR/evaluation" --n-boot "${N_BOOT:-5000}" \
      --ci-level 0.95 --bootstrap-unit auto --target-auc 0.80 \
      --width-min 10 --width-max 100 \
      --primary-population "${TESTB_POPULATION:-detected_conditioned}"
    ;;

  freeze-results)
    "$PYTHON_BIN" "$SCRIPT_DIR/freeze_test_b_artifacts.py" \
      --run-root "$TESTB_DIR" --repository "$STAGE4_CODE_DIR" \
      --output "$TESTB_DIR/SYNTHETIC_TEST_B_FROZEN.json" \
      --acknowledge-no-further-tuning
    ;;

  midres-audit)
    require_vars HIGH_RESOLUTION_FILE MID_RESOLUTION_FILE
    "$PYTHON_BIN" "$SCRIPT_DIR/inspect_midresolution_handoff.py" \
      --high-resolution-file "$HIGH_RESOLUTION_FILE" \
      --mid-resolution-file "$MID_RESOLUTION_FILE" \
      --output-json "$INTEGRATION_OUT/midresolution_handoff.json"
    ;;

  help|*)
    cat <<'USAGE'
Usage: ./run_bliss_stage4_pipeline.sh COMMAND

Commands, in order:
  preflight             CPU-only integration and existing Stage-4 tests
  observations          inspect headers and create the two-station manifest
  adapt-calibration     adapt independent BLISS calibration hits and truth
  estimate-calibration  measure recovered-parameter errors; draft policy
  freeze-policy         review/freeze policy without Test-B labels
  freeze-preregistration freeze design, seed, policy and model hashes
  adapt-test-b          adapt the untouched locked Test-B BLISS exports
  union                 build A union B and retain one-station detections
  extract               extract the corresponding 16x1024 station pairs
  infer                 label-blind frozen Stage-4 inference
  reveal-labels         join segregated truth only after inference exists
  evaluate              paired statistics and separate pipeline denominators
  freeze-results        checksums, environment and Git state
  midres-audit          verify actual mid-resolution header and routing

See README_BLISS_STAGE4_INTEGRATION.md for required environment variables and
the non-negotiable separation between calibration, locked Test B, and real
Ireland-Sweden external validation.
USAGE
    ;;
esac
