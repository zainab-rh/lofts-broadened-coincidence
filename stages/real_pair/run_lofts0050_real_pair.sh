#!/usr/bin/env bash
set -Eeuo pipefail

# Real, simultaneous, barycentric Ireland--Sweden processing pilot for
# LOFTS0050 part000.  This runner never reveals/invents labels and never reports
# real-data AUC.  The labeled Synthetic-Test-B runner remains separate.

# The Sweden project files live on a shared filesystem whose lock service may
# return ENOLCK (errno 37).  Every HDF5 access in this runner is read-only, so
# disabling HDF5's advisory file locking is safe and avoids a site-specific
# failure without changing any bytes or scientific values.
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
STAGE3_CODE_DIR=${STAGE3_CODE_DIR:-$REPO_ROOT/stages/stage3}
STAGE4_CODE_DIR=${STAGE4_CODE_DIR:-$REPO_ROOT/stages/stage4}
REAL_OUT=${REAL_OUT:-$REPO_ROOT/results/real_lofts0050}

# Site-specific storage and software locations are supplied by the job
# environment. Keeping them out of source avoids encoding one user's account
# layout in the reproducible workflow.
IRL_H5=${IRL_H5:-}
SWE_H5=${SWE_H5:-}
IRL_BANK_CSV_EXTERNAL=${IRL_BANK_CSV_EXTERNAL:-}
IRL_RAW_CSV_EXTERNAL=${IRL_RAW_CSV_EXTERNAL:-}
IRL_PER_TEMPLATE_CSV_EXTERNAL=${IRL_PER_TEMPLATE_CSV_EXTERNAL:-}

NAOISE_REPO=${NAOISE_REPO:-}
NAOISE_SCRIPT=${NAOISE_SCRIPT:-${NAOISE_REPO:+$NAOISE_REPO/search/blind_hit_finder.py}}
BLISS_PYTHON=${BLISS_PYTHON:-}
EXPECTED_NAOISE_COMMIT=${EXPECTED_NAOISE_COMMIT:-dee329949384f0a0ddb6306d8bbbc2b0db74011a}
EXPECTED_NAOISE_SCRIPT_SHA256=${EXPECTED_NAOISE_SCRIPT_SHA256:-50d3c512f68946fd1786ecc122441341382d46ec163ef4c009c40d8692b26c6a}
BLISS_REPO=${BLISS_REPO:-}
EXPECTED_BLISS_COMMIT=${EXPECTED_BLISS_COMMIT:-2b98afe960f13ee7e467aca499576d87ee7502f5}

# CPU is the explicit, reproducible default for both the pinned blind search
# and Stage-4 inference. Set BLIND_SEARCH_BACKEND=cuda or auto only on a node
# with a working CUDA runtime. These limits avoid CPU oversubscription.
BLIND_SEARCH_BACKEND=${BLIND_SEARCH_BACKEND:-cpu}
STAGE4_DEVICE=${STAGE4_DEVICE:-cpu}
CPU_THREADS=${CPU_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$CPU_THREADS}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS}

CATALOG_DIR=$REAL_OUT/catalogs
GENERATED_IRL_DIR=$CATALOG_DIR/irl_generated
SWE_DIR=$CATALOG_DIR/swe
GENERATED_IRL_BANK_CSV=$GENERATED_IRL_DIR/LOFTS0050_IRL_bank.csv
GENERATED_IRL_RAW_CSV=$GENERATED_IRL_DIR/LOFTS0050_IRL_bank.raw.csv
GENERATED_IRL_PER_TEMPLATE_CSV=$GENERATED_IRL_DIR/LOFTS0050_IRL_pertemplate.csv
SWE_BANK_CSV=$SWE_DIR/LOFTS0050_SWE_bank.csv
SWE_RAW_CSV=$SWE_DIR/LOFTS0050_SWE_bank.raw.csv
SWE_PER_TEMPLATE_CSV=$SWE_DIR/LOFTS0050_SWE_pertemplate.csv
CPU_SMOKE_DIR=$CATALOG_DIR/cpu_smoke

if [[ "${USE_GENERATED_IRL:-0}" == "1" ]]; then
  IRL_BANK_CSV=$GENERATED_IRL_BANK_CSV
  IRL_RAW_CSV=$GENERATED_IRL_RAW_CSV
  IRL_PER_TEMPLATE_CSV=$GENERATED_IRL_PER_TEMPLATE_CSV
else
  IRL_BANK_CSV=$IRL_BANK_CSV_EXTERNAL
  IRL_RAW_CSV=$IRL_RAW_CSV_EXTERNAL
  IRL_PER_TEMPLATE_CSV=$IRL_PER_TEMPLATE_CSV_EXTERNAL
fi

PROVENANCE_DIR=$REAL_OUT/provenance
NAOISE_PROVENANCE=$PROVENANCE_DIR/naoise_blind_hit_finder.json
BLISS_PROVENANCE=$PROVENANCE_DIR/bliss_backend.json
OBSERVATION_CSV=$REAL_OUT/config/observations_lofts0050.csv
OBSERVATIONS=$REAL_OUT/observations.jsonl
CANONICAL_DIR=$REAL_OUT/canonical
IRL_CANONICAL=$CANONICAL_DIR/candidates_irl.jsonl
SWE_CANONICAL=$CANONICAL_DIR/candidates_swe.jsonl
POLICY_DRAFT=$REAL_OUT/policy/real_pair_policy.draft.json
POLICY_FROZEN=$REAL_OUT/policy/real_pair_policy.frozen.json
UNION=$REAL_OUT/union/real_candidate_union.jsonl
EXTRACTED=$REAL_OUT/extracted/primary
CONTROLS=$REAL_OUT/extracted/controls
PRIMARY_INFERENCE=$REAL_OUT/inference/primary
CONTROL_INFERENCE=$REAL_OUT/inference/controls
ANALYSIS_DIR=$REAL_OUT/analysis
ROADMAP_PLOT_DIR=$ANALYSIS_DIR/roadmap_plots
PRESENTATION_PLOT_DIR=${PRESENTATION_PLOT_DIR:-$ANALYSIS_DIR/presentation_figures}

STAGE3_CHECKPOINT=${STAGE3_CHECKPOINT:-$REPO_ROOT/checkpoints/stage3/model_high_freq_broadened.pth}
STAGE4_CHECKPOINT=${STAGE4_CHECKPOINT:-$REPO_ROOT/checkpoints/stage4/model_stage4_candidate_conditioned.pt}
TESTA_FROZEN_DIR=${TESTA_FROZEN_DIR:-$REPO_ROOT/results/test_a/evaluation}
TESTA_REANALYSIS_DIR=${TESTA_REANALYSIS_DIR:-$REPO_ROOT/results/test_a/pair_export}
TESTA_DIRECT_COMPARISON=${TESTA_DIRECT_COMPARISON:-$REPO_ROOT/results/test_a/test_a_stage4_minus_filter.json}
TESTA_STATION_A=${TESTA_STATION_A:-/datax2/projects/LOFTS/2025-05-14/LOFTS0192/LOFTS0192.rawspec.0000.fil}
TESTA_STATION_B=${TESTA_STATION_B:-/datax2/projects/LOFTS/2025-05-21/LOFTS0199/LOFTS0199.rawspec.0000.fil}

mkdir -p "$REAL_OUT" "$PROVENANCE_DIR" "$CATALOG_DIR" "$CANONICAL_DIR"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$REAL_OUT/.matplotlib}
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$SCRIPT_DIR:$STAGE4_CODE_DIR:$STAGE3_CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

require_vars() {
  local missing=0 name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "ERROR: required environment variable $name is unset" >&2
      missing=1
    fi
  done
  [[ $missing -eq 0 ]]
}

require_files() {
  local path
  for path in "$@"; do
    if [[ ! -s "$path" ]]; then
      echo "ERROR: required non-empty file is missing: $path" >&2
      return 1
    fi
  done
}

run_blind_search() {
  local station=$1 input_h5=$2 output_csv=$3 per_template_csv=$4 log=$5
  local coarse_channel=${6:-}
  local runtime_provenance
  require_vars BLISS_PYTHON NAOISE_SCRIPT
  runtime_provenance=$(dirname -- "$log")/blind_search_runtime.json
  mkdir -p "$(dirname -- "$output_csv")" "$(dirname -- "$log")"
  require_files "$NAOISE_PROVENANCE" "$BLISS_PROVENANCE" \
    "$input_h5" "$NAOISE_SCRIPT" "$SCRIPT_DIR/run_pinned_naoise_search.py"
  local search_command=(
    "$BLISS_PYTHON" -u "$SCRIPT_DIR/run_pinned_naoise_search.py"
    --script "$NAOISE_SCRIPT"
    --expected-script-sha256 "$EXPECTED_NAOISE_SCRIPT_SHA256"
    --backend "$BLIND_SEARCH_BACKEND"
    --runtime-provenance "$runtime_provenance"
    --
    "$input_h5"
    --fpc 262144
    --drift 0.2
    --drift-res 1
    --floor 20
    --width-tol 0.10
    --wide-thresh 120
    --rfi-ratio 8.0
    --rfi-merge-khz 20.0
    --min-sep 300
    --refine-half 400
    --obs LOFTS0050_part000
    --station "$station"
    --csv "$output_csv"
    --per-template-csv "$per_template_csv"
  )
  if [[ -n "$coarse_channel" ]]; then
    search_command+=(--coarse "$coarse_channel")
  fi
  {
    printf 'EXACT_COMMAND='
    printf ' %q' "${search_command[@]}"
    printf '\n'
  } | tee "$log"
  "${search_command[@]}" 2>&1 | tee -a "$log"
  local raw_csv=${output_csv%.csv}.raw.csv
  require_files "$output_csv" "$raw_csv" "$per_template_csv" "$log" \
    "$runtime_provenance"
}

run_inference() {
  local pair_manifest=$1 output_dir=$2
  require_files "$pair_manifest" "$STAGE3_CHECKPOINT" "$STAGE4_CHECKPOINT"
  local extra=() device_args=() batch_size
  batch_size=${BATCH_SIZE:-}
  if [[ -z "$batch_size" ]]; then
    if [[ "$STAGE4_DEVICE" == cpu ]]; then
      batch_size=8
    else
      batch_size=32
    fi
  fi
  if [[ "$STAGE4_DEVICE" != auto ]]; then
    device_args=(--device "$STAGE4_DEVICE")
  fi
  if [[ "${USE_AMP:-0}" == "1" ]]; then
    if [[ "$STAGE4_DEVICE" == cuda* || "$STAGE4_DEVICE" == auto ]]; then
      extra+=(--amp)
    else
      echo "WARNING: USE_AMP=1 ignored because Stage-4 is explicitly on CPU." >&2
    fi
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/run_stage4_on_bliss.py" \
    --pair-manifest "$pair_manifest" \
    --stage3-checkpoint "$STAGE3_CHECKPOINT" \
    --stage4-checkpoint "$STAGE4_CHECKPOINT" \
    --stage4-code-dir "$STAGE4_CODE_DIR" \
    --out-dir "$output_dir" \
    --dataset-role unlabeled_real_pair \
    --acknowledge-synthetic-threshold-is-exploratory \
    --batch-size "$batch_size" \
    --cpu-threads "$CPU_THREADS" \
    --qa-count "${QA_COUNT:-20}" \
    ${device_args[@]+"${device_args[@]}"} ${extra[@]+"${extra[@]}"}
}

command=${1:-help}
case "$command" in
  preflight)
    # Bash 4.3 treats an empty array expansion as unbound under `set -u`.
    # Exercise the compatibility form used throughout both runners.
    bash -uc 'extra=(); : ${extra[@]+"${extra[@]}"}'
    "$PYTHON_BIN" -m py_compile "$SCRIPT_DIR"/*.py
    (cd "$SCRIPT_DIR" && "$PYTHON_BIN" -m unittest -v \
      test_bliss_stage4_integration.py test_real_pair_v3.py \
      test_cpu_and_plots_v3_2.py)
    "$PYTHON_BIN" - <<'PY'
import h5py, hdf5plugin, matplotlib, numpy, scipy, sklearn
print("Real-pair dependencies import successfully")
PY
    if [[ -f "$STAGE4_CODE_DIR/test_stage4_core.py" ]]; then
      (cd "$STAGE4_CODE_DIR" && "$PYTHON_BIN" -m unittest -v test_stage4_core.py)
    fi
    ;;

  inputs)
    require_vars IRL_H5 SWE_H5 IRL_RAW_CSV IRL_PER_TEMPLATE_CSV
    require_files "$IRL_H5" "$SWE_H5" "$IRL_RAW_CSV" "$IRL_PER_TEMPLATE_CSV"
    "$PYTHON_BIN" - "$IRL_H5" "$SWE_H5" <<'PY'
import sys
import h5py
import hdf5plugin  # registers bitshuffle/LZ4 filters before the dataset read
import numpy as np

for path in sys.argv[1:]:
    with h5py.File(path, "r") as handle:
        sample = np.asarray(handle["data"][0, 0, :8])
        if sample.shape != (8,) or not np.isfinite(sample).all():
            raise SystemExit("invalid HDF5 sample read: %s" % path)
    print("Exact HDF5 data read OK:", path)
PY
    echo "IRL H5: $IRL_H5"
    echo "SWE H5: $SWE_H5"
    echo "IRL raw catalog: $IRL_RAW_CSV"
    echo "IRL per-template catalog: $IRL_PER_TEMPLATE_CSV"
    head -n 2 "$IRL_RAW_CSV"
    head -n 2 "$IRL_PER_TEMPLATE_CSV"
    ;;

  verify-naoise)
    require_vars NAOISE_REPO NAOISE_SCRIPT
    extra=()
    if [[ "${ALLOW_DIRTY_NAOISE_CHECKOUT:-0}" == "1" ]]; then
      extra+=(--allow-dirty)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/verify_naoise_checkout.py" \
      --repository "$NAOISE_REPO" --script "$NAOISE_SCRIPT" \
      --expected-commit "$EXPECTED_NAOISE_COMMIT" \
      --output "$NAOISE_PROVENANCE" ${extra[@]+"${extra[@]}"}
    ;;

  verify-backend)
    require_vars BLISS_REPO BLISS_PYTHON
    extra=()
    if [[ "${ALLOW_DIRTY_BLISS_CHECKOUT:-0}" == "1" ]]; then
      extra+=(--allow-dirty)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/verify_bliss_backend.py" \
      --repository "$BLISS_REPO" --python "$BLISS_PYTHON" \
      --expected-commit "$EXPECTED_BLISS_COMMIT" \
      --output "$BLISS_PROVENANCE" ${extra[@]+"${extra[@]}"}
    ;;

  search-backend)
    require_vars BLISS_PYTHON NAOISE_SCRIPT
    require_files "$NAOISE_SCRIPT" "$SCRIPT_DIR/run_pinned_naoise_search.py"
    "$BLISS_PYTHON" "$SCRIPT_DIR/run_pinned_naoise_search.py" \
      --script "$NAOISE_SCRIPT" \
      --expected-script-sha256 "$EXPECTED_NAOISE_SCRIPT_SHA256" \
      --backend "$BLIND_SEARCH_BACKEND" --probe-only
    ;;

  observations)
    require_vars IRL_H5 SWE_H5 BARYCENTRIC_TOOL BARYCENTRIC_VERSION
    require_files "$IRL_H5" "$SWE_H5" "$NAOISE_PROVENANCE"
    extra=()
    if [[ "${ALLOW_UNVERIFIED_BARYCENTRIC_PROVENANCE:-0}" == "1" ]]; then
      extra+=(--allow-unverified-barycentric-provenance)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_lofts0050_observation_csv.py" \
      --irl-h5 "$IRL_H5" --swe-h5 "$SWE_H5" \
      --naoise-provenance "$NAOISE_PROVENANCE" --output "$OBSERVATION_CSV" \
      --barycentric-tool "$BARYCENTRIC_TOOL" \
      --barycentric-version "$BARYCENTRIC_VERSION" \
      ${extra[@]+"${extra[@]}"}
    manifest_extra=(--require-barycentric-provenance)
    if [[ "${ALLOW_UNVERIFIED_BARYCENTRIC_PROVENANCE:-0}" == "1" ]]; then
      manifest_extra=()
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/make_observation_manifest.py" \
      --input-csv "$OBSERVATION_CSV" --output "$OBSERVATIONS" \
      ${manifest_extra[@]+"${manifest_extra[@]}"}
    ;;

  search-swe)
    run_blind_search SWE "$SWE_H5" "$SWE_BANK_CSV" \
      "$SWE_PER_TEMPLATE_CSV" "$SWE_DIR/blind_hit_finder.log"
    ;;

  search-smoke-swe)
    smoke_coarse=${CPU_SMOKE_COARSE:-0}
    smoke_dir=$CPU_SMOKE_DIR/swe_coarse_${smoke_coarse}
    run_blind_search SWE "$SWE_H5" \
      "$smoke_dir/LOFTS0050_SWE_bank.csv" \
      "$smoke_dir/LOFTS0050_SWE_pertemplate.csv" \
      "$smoke_dir/blind_hit_finder.log" "$smoke_coarse"
    echo "CPU smoke test completed for coarse channel $smoke_coarse."
    echo "These partial-band catalogs are diagnostic only and are never used by adapt."
    ;;

  bank-audit)
    require_files "$OBSERVATIONS"
    "$PYTHON_BIN" "$SCRIPT_DIR/audit_stage4_bank_coverage.py" \
      --observations "$OBSERVATIONS" \
      --output-json "$REAL_OUT/analysis/bank_coverage_audit.json" \
      --output-markdown "$REAL_OUT/analysis/bank_coverage_audit.md"
    "$0" plot-roadmap
    ;;

  plot-testa)
    require_files "$TESTA_FROZEN_DIR/stage4_width_results.csv" \
      "$TESTA_FROZEN_DIR/stage4_evaluation.json" \
      "$SCRIPT_DIR/plot_stage4_frozen_results.py"
    extra=()
    paired_filter=$TESTA_DIRECT_COMPARISON
    if [[ ! -s "$paired_filter" && -s "$TESTA_REANALYSIS_DIR/test_a_stage4_minus_filter.json" ]]; then
      paired_filter=$TESTA_REANALYSIS_DIR/test_a_stage4_minus_filter.json
    fi
    if [[ -s "$paired_filter" ]]; then
      extra+=(--paired-filter-json "$paired_filter")
    else
      echo "WARNING: locked paired Stage-4-minus-filter JSON was not found; the paired incremental-value figure will be omitted." >&2
      echo "Set TESTA_DIRECT_COMPARISON to the completed locked result and rerun plot-testa." >&2
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_stage4_frozen_results.py" \
      --results-csv "$TESTA_FROZEN_DIR/stage4_width_results.csv" \
      --evaluation-json "$TESTA_FROZEN_DIR/stage4_evaluation.json" \
      --out-dir "$TESTA_FROZEN_DIR/publication_figures" \
      --width-min 10 --width-max 100 --formats png,pdf --dpi 300 \
      ${extra[@]+"${extra[@]}"}
    ;;

  testa-incremental)
    require_files "$TESTA_FROZEN_DIR/test_manifest.jsonl" \
      "$TESTA_FROZEN_DIR/stage4_width_results.csv" \
      "$TESTA_STATION_A" "$TESTA_STATION_B" \
      "$STAGE3_CHECKPOINT" "$STAGE4_CHECKPOINT" \
      "$STAGE4_CODE_DIR/evaluate_stage4.py"
    if [[ -e "$TESTA_REANALYSIS_DIR" ]]; then
      echo "ERROR: Test-A reanalysis directory already exists: $TESTA_REANALYSIS_DIR" >&2
      exit 1
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/rerun_frozen_test_a_pair_export.py" \
      --stage4-code-dir "$STAGE4_CODE_DIR" \
      --reference-manifest "$TESTA_FROZEN_DIR/test_manifest.jsonl" \
      --reference-results-csv "$TESTA_FROZEN_DIR/stage4_width_results.csv" \
      --pair-output "$TESTA_REANALYSIS_DIR/test_a_pair_scores.jsonl" \
      --direct-output "$TESTA_REANALYSIS_DIR/test_a_stage4_minus_filter.json" \
      --acknowledge-locked-posthoc-analysis \
      --mode high_freq --station_a "$TESTA_STATION_A" --station_b "$TESTA_STATION_B" \
      --stage3_checkpoint "$STAGE3_CHECKPOINT" --stage4_checkpoint "$STAGE4_CHECKPOINT" \
      --out_dir "$TESTA_REANALYSIS_DIR" \
      --widths 3,10,20,30,50,75,100 \
      --shapes lorentzian,box,gaussian --snr_modes detected,power \
      --n_per_class 300 --n_boot 2000 --ci_level 0.95 \
      --batch_size "${BATCH_SIZE:-32}" --device cpu --seed 20260804
    ;;

  search-irl)
    run_blind_search IRL "$IRL_H5" "$GENERATED_IRL_BANK_CSV" \
      "$GENERATED_IRL_PER_TEMPLATE_CSV" "$GENERATED_IRL_DIR/blind_hit_finder.log"
    echo "Re-run subsequent commands with: export USE_GENERATED_IRL=1"
    ;;

  search-both)
    run_blind_search IRL "$IRL_H5" "$GENERATED_IRL_BANK_CSV" \
      "$GENERATED_IRL_PER_TEMPLATE_CSV" "$GENERATED_IRL_DIR/blind_hit_finder.log"
    run_blind_search SWE "$SWE_H5" "$SWE_BANK_CSV" \
      "$SWE_PER_TEMPLATE_CSV" "$SWE_DIR/blind_hit_finder.log"
    echo "Re-run subsequent commands with: export USE_GENERATED_IRL=1"
    ;;

  adapt)
    require_files "$OBSERVATIONS" "$IRL_RAW_CSV" "$IRL_PER_TEMPLATE_CSV" \
      "$SWE_RAW_CSV" "$SWE_PER_TEMPLATE_CSV"
    "$PYTHON_BIN" "$SCRIPT_DIR/adapt_naoise_blind_catalog.py" \
      --raw-csv "$IRL_RAW_CSV" --per-template-csv "$IRL_PER_TEMPLATE_CSV" \
      --observations "$OBSERVATIONS" --output "$IRL_CANONICAL" \
      --observation-id LOFTS0050_IRL_part000 \
      --simultaneous-group-id LOFTS0050_part000 --station-id IRL \
      --expected-obs-label LOFTS0050_part000 \
      --search-git-commit "$EXPECTED_NAOISE_COMMIT"
    "$PYTHON_BIN" "$SCRIPT_DIR/adapt_naoise_blind_catalog.py" \
      --raw-csv "$SWE_RAW_CSV" --per-template-csv "$SWE_PER_TEMPLATE_CSV" \
      --observations "$OBSERVATIONS" --output "$SWE_CANONICAL" \
      --observation-id LOFTS0050_SWE_part000 \
      --simultaneous-group-id LOFTS0050_part000 --station-id SWE \
      --expected-obs-label LOFTS0050_part000 \
      --search-git-commit "$EXPECTED_NAOISE_COMMIT"
    ;;

  derive-policy)
    require_files "$OBSERVATIONS" "$IRL_CANONICAL" "$SWE_CANONICAL"
    mkdir -p "$(dirname -- "$POLICY_DRAFT")"
    "$PYTHON_BIN" "$SCRIPT_DIR/derive_real_pair_policy.py" \
      --observations "$OBSERVATIONS" --output "$POLICY_DRAFT" \
      --width-mode native --frequency-base-channels 2 \
      --frequency-width-sum-fraction 1.0 --drift-tolerance-bins 2 \
      --coverage-guard-fwhm-fraction 0.5 --maximum-component-nodes 512 \
      --control-shifts-hz "${CONTROL_SHIFTS_HZ:--300000,-100000,100000,300000}" \
      --control-minimum-per-pair "${MINIMUM_CONTROLS_PER_PAIR:-2}" \
      --control-candidate-exclusion-widths \
        "${CONTROL_CANDIDATE_EXCLUSION_WIDTHS:-4}" \
      --control-candidate-exclusion-base-hz \
        "${CONTROL_CANDIDATE_EXCLUSION_BASE_HZ:-6}" \
      --control-edge-guard-widths "${CONTROL_EDGE_GUARD_WIDTHS:-4}"
    ;;

  freeze-policy)
    require_vars POLICY_REVIEWER POLICY_NOTES
    require_files "$POLICY_DRAFT"
    "$PYTHON_BIN" "$SCRIPT_DIR/freeze_real_pair_policy.py" \
      --input "$POLICY_DRAFT" --output "$POLICY_FROZEN" \
      --reviewer "$POLICY_REVIEWER" --notes "$POLICY_NOTES" \
      --acknowledge-before-union-and-scores
    ;;

  union)
    require_files "$OBSERVATIONS" "$IRL_CANONICAL" "$SWE_CANONICAL" "$POLICY_FROZEN"
    mkdir -p "$(dirname -- "$UNION")"
    "$PYTHON_BIN" "$SCRIPT_DIR/real_candidate_union.py" \
      --observations "$OBSERVATIONS" \
      --candidate-files "$IRL_CANONICAL" "$SWE_CANONICAL" \
      --policy "$POLICY_FROZEN" --output "$UNION"
    ;;

  extract)
    require_files "$OBSERVATIONS" "$UNION"
    extra=(--fail-on-exclusion)
    if [[ "${OVERWRITE_PRIMARY_EXTRACTION:-0}" == "1" ]]; then
      extra+=(--overwrite)
    fi
    if [[ "${ALLOW_MASKED_DATA:-0}" == "1" ]]; then
      extra+=(--allow-masked-data)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/extract_candidate_pairs.py" \
      --observations "$OBSERVATIONS" --union "$UNION" \
      --out-dir "$EXTRACTED" --include-routes high_resolution_stage4 \
      --n-rows 16 --n-cols 1024 --edge-guard-widths 4 \
      --qa-count "${QA_COUNT:-20}" --minimum-extracted 1 \
      ${extra[@]+"${extra[@]}"}
    ;;

  infer)
    run_inference "$EXTRACTED/pair_manifest.jsonl" "$PRIMARY_INFERENCE"
    ;;

  controls)
    require_files "$OBSERVATIONS" "$EXTRACTED/pair_manifest.jsonl" \
      "$IRL_CANONICAL" "$SWE_CANONICAL" "$POLICY_FROZEN"
    extra=()
    if [[ "${OVERWRITE_CONTROL_EXTRACTION:-0}" == "1" ]]; then
      extra+=(--overwrite)
    fi
    if [[ "${ALLOW_MASKED_DATA:-0}" == "1" ]]; then
      extra+=(--allow-masked-data)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/extract_shifted_controls.py" \
      --observations "$OBSERVATIONS" \
      --pair-manifest "$EXTRACTED/pair_manifest.jsonl" \
      --candidate-files "$IRL_CANONICAL" "$SWE_CANONICAL" \
      --policy "$POLICY_FROZEN" \
      --out-dir "$CONTROLS" \
      --qa-count "${CONTROL_QA_COUNT:-12}" ${extra[@]+"${extra[@]}"}
    ;;

  infer-controls)
    run_inference "$CONTROLS/control_pair_manifest.jsonl" "$CONTROL_INFERENCE"
    ;;

  analyze)
    require_files "$IRL_CANONICAL" "$SWE_CANONICAL" "$UNION" \
      "$PRIMARY_INFERENCE/stage4_bliss_predictions.jsonl" \
      "$CONTROL_INFERENCE/stage4_bliss_predictions.jsonl"
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_real_pair.py" \
      --candidate-files "$IRL_CANONICAL" "$SWE_CANONICAL" \
      --union "$UNION" \
      --primary-predictions "$PRIMARY_INFERENCE/stage4_bliss_predictions.jsonl" \
      --control-predictions "$CONTROL_INFERENCE/stage4_bliss_predictions.jsonl" \
      --out-dir "$ANALYSIS_DIR" --n-boot "${N_BOOT:-5000}" \
      --seed "${ANALYSIS_SEED:-20260810}" --top-n "${TOP_N:-50}"
    "$0" plot-roadmap
    ;;

  plot-roadmap)
    require_files "$SCRIPT_DIR/plot_real_pair_roadmap.py"
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_real_pair_roadmap.py" \
      --observations-summary "$REAL_OUT/observations.summary.json" \
      --bank-audit "$REAL_OUT/analysis/bank_coverage_audit.json" \
      --candidate-files "$IRL_CANONICAL" "$SWE_CANONICAL" \
      --union "$UNION" \
      --primary-predictions "$PRIMARY_INFERENCE/stage4_bliss_predictions.jsonl" \
      --control-predictions "$CONTROL_INFERENCE/stage4_bliss_predictions.jsonl" \
      --out-dir "$ROADMAP_PLOT_DIR" \
      --formats png,pdf --dpi "${PLOT_DPI:-220}" \
      --max-scatter "${PLOT_MAX_SCATTER:-50000}" \
      --top-n "${PLOT_TOP_N:-30}" --n-boot "${PLOT_N_BOOT:-5000}" \
      --seed "${PLOT_SEED:-20260811}"
    ;;

  plot-pair-example)
    require_files "$EXTRACTED/pair_manifest.jsonl" \
      "$PRIMARY_INFERENCE/stage4_bliss_inference_summary.json" \
      "$STAGE4_CODE_DIR/candidate_preprocessing.py" \
      "$SCRIPT_DIR/plot_representative_pair.py"
    extra=()
    if [[ -n "${PRESENTATION_PAIR_ID:-}" ]]; then
      extra+=(--pair-id "$PRESENTATION_PAIR_ID")
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/plot_representative_pair.py" \
      --pair-manifest "$EXTRACTED/pair_manifest.jsonl" \
      --stage4-code-dir "$STAGE4_CODE_DIR" \
      --inference-summary "$PRIMARY_INFERENCE/stage4_bliss_inference_summary.json" \
      --out-dir "$PRESENTATION_PLOT_DIR" \
      --formats png,pdf --dpi "${PLOT_DPI:-300}" \
      ${extra[@]+"${extra[@]}"}
    ;;

  plot-presentation)
    "$0" plot-testa
    "$0" plot-roadmap
    "$0" plot-pair-example
    ;;

  freeze-results)
    require_files "$ANALYSIS_DIR/real_pair_analysis.json"
    "$PYTHON_BIN" "$SCRIPT_DIR/freeze_real_pair_artifacts.py" \
      --run-root "$REAL_OUT" --repository "$STAGE4_CODE_DIR" \
      --output "$REAL_OUT/LOFTS0050_REAL_PAIR_FROZEN.json" \
      --acknowledge-unlabeled-external-pilot
    ;;

  full-after-search)
    "$0" adapt
    "$0" derive-policy
    "$0" freeze-policy
    "$0" union
    "$0" extract
    "$0" infer
    "$0" controls
    "$0" infer-controls
    "$0" analyze
    "$0" freeze-results
    ;;

  help|*)
    cat <<'USAGE'
Usage: ./run_lofts0050_real_pair.sh COMMAND

Preparation and search:
  preflight         run all integration, real-pair, CPU, and plotting tests
  inputs            verify the real H5 and existing IRL export paths
  verify-naoise     pin the exact Naoise commit and script hash
  verify-backend    pin the BLISS backend commit and imported module path
  search-backend    prove the selected search backend without reading the H5
  observations      hash/inspect both H5 files and verify barycentric provenance
  search-swe        run the pinned blind search on Sweden
  search-smoke-swe  run one isolated Sweden coarse channel as a CPU smoke test
  search-irl        optionally reproduce Ireland locally with identical code
  search-both       reproduce both searches with identical code
  bank-audit        quantify native width-bank coverage of 10--100 Hz
  testa-incremental  reproduce frozen Test A and add paired Stage4-minus-filter CI
  plot-testa        create the complete frozen labeled Test-A figure suite

CPU defaults (safe when CuPy imports but CUDA is broken):
  export BLIND_SEARCH_BACKEND=cpu
  export STAGE4_DEVICE=cpu
  export CPU_THREADS=4

The pinned Naoise source is not modified. The launcher verifies its SHA-256,
forces its existing NumPy/BLISS CPU branch, and writes a per-station runtime
provenance JSON. Increase CPU_THREADS only if the node allocation permits it.

Label-free real-pair pipeline (in order):
  adapt             strictly validate raw and six-template station catalogs
  derive-policy     derive resolution gates and register controls before scores
  freeze-policy     freeze association and control policy before union/scoring
  union             build sparse A-union-B and classify unsearched rolloff gaps
  extract           extract exact 16x1024 high-resolution pairs
  infer             run the frozen model; threshold is exploratory only
  controls          extract clean-band, candidate-excluded shifted controls
  infer-controls    score the controls with the same frozen model
  analyze           descriptive rankings and paired score-control diagnostics
  plot-roadmap      plot every currently available real-pilot roadmap stage
  plot-pair-example create one academic raw/candidate-informed waterfall figure
  plot-presentation regenerate presentation figures only; no search or inference
  freeze-results    hash the completed unlabeled external-pipeline pilot
  full-after-search run adapt through freeze-results

Required before `observations`:
  Export IRL_H5 and SWE_H5 as absolute paths to the two simultaneous
  barycentric HDF5 products. See the repository .env.example.

  For a verified final run:
    export BARYCENTRIC_TOOL='verified upstream tool name'
    export BARYCENTRIC_VERSION='verified upstream tool version'

  While the exact upstream provenance is pending, a real-data engineering
  pilot may proceed without recorrecting the files:
    export BARYCENTRIC_TOOL='LOFAR/BL upstream pipeline'
    export BARYCENTRIC_VERSION='pending_upstream_confirmation'
    export ALLOW_UNVERIFIED_BARYCENTRIC_PROVENANCE=1

Required before `freeze-policy`:
  export POLICY_REVIEWER='Your name'
  export POLICY_NOTES='Frozen before associations or Stage-4 scores were inspected.'

The existing Synthetic-Test-B runner is still required for labeled blind
injection completeness/AUC. This runner deliberately cannot compute them.
USAGE
    ;;
esac
