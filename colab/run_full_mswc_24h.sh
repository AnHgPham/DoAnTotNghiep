#!/usr/bin/env bash
set -Eeuo pipefail

# Colab Pro+ 24h runner for KWS experiments.
#
# Design choices:
# - Do NOT save MSWC WAV clips to Google Drive. Full MSWC has millions of small
#   files; copying them to Drive is slow and wastes compute units.
# - Keep MSWC audio local under /content for the active Colab session.
# - Continuously sync only small/important artifacts to Drive:
#   checkpoints, results, reports, logs, and configs.
# - Run full-MSWC priority experiments first, then optional follow-up stages if
#   there is enough runtime left.

PROJECT_DIR="${PROJECT_DIR:-/content/DoAnTotNghiep}"
DRIVE_RUN_ROOT="${DRIVE_RUN_ROOT:-/content/drive/MyDrive/DoAnTotNghiep_colab_runs}"
RUN_ID="${RUN_ID:-colab_full_mswc_$(date '+%Y%m%d_%H%M%S')}"
DATA_DIR="${DATA_DIR:-data/mswc_en}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs_colab/$RUN_ID}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE_RUN_ROOT/$RUN_ID}"

# 23.5h leaves a little time for final artifact sync before Colab disconnects.
MAX_SECONDS="${MAX_SECONDS:-84600}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"
MIN_STAGE_SECONDS="${MIN_STAGE_SECONDS:-3600}"

MSWC_MIN_CLIPS="${MSWC_MIN_CLIPS:-1}"
MSWC_VAL_FRACTION="${MSWC_VAL_FRACTION:-0.02}"
MSWC_SPLIT_SEED="${MSWC_SPLIT_SEED:-42}"
MSWC_MIRROR="${MSWC_MIRROR:-cloudflare}"
CONVERT_WORKERS="${CONVERT_WORKERS:-$(nproc)}"
CONVERT_BATCH_SIZE="${CONVERT_BATCH_SIZE:-16}"

TRAIN_FILES_NAME="${TRAIN_FILES_NAME:-train_files_full.json}"
VAL_FILES_NAME="${VAL_FILES_NAME:-val_files_full.json}"
MANIFEST_SUFFIX="${MANIFEST_SUFFIX:-full}"
RUN_WORKERS="${RUN_WORKERS:-8}"
RUN_N_CLASSES="${RUN_N_CLASSES:-30}"
RUN_N_SAMPLES="${RUN_N_SAMPLES:-10}"
RUN_K_SHOT="${RUN_K_SHOT:-10}"
TARGET_FAR="${TARGET_FAR:-0.01}"
FINAL_DEV_RUNS="${FINAL_DEV_RUNS:-30}"
FINAL_TEST_RUNS="${FINAL_TEST_RUNS:-100}"

# Priority stage defaults. Increase epochs if using H100/A100 and you want the
# first stage to keep running longer.
FULL_EPOCHS="${FULL_EPOCHS:-10}"
FULL_EPISODES="${FULL_EPISODES:-200}"
FULL_GSC_EVERY="${FULL_GSC_EVERY:-2}"
FULL_GSC_RUNS="${FULL_GSC_RUNS:-5}"

# Follow-up stages after the two main GE2E candidates.
RUN_EXTRA_HYBRID="${RUN_EXTRA_HYBRID:-1}"
RUN_EXTRA_DSCNN_HYBRID="${RUN_EXTRA_DSCNN_HYBRID:-0}"
RUN_TOP500_AFTER_FULL="${RUN_TOP500_AFTER_FULL:-1}"
TOP500_EPOCHS="${TOP500_EPOCHS:-10}"
TOP500_EPISODES="${TOP500_EPISODES:-200}"

RESUME_FROM_DRIVE_RUN_ID="${RESUME_FROM_DRIVE_RUN_ID:-}"
RESUME_TRAIN="${RESUME_TRAIN:-1}"

START_TS="$(date +%s)"
DEADLINE_TS="$((START_TS + MAX_SECONDS))"
SYNC_PID=""

mkdir -p "$LOG_DIR" "$DRIVE_OUT"
LOG_FILE="$LOG_DIR/run.log"
SUMMARY_FILE="$LOG_DIR/stages.tsv"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

seconds_left() {
  local now
  now="$(date +%s)"
  echo "$((DEADLINE_TS - now))"
}

require_drive() {
  if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive is not mounted. Run this in Colab first:" >&2
    echo "from google.colab import drive; drive.mount('/content/drive')" >&2
    exit 2
  fi
}

sync_dir() {
  local src="$1"
  local dst="$2"
  if [ ! -e "$PROJECT_DIR/$src" ]; then
    return 0
  fi
  mkdir -p "$DRIVE_OUT/$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --update \
      --exclude '__pycache__' \
      --exclude '.ipynb_checkpoints' \
      "$PROJECT_DIR/$src/" "$DRIVE_OUT/$dst/"
  else
    cp -ru "$PROJECT_DIR/$src/." "$DRIVE_OUT/$dst/"
  fi
}

sync_artifacts_once() {
  mkdir -p "$DRIVE_OUT"
  sync_dir "checkpoints" "checkpoints" || true
  sync_dir "results" "results" || true
  sync_dir "reports" "reports" || true
  sync_dir "logs_colab" "logs_colab" || true
  sync_dir "configs" "configs" || true
  sync_dir "colab" "colab" || true
  cp -f "$LOG_FILE" "$DRIVE_OUT/run.log" 2>/dev/null || true
}

sync_artifacts_loop() {
  while true; do
    sync_artifacts_once || true
    sleep "$SYNC_SECONDS"
  done
}

restore_artifacts_from_drive() {
  if [ -z "$RESUME_FROM_DRIVE_RUN_ID" ]; then
    return 0
  fi
  local src="$DRIVE_RUN_ROOT/$RESUME_FROM_DRIVE_RUN_ID"
  if [ ! -d "$src" ]; then
    log "Requested resume run not found on Drive: $src"
    return 0
  fi
  log "Restoring small artifacts from Drive run: $RESUME_FROM_DRIVE_RUN_ID"
  mkdir -p "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/results" "$PROJECT_DIR/logs_colab"
  [ -d "$src/checkpoints" ] && cp -ru "$src/checkpoints/." "$PROJECT_DIR/checkpoints/"
  [ -d "$src/results" ] && cp -ru "$src/results/." "$PROJECT_DIR/results/"
  [ -d "$src/logs_colab" ] && cp -ru "$src/logs_colab/." "$PROJECT_DIR/logs_colab/"
}

cleanup() {
  set +e
  log "Final artifact sync"
  if [ -n "${SYNC_PID:-}" ]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_artifacts_once || true
}
trap cleanup EXIT

install_colab_deps() {
  log "Installing Colab dependencies without reinstalling torch"
  apt-get update -qq
  apt-get install -y -qq ffmpeg rsync
  python -m pip install -q --upgrade pip setuptools wheel
  python -m pip install -q \
    "numpy<2.0" \
    pyyaml scipy soundfile scikit-learn matplotlib tensorboard tqdm requests \
    fastapi "uvicorn[standard]>=0.27" python-multipart
}

print_env() {
  log "Environment"
  pwd
  df -h /content || true
  nvidia-smi || true
  python - <<'PY'
import os, sys, torch
print("python:", sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
print("cwd:", os.getcwd())
PY
}

check_disk_for_full_mswc() {
  local free_gb
  free_gb="$(python - <<'PY'
import shutil
print(int(shutil.disk_usage("/content").free / 1024**3))
PY
)"
  log "Free /content disk before Full MSWC: ${free_gb} GB"
  if [ "$free_gb" -lt 120 ]; then
    echo "ERROR: Full MSWC local run needs large /content disk." >&2
    echo "Current free space: ${free_gb} GB; recommended: 150 GB+." >&2
    echo "Use Colab high-RAM / high-disk runtime or reduce the plan." >&2
    exit 3
  fi
  if [ "$free_gb" -lt 150 ]; then
    log "Warning: disk is below 150GB. Full MSWC may be tight but will continue."
  fi
}

prepare_data_full_mswc() {
  log "Preparing GSC v2"
  if [ ! -d data/gsc_v2 ] || [ ! -f data/gsc_v2/testing_list.txt ] || [ ! -f data/gsc_v2/validation_list.txt ]; then
    python data/download_gsc.py --output-dir data/gsc_v2
  else
    echo "GSC v2 already present"
  fi

  local ready_marker="$DATA_DIR/.colab_full_mswc_data_ready"
  if [ -f "$ready_marker" ] && [ -d "$DATA_DIR/clips" ]; then
    log "Full MSWC local data marker exists; skipping download/extract/convert"
  else
    check_disk_for_full_mswc
    log "Preparing FULL MSWC English locally under $DATA_DIR"
    log "No WAV cache will be saved to Drive."
    mkdir -p "$DATA_DIR"
    {
      echo "mode=full_mswc_english_colab_no_drive_wav_cache"
      echo "min_clips=$MSWC_MIN_CLIPS"
      echo "max_per_word=0"
      echo "created_at=$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$DATA_DIR/.full_mswc_english_mode"

    python data/download_mswc.py \
      --min-clips "$MSWC_MIN_CLIPS" \
      --val-fraction "$MSWC_VAL_FRACTION" \
      --split-seed "$MSWC_SPLIT_SEED" \
      --max-per-word 0 \
      --mirror "$MSWC_MIRROR"

    python data/convert_opus.py \
      --clips-dir "$DATA_DIR/clips" \
      --workers "$CONVERT_WORKERS" \
      --batch-size "$CONVERT_BATCH_SIZE" \
      --delete-opus

    date '+%Y-%m-%d %H:%M:%S' > "$ready_marker"
  fi

  log "Building full all-clips manifests"
  python data/build_mswc_file_splits.py \
    --data-dir "$DATA_DIR" \
    --max-per-word 0 \
    --output-suffix "$MANIFEST_SUFFIX" \
    --source clips \
    --overwrite

  python scripts/mswc_data_report.py --data-dir "$DATA_DIR" --top-n 20 || true
  sync_artifacts_once || true
}

extract_best_metric() {
  local run_tag="$1"
  python - "$run_tag" "$PROJECT_DIR/logs_colab" <<'PY'
import re
import sys
from pathlib import Path

run_tag = sys.argv[1]
root = Path(sys.argv[2])
best = None
for path in root.rglob("*.log"):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if run_tag not in text:
        continue
    for match in re.finditer(r"New best GSC-dev ACC@1%FAR: ([0-9.]+)", text):
        value = float(match.group(1))
        best = value if best is None else max(best, value)
    for match in re.finditer(r"Done! Best metric: ([0-9.]+)", text):
        value = float(match.group(1))
        best = value if best is None else max(best, value)
if best is not None:
    print(best)
PY
}

evaluate_checkpoint() {
  local run_tag="$1"
  local model_family="$2"
  local frontend="$3"
  local edge_tau="$4"
  local split="$5"
  local n_runs="$6"
  local label="$7"
  local checkpoint="checkpoints/$run_tag/best.pt"
  local output_dir="results/$RUN_ID/$run_tag/$label"

  if [ ! -f "$checkpoint" ]; then
    log "Missing checkpoint, skip eval: $checkpoint"
    return 2
  fi

  local args=(
    scripts/evaluate.py
    --config configs/default.yaml
    --checkpoint "$checkpoint"
    --model-family "$model_family"
    --feature-type "$frontend"
    --protocol gsc_edgespot_exact
    --gsc-query-split "$split"
    --target-far "$TARGET_FAR"
    --k-shot "$RUN_K_SHOT"
    --n-runs "$n_runs"
    --output-dir "$output_dir"
    --plot-det
  )
  if [ -n "$edge_tau" ]; then
    args+=(--edge-tau "$edge_tau")
  fi

  log "Evaluating $run_tag on GSC-$split, runs=$n_runs"
  python "${args[@]}"
  sync_artifacts_once || true
}

run_train_eval_stage() {
  local id="$1"
  local data_dir="$2"
  local train_files="$3"
  local val_files="$4"
  local model_family="$5"
  local frontend="$6"
  local loss="$7"
  local edge_tau="$8"
  local epochs="$9"
  local episodes="${10}"
  local gsc_every="${11}"
  local gsc_runs="${12}"
  local min_seconds="${13:-$MIN_STAGE_SECONDS}"
  local left
  left="$(seconds_left)"
  if [ "$left" -lt "$min_seconds" ]; then
    log "Skipping $id: only ${left}s left, need at least ${min_seconds}s"
    return 0
  fi

  local run_tag="${id}_${RUN_ID}"
  local started_at finished_at train_status dev_status test_status
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  train_status="not_started"
  dev_status="not_started"
  test_status="not_started"

  log "Stage start: $run_tag"
  log "model=$model_family frontend=$frontend loss=$loss edge_tau=${edge_tau:-none}"
  log "data=$data_dir train_files=$train_files val_files=$val_files epochs=$epochs episodes=$episodes"

  local train_args=(
    scripts/train.py
    --config configs/default.yaml
    --data-dir "$data_dir"
    --model-family "$model_family"
    --feature-type "$frontend"
    --loss "$loss"
    --epochs "$epochs"
    --episodes "$episodes"
    --n-classes "$RUN_N_CLASSES"
    --n-samples "$RUN_N_SAMPLES"
    --max-per-word 0
    --train-files "$train_files"
    --val-files "$val_files"
    --num-workers "$RUN_WORKERS"
    --run-tag "$run_tag"
    --select-by-gsc-dev
    --gsc-dev-every "$gsc_every"
    --gsc-dev-runs "$gsc_runs"
    --gsc-dev-k-shot "$RUN_K_SHOT"
    --save-every 1
    --save-latest-every-epoch
  )
  if [ -n "$edge_tau" ]; then
    train_args+=(--edge-tau "$edge_tau")
  fi

  local latest="checkpoints/$run_tag/latest.pt"
  if [ "$RESUME_TRAIN" = "1" ] && [ -f "$latest" ]; then
    local initial_best
    initial_best="$(extract_best_metric "$run_tag" || true)"
    log "Resuming $run_tag from $latest; parsed initial_best=${initial_best:-none}"
    train_args+=(--resume "$latest")
    if [ -n "$initial_best" ]; then
      train_args+=(--initial-best-metric "$initial_best")
    else
      # Keep existing best.pt from being overwritten by an unknown worse metric.
      train_args+=(--initial-best-metric 1.0)
    fi
  fi

  set +e
  python "${train_args[@]}"
  local rc=$?
  set -e
  sync_artifacts_once || true

  if [ "$rc" -eq 0 ]; then
    train_status="ok"
    set +e
    evaluate_checkpoint "$run_tag" "$model_family" "$frontend" "$edge_tau" "dev" "$FINAL_DEV_RUNS" "dev${FINAL_DEV_RUNS}"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then dev_status="ok"; else dev_status="failed_rc_${rc}"; fi

    set +e
    evaluate_checkpoint "$run_tag" "$model_family" "$frontend" "$edge_tau" "test" "$FINAL_TEST_RUNS" "test${FINAL_TEST_RUNS}"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then test_status="ok"; else test_status="failed_rc_${rc}"; fi
  else
    train_status="failed_rc_${rc}"
    log "Train failed for $run_tag rc=$rc"
  fi

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo -e "${run_tag}\t${model_family}\t${frontend}\t${loss}\t${edge_tau:-}\t${data_dir}\t${train_files}\t${epochs}\t${episodes}\t${train_status}\t${dev_status}\t${test_status}\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
  sync_artifacts_once || true
}

prepare_top500_profile() {
  log "Preparing Top500 profile from local full MSWC"
  python data/build_mswc_top500_profile.py \
    --source-data-dir "$DATA_DIR" \
    --output-data-dir data/mswc_top500_full \
    --max-per-word 0
}

main() {
  require_drive
  cd "$PROJECT_DIR"
  mkdir -p "$LOG_DIR" "$DRIVE_OUT"
  echo -e "run_tag\tmodel_family\tfrontend\tloss\tedge_tau\tdata_dir\ttrain_files\tepochs\tepisodes\ttrain_status\tdev_status\ttest_status\tstarted_at\tfinished_at" > "$SUMMARY_FILE"

  log "Colab run id: $RUN_ID"
  log "Drive output: $DRIVE_OUT"
  log "Important: WAV clips are local only and are NOT copied to Drive."

  restore_artifacts_from_drive
  sync_artifacts_loop &
  SYNC_PID="$!"

  install_colab_deps
  print_env
  prepare_data_full_mswc

  run_train_eval_stage \
    "dscnn_pcen_ge2e_full_allclips_e${FULL_EPOCHS}_ep${FULL_EPISODES}" \
    "$DATA_DIR" "$TRAIN_FILES_NAME" "$VAL_FILES_NAME" \
    "dscnn" "mel_pcen" "ge2e" "" \
    "$FULL_EPOCHS" "$FULL_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 7200

  run_train_eval_stage \
    "edgespot_full_t4_pcen_ge2e_full_allclips_e${FULL_EPOCHS}_ep${FULL_EPISODES}" \
    "$DATA_DIR" "$TRAIN_FILES_NAME" "$VAL_FILES_NAME" \
    "edgespot_full" "mel_pcen" "ge2e" "4" \
    "$FULL_EPOCHS" "$FULL_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 5400

  if [ "$RUN_EXTRA_HYBRID" = "1" ]; then
    run_train_eval_stage \
      "edgespot_full_t4_pcen_scaf_ge2e_full_allclips_e${FULL_EPOCHS}_ep${FULL_EPISODES}" \
      "$DATA_DIR" "$TRAIN_FILES_NAME" "$VAL_FILES_NAME" \
      "edgespot_full" "mel_pcen" "scaf_ge2e" "4" \
      "$FULL_EPOCHS" "$FULL_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 5400
  fi

  if [ "$RUN_EXTRA_DSCNN_HYBRID" = "1" ]; then
    run_train_eval_stage \
      "dscnn_pcen_scaf_ge2e_full_allclips_e${FULL_EPOCHS}_ep${FULL_EPISODES}" \
      "$DATA_DIR" "$TRAIN_FILES_NAME" "$VAL_FILES_NAME" \
      "dscnn" "mel_pcen" "scaf_ge2e" "" \
      "$FULL_EPOCHS" "$FULL_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 5400
  fi

  if [ "$RUN_TOP500_AFTER_FULL" = "1" ]; then
    if [ "$(seconds_left)" -ge 5400 ]; then
      prepare_top500_profile
      run_train_eval_stage \
        "dscnn_pcen_ge2e_top500_full_e${TOP500_EPOCHS}_ep${TOP500_EPISODES}" \
        "data/mswc_top500_full" "train_files.json" "val_files.json" \
        "dscnn" "mel_pcen" "ge2e" "" \
        "$TOP500_EPOCHS" "$TOP500_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 3600
      run_train_eval_stage \
        "edgespot_full_t4_pcen_scaf_ge2e_top500_full_e${TOP500_EPOCHS}_ep${TOP500_EPISODES}" \
        "data/mswc_top500_full" "train_files.json" "val_files.json" \
        "edgespot_full" "mel_pcen" "scaf_ge2e" "4" \
        "$TOP500_EPOCHS" "$TOP500_EPISODES" "$FULL_GSC_EVERY" "$FULL_GSC_RUNS" 3600
    else
      log "Skipping Top500 follow-up: not enough time left"
    fi
  fi

  log "All scheduled stages finished or skipped. Summary: $SUMMARY_FILE"
  sync_artifacts_once || true
}

main "$@"
