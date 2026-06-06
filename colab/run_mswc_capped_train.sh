#!/usr/bin/env bash
set -Eeuo pipefail

# Colab runner for MSWC capped experiments.
#
# This is the safe Colab path for a 236GB /content disk:
# - extract at most MSWC_MAX_PER_WORD clips per word;
# - convert only that capped subset OPUS -> WAV;
# - never save WAV/audio clips to Drive;
# - continuously sync only checkpoints/results/reports/logs/configs.

PROJECT_DIR="${PROJECT_DIR:-/content/DoAnTotNghiep}"
DRIVE_RUN_ROOT="${DRIVE_RUN_ROOT:-/content/drive/MyDrive/DoAnTotNghiep_colab_runs}"
RUN_ID="${RUN_ID:-colab_mswc_cap${MSWC_MAX_PER_WORD:-50}_$(date '+%Y%m%d_%H%M%S')}"
DATA_DIR="${DATA_DIR:-data/mswc_en}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs_colab/$RUN_ID}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE_RUN_ROOT/$RUN_ID}"

MAX_SECONDS="${MAX_SECONDS:-84600}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"

MSWC_MAX_PER_WORD="${MSWC_MAX_PER_WORD:-50}"
MSWC_MIN_CLIPS="${MSWC_MIN_CLIPS:-1}"
MSWC_VAL_FRACTION="${MSWC_VAL_FRACTION:-0.02}"
MSWC_SPLIT_SEED="${MSWC_SPLIT_SEED:-42}"
MSWC_MIRROR="${MSWC_MIRROR:-cloudflare}"
CONVERT_WORKERS="${CONVERT_WORKERS:-8}"
CONVERT_BATCH_SIZE="${CONVERT_BATCH_SIZE:-8}"

RUN_EPOCHS="${RUN_EPOCHS:-10}"
RUN_EPISODES="${RUN_EPISODES:-200}"
RUN_WORKERS="${RUN_WORKERS:-8}"
RUN_N_CLASSES="${RUN_N_CLASSES:-30}"
RUN_N_SAMPLES="${RUN_N_SAMPLES:-10}"
RUN_K_SHOT="${RUN_K_SHOT:-10}"
GSC_EVERY="${GSC_EVERY:-2}"
GSC_DEV_RUNS="${GSC_DEV_RUNS:-5}"
FINAL_DEV_RUNS="${FINAL_DEV_RUNS:-30}"
FINAL_TEST_RUNS="${FINAL_TEST_RUNS:-100}"

RUN_DSCNN="${RUN_DSCNN:-1}"
RUN_EDGESPOT="${RUN_EDGESPOT:-1}"
RUN_EDGESPOT_HYBRID="${RUN_EDGESPOT_HYBRID:-0}"

START_TS="$(date +%s)"
DEADLINE_TS="$((START_TS + MAX_SECONDS))"
SYNC_PID=""
SUMMARY_FILE="$LOG_DIR/stages.tsv"
LOG_FILE="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR" "$DRIVE_OUT"
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
    echo "ERROR: Google Drive is not mounted. Run: from google.colab import drive; drive.mount('/content/drive')" >&2
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
  python -m pip install -q --upgrade pip "setuptools<82" wheel
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

prepare_data_capped() {
  log "Preparing GSC v2"
  if [ ! -d data/gsc_v2 ] || [ ! -f data/gsc_v2/testing_list.txt ] || [ ! -f data/gsc_v2/validation_list.txt ]; then
    python data/download_gsc.py --output-dir data/gsc_v2
  else
    echo "GSC v2 already present"
  fi

  log "Preparing capped MSWC English: max_per_word=$MSWC_MAX_PER_WORD"
  python data/download_mswc.py \
    --min-clips "$MSWC_MIN_CLIPS" \
    --val-fraction "$MSWC_VAL_FRACTION" \
    --split-seed "$MSWC_SPLIT_SEED" \
    --max-per-word "$MSWC_MAX_PER_WORD" \
    --mirror "$MSWC_MIRROR"

  rm -f "$DATA_DIR/en.tar.gz" 2>/dev/null || true

  log "Converting capped OPUS subset to WAV"
  python data/convert_opus.py \
    --clips-dir "$DATA_DIR/clips" \
    --workers "$CONVERT_WORKERS" \
    --batch-size "$CONVERT_BATCH_SIZE" \
    --delete-opus

  log "Building capped manifests from WAV subset"
  python data/build_mswc_file_splits.py \
    --data-dir "$DATA_DIR" \
    --max-per-word "$MSWC_MAX_PER_WORD" \
    --output-suffix "cap${MSWC_MAX_PER_WORD}" \
    --source clips \
    --overwrite

  python scripts/mswc_data_report.py --data-dir "$DATA_DIR" --top-n 20 || true
  sync_artifacts_once || true
  df -h /content || true
}

evaluate_checkpoint() {
  local ckpt="$1"
  local model_family="$2"
  local frontend="$3"
  local edge_tau="$4"
  local run_tag="$5"
  local split="$6"
  local n_runs="$7"
  local out_dir="results/$run_tag/gsc_${split}${n_runs}"

  if [ ! -f "$ckpt" ]; then
    log "Skip eval; missing checkpoint: $ckpt"
    return 0
  fi

  local args=(
    scripts/evaluate_edgespot_protocol.py
    --checkpoint "$ckpt"
    --config configs/default.yaml
    --model-family "$model_family"
    --feature-type "$frontend"
    --k-shot "$RUN_K_SHOT"
    --n-runs "$n_runs"
    --gsc-query-split "$split"
    --output-dir "$out_dir"
  )
  if [ -n "$edge_tau" ]; then
    args+=(--edge-tau "$edge_tau")
  fi

  log "Evaluating $run_tag on GSC-$split runs=$n_runs"
  python "${args[@]}"
  sync_artifacts_once || true
}

run_train_stage() {
  local name="$1"
  local model_family="$2"
  local frontend="$3"
  local loss="$4"
  local edge_tau="$5"
  local run_tag="${name}_${RUN_ID}"
  local started_at finished_at train_status

  if [ "$(seconds_left)" -lt 3600 ]; then
    log "Skipping $run_tag: not enough runtime left"
    return 0
  fi

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  train_status="not_started"
  log "Stage start: $run_tag"
  log "model=$model_family frontend=$frontend loss=$loss edge_tau=${edge_tau:-none}"

  local train_args=(
    scripts/train.py
    --config configs/default.yaml
    --data-dir "$DATA_DIR"
    --model-family "$model_family"
    --feature-type "$frontend"
    --loss "$loss"
    --epochs "$RUN_EPOCHS"
    --episodes "$RUN_EPISODES"
    --n-classes "$RUN_N_CLASSES"
    --n-samples "$RUN_N_SAMPLES"
    --max-per-word 0
    --train-files "train_files_cap${MSWC_MAX_PER_WORD}.json"
    --val-files "val_files_cap${MSWC_MAX_PER_WORD}.json"
    --num-workers "$RUN_WORKERS"
    --run-tag "$run_tag"
    --select-by-gsc-dev
    --gsc-dev-every "$GSC_EVERY"
    --gsc-dev-runs "$GSC_DEV_RUNS"
    --gsc-dev-k-shot "$RUN_K_SHOT"
    --save-every 1
    --save-latest-every-epoch
  )
  if [ -n "$edge_tau" ]; then
    train_args+=(--edge-tau "$edge_tau")
  fi

  local latest="checkpoints/$run_tag/latest.pt"
  if [ -f "$latest" ]; then
    log "Resuming from $latest"
    train_args+=(--resume "$latest")
  fi

  set +e
  python "${train_args[@]}"
  local rc=$?
  set -e
  sync_artifacts_once || true

  if [ "$rc" -eq 0 ]; then
    train_status="ok"
    evaluate_checkpoint "checkpoints/$run_tag/best.pt" "$model_family" "$frontend" "$edge_tau" "$run_tag" dev "$FINAL_DEV_RUNS" || true
    evaluate_checkpoint "checkpoints/$run_tag/best.pt" "$model_family" "$frontend" "$edge_tau" "$run_tag" test "$FINAL_TEST_RUNS" || true
  else
    train_status="failed_$rc"
    log "Train failed: $run_tag rc=$rc"
  fi

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo -e "${run_tag}\t${model_family}\t${frontend}\t${loss}\t${edge_tau:-}\tcap${MSWC_MAX_PER_WORD}\t${RUN_EPOCHS}\t${RUN_EPISODES}\t${train_status}\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
  sync_artifacts_once || true
}

main() {
  require_drive
  cd "$PROJECT_DIR"
  mkdir -p "$LOG_DIR" "$DRIVE_OUT"
  echo -e "run_tag\tmodel_family\tfrontend\tloss\tedge_tau\tdata_profile\tepochs\tepisodes\ttrain_status\tstarted_at\tfinished_at" > "$SUMMARY_FILE"

  log "Colab capped MSWC run id: $RUN_ID"
  log "Drive output: $DRIVE_OUT"
  log "Important: audio clips are local only and are NOT copied to Drive."
  log "MSWC_MAX_PER_WORD=$MSWC_MAX_PER_WORD"

  sync_artifacts_loop &
  SYNC_PID="$!"

  install_colab_deps
  print_env
  prepare_data_capped

  if [ "$RUN_DSCNN" = "1" ]; then
    run_train_stage "dscnn_pcen_ge2e_cap${MSWC_MAX_PER_WORD}_e${RUN_EPOCHS}_ep${RUN_EPISODES}" "dscnn" "mel_pcen" "ge2e" ""
  fi

  if [ "$RUN_EDGESPOT" = "1" ]; then
    run_train_stage "edgespot_full_t4_pcen_ge2e_cap${MSWC_MAX_PER_WORD}_e${RUN_EPOCHS}_ep${RUN_EPISODES}" "edgespot_full" "mel_pcen" "ge2e" "4"
  fi

  if [ "$RUN_EDGESPOT_HYBRID" = "1" ]; then
    run_train_stage "edgespot_full_t4_pcen_scaf_ge2e_cap${MSWC_MAX_PER_WORD}_e${RUN_EPOCHS}_ep${RUN_EPISODES}" "edgespot_full" "mel_pcen" "scaf_ge2e" "4"
  fi

  log "All scheduled stages finished or skipped. Summary: $SUMMARY_FILE"
  sync_artifacts_once || true
}

main "$@"
