#!/usr/bin/env bash
set -Eeuo pipefail

# Development runner after the fixed 16-pipeline cap620 result.
# Goals:
# 1) Accuracy: DSCNN-L + PCEN + GE2E with larger episode budget, hard-pair episodes,
#    and composite checkpoint selection: ACC@1%FAR + AUC + F1.
# 2) Compact: EdgeSpotFull T4 + PCEN + Triplet/GE2E, avoiding SCAF.
# 3) EdgeSpot > EdgeSpot-4 direction: optional KD via Wav2Vec2 teacher embeddings.
# 4) SCAF recovery: small/subset ablation only, with reduced weight/margin/scale and warmup.
#
# This runner assumes the same Colab layout as run_mswc_cap620_16_pipeline_e40_fixed.sh.

PROJECT_DIR="${PROJECT_DIR:-/content/DoAnTotNghiep}"
DRIVE_RUN_ROOT="${DRIVE_RUN_ROOT:-/content/drive/MyDrive/DoAnTotNghiep_colab_runs}"

readonly DATA_DIR="data/mswc_en"
readonly MSWC_MAX_PER_WORD=620
readonly RUN_WORKERS="${RUN_WORKERS:-8}"
readonly RUN_K_SHOT=10
readonly FINAL_DEV_RUNS="${FINAL_DEV_RUNS:-30}"
readonly FINAL_TEST_RUNS="${FINAL_TEST_RUNS:-100}"

readonly MSWC_MIN_CLIPS=1
readonly MSWC_VAL_FRACTION=0.02
readonly MSWC_SPLIT_SEED=42
readonly MSWC_MIRROR="${MSWC_MIRROR:-cloudflare}"
readonly CONVERT_WORKERS="${CONVERT_WORKERS:-12}"
readonly CONVERT_BATCH_SIZE="${CONVERT_BATCH_SIZE:-16}"
readonly FLAC_COMPRESSION_LEVEL="${FLAC_COMPRESSION_LEVEL:-3}"
readonly MAX_DISK_USE_PERCENT="${MAX_DISK_USE_PERCENT:-90}"

RUN_ID="${RUN_ID:-colab_mswc_cap620_development_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs_colab/$RUN_ID}"
DRIVE_OUT="${DRIVE_OUT:-$DRIVE_RUN_ROOT/$RUN_ID}"
MAX_SECONDS="${MAX_SECONDS:-172800}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"
ALLOW_EXISTING_DATA="${ALLOW_EXISTING_DATA:-1}"

# Toggle groups.
RUN_ACCURACY="${RUN_ACCURACY:-1}"
RUN_COMPACT="${RUN_COMPACT:-1}"
RUN_KD="${RUN_KD:-0}"
RUN_SCAF_ABLATION="${RUN_SCAF_ABLATION:-0}"

# Main development budgets.
ACC_EPOCHS="${ACC_EPOCHS:-60}"
ACC_EPISODES="${ACC_EPISODES:-300}"
ACC_N_CLASSES="${ACC_N_CLASSES:-40}"
ACC_N_SAMPLES="${ACC_N_SAMPLES:-10}"

COMPACT_EPOCHS="${COMPACT_EPOCHS:-60}"
COMPACT_EPISODES="${COMPACT_EPISODES:-300}"
COMPACT_N_CLASSES="${COMPACT_N_CLASSES:-40}"
COMPACT_N_SAMPLES="${COMPACT_N_SAMPLES:-10}"

GSC_EVERY="${GSC_EVERY:-5}"
GSC_DEV_RUNS="${GSC_DEV_RUNS:-5}"
GSC_SELECT_METRIC="${GSC_SELECT_METRIC:-composite}"

HARD_PAIRS_PATH="${HARD_PAIRS_PATH:-results/hard_pairs.json}"
HARD_PAIR_PROB="${HARD_PAIR_PROB:-0.35}"

# KD is expensive. Keep it explicitly opt-in and subset-capped by default.
TEACHER_HEAD="${TEACHER_HEAD:-outputs/teacher_head/cap620_subset_teacher_head.pt}"
TEACHER_EMB_DIR="${TEACHER_EMB_DIR:-outputs/teacher_embeddings/cap620_kd_subset}"
KD_MAX_WORDS="${KD_MAX_WORDS:-1000}"
KD_MAX_PER_WORD="${KD_MAX_PER_WORD:-80}"
KD_HEAD_EPOCHS="${KD_HEAD_EPOCHS:-20}"
KD_EPOCHS="${KD_EPOCHS:-40}"
KD_EPISODES="${KD_EPISODES:-250}"
KD_N_CLASSES="${KD_N_CLASSES:-40}"
KD_N_SAMPLES="${KD_N_SAMPLES:-10}"
KD_WEIGHT="${KD_WEIGHT:-0.5}"

# SCAF recovery is deliberately subset-only by default.
SCAF_SUBSET_WORDS="${SCAF_SUBSET_WORDS:-1000}"
SCAF_MAX_PER_WORD="${SCAF_MAX_PER_WORD:-80}"
SCAF_WARMUP_EPOCHS="${SCAF_WARMUP_EPOCHS:-10}"
SCAF_FINETUNE_EPOCHS="${SCAF_FINETUNE_EPOCHS:-30}"
SCAF_EPISODES="${SCAF_EPISODES:-180}"
SCAF_N_CLASSES="${SCAF_N_CLASSES:-30}"
SCAF_N_SAMPLES="${SCAF_N_SAMPLES:-10}"
SCAF_WEIGHT="${SCAF_WEIGHT:-0.05}"
SCAF_SCALE="${SCAF_SCALE:-16.0}"
SCAF_MARGIN="${SCAF_MARGIN:-0.2}"

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

disk_use_percent() {
  df -P /content | awk 'NR==2 {gsub("%", "", $5); print $5}'
}

abort_if_disk_high() {
  local stage="$1"
  local used
  used="$(disk_use_percent)"
  df -h /content || true
  if [ -n "$used" ] && [ "$used" -ge "$MAX_DISK_USE_PERCENT" ]; then
    echo "ERROR: disk use is ${used}% after ${stage}; threshold=${MAX_DISK_USE_PERCENT}%." >&2
    exit 4
  fi
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
  sync_dir "outputs" "outputs" || true
  sync_dir "data/mswc_en/splits" "data/mswc_en/splits" || true
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
  if [ "$RUN_KD" = "1" ]; then
    python -m pip install -q "transformers>=4.40,<5" safetensors
  fi
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
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
print("cwd:", os.getcwd())
PY
}

prepare_gsc() {
  log "Preparing GSC v2"
  if [ ! -d data/gsc_v2 ] || [ ! -f data/gsc_v2/testing_list.txt ] || [ ! -f data/gsc_v2/validation_list.txt ]; then
    python data/download_gsc.py --output-dir data/gsc_v2
  else
    echo "GSC v2 already present"
  fi
}

prepare_cap620_flac() {
  local train_manifest="$DATA_DIR/splits/train_files_cap620_flac.json"
  local val_manifest="$DATA_DIR/splits/val_files_cap620_flac.json"

  if [ -f "$train_manifest" ] && [ -f "$val_manifest" ]; then
    log "Existing cap620 FLAC manifests found; reusing local data"
    python - <<'PY'
import json
from pathlib import Path
for p in [Path("data/mswc_en/splits/train_files_cap620_flac.json"), Path("data/mswc_en/splits/val_files_cap620_flac.json")]:
    print(f"{p.name}: {len(json.loads(p.read_text(encoding='utf-8'))):,} files")
PY
    return 0
  fi

  if [ "$ALLOW_EXISTING_DATA" != "1" ] && [ -d "$DATA_DIR/clips" ]; then
    echo "ERROR: existing $DATA_DIR/clips found. Use fresh runtime or set ALLOW_EXISTING_DATA=1." >&2
    exit 3
  fi

  log "Preparing MSWC cap620 OPUS subset"
  python data/download_mswc.py \
    --min-clips "$MSWC_MIN_CLIPS" \
    --val-fraction "$MSWC_VAL_FRACTION" \
    --split-seed "$MSWC_SPLIT_SEED" \
    --max-per-word "$MSWC_MAX_PER_WORD" \
    --mirror "$MSWC_MIRROR"

  rm -f "$DATA_DIR/en.tar.gz" 2>/dev/null || true
  abort_if_disk_high "MSWC extraction"

  log "Converting OPUS -> FLAC and deleting OPUS"
  python data/convert_opus_to_flac.py \
    --clips-dir "$DATA_DIR/clips" \
    --workers "$CONVERT_WORKERS" \
    --batch-size "$CONVERT_BATCH_SIZE" \
    --compression-level "$FLAC_COMPRESSION_LEVEL" \
    --delete-opus

  abort_if_disk_high "OPUS->FLAC conversion"

  log "Building cap620 FLAC manifests"
  python data/build_mswc_file_splits.py \
    --data-dir "$DATA_DIR" \
    --max-per-word "$MSWC_MAX_PER_WORD" \
    --output-suffix "cap620_flac" \
    --source clips \
    --overwrite

  sync_artifacts_once || true
}

prepare_subset_words() {
  local count="$1"
  local out_name="$2"
  local out_path="$DATA_DIR/splits/$out_name"
  if [ -f "$out_path" ]; then
    echo "$out_name already exists"
    return 0
  fi
  python - "$count" "$out_path" <<'PY'
import json, sys
from pathlib import Path
count = int(sys.argv[1])
out = Path(sys.argv[2])
words = json.loads(Path("data/mswc_en/splits/train_words.json").read_text(encoding="utf-8"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(words[:count], ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out}: {min(count, len(words))} words")
PY
}

is_completed_ok() {
  local run_tag="$1"
  [ -f "$SUMMARY_FILE" ] || return 1
  awk -F '\t' -v tag="$run_tag" \
    '$1 == tag && $9 == "ok" && $10 == "ok" && $11 == "ok" && $12 == "ok" { found=1 } END { exit(found ? 0 : 1) }' \
    "$SUMMARY_FILE"
}

evaluate_checkpoint() {
  local ckpt="$1"
  local model_family="$2"
  local frontend="$3"
  local edge_tau="$4"
  local run_tag="$5"
  local split="$6"
  local n_runs="$7"
  local target_far="$8"
  local label="$9"
  local out_dir="results/$RUN_ID/$run_tag/$label"

  if [ ! -f "$ckpt" ]; then
    log "Skip eval; missing checkpoint: $ckpt"
    return 2
  fi

  local args=(
    scripts/evaluate.py
    --config configs/default.yaml
    --checkpoint "$ckpt"
    --model-family "$model_family"
    --feature-type "$frontend"
    --protocol gsc_edgespot_exact
    --gsc-query-split "$split"
    --target-far "$target_far"
    --k-shot "$RUN_K_SHOT"
    --n-runs "$n_runs"
    --output-dir "$out_dir"
    --plot-det
  )
  if [ -n "$edge_tau" ]; then
    args+=(--edge-tau "$edge_tau")
  fi

  log "Evaluating $label: run_tag=$run_tag split=$split n_runs=$n_runs target_far=$target_far"
  python "${args[@]}"
  sync_artifacts_once || true
}

run_train_stage() {
  local id="$1"
  local model_family="$2"
  local frontend="$3"
  local loss="$4"
  local edge_tau="$5"
  local epochs="$6"
  local episodes="$7"
  local n_classes="$8"
  local n_samples="$9"
  shift 9
  local extra_args=("$@")

  local run_tag="${id}_${RUN_ID}"
  local latest="checkpoints/$run_tag/latest.pt"
  local started_at finished_at train_status dev_status test1_status test5_status

  if is_completed_ok "$run_tag"; then
    log "Skipping completed stage: $run_tag"
    return 0
  fi
  if [ "$(seconds_left)" -lt 3600 ]; then
    log "Skipping $run_tag: not enough runtime left"
    return 0
  fi

  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  train_status="not_started"
  dev_status="not_started"
  test1_status="not_started"
  test5_status="not_started"

  log "Stage start: $run_tag"
  log "model=$model_family frontend=$frontend loss=$loss edge_tau=${edge_tau:-none}"
  log "epochs=$epochs episodes=$episodes n_classes=$n_classes n_samples=$n_samples extra=${extra_args[*]:-none}"
  local has_manual_resume=0
  local item
  for item in "${extra_args[@]}"; do
    if [ "$item" = "--resume" ]; then
      has_manual_resume=1
      break
    fi
  done

  local train_args=(
    scripts/train.py
    --config configs/default.yaml
    --data-dir "$DATA_DIR"
    --model-family "$model_family"
    --feature-type "$frontend"
    --loss "$loss"
    --epochs "$epochs"
    --episodes "$episodes"
    --n-classes "$n_classes"
    --n-samples "$n_samples"
    --max-per-word 0
    --train-files "train_files_cap620_flac.json"
    --val-files "val_files_cap620_flac.json"
    --num-workers "$RUN_WORKERS"
    --run-tag "$run_tag"
    --select-by-gsc-dev
    --gsc-dev-select-metric "$GSC_SELECT_METRIC"
    --gsc-dev-every "$GSC_EVERY"
    --gsc-dev-runs "$GSC_DEV_RUNS"
    --gsc-dev-k-shot "$RUN_K_SHOT"
    --save-every 1
    --save-latest-every-epoch
  )
  if [ -n "$edge_tau" ]; then
    train_args+=(--edge-tau "$edge_tau")
  fi
  if [ "$has_manual_resume" -eq 0 ] && [ -f "$latest" ]; then
    log "Resuming from $latest"
    train_args+=(--resume "$latest")
  fi
  if [ "${#extra_args[@]}" -gt 0 ]; then
    train_args+=("${extra_args[@]}")
  fi

  set +e
  python "${train_args[@]}"
  local rc=$?
  set -e
  sync_artifacts_once || true

  if [ "$rc" -eq 0 ]; then
    train_status="ok"

    set +e
    evaluate_checkpoint "checkpoints/$run_tag/best.pt" "$model_family" "$frontend" "$edge_tau" "$run_tag" dev "$FINAL_DEV_RUNS" "0.01" "dev${FINAL_DEV_RUNS}_far1"
    rc=$?
    set -e
    dev_status=$([ "$rc" -eq 0 ] && echo "ok" || echo "failed_rc_${rc}")

    set +e
    evaluate_checkpoint "checkpoints/$run_tag/best.pt" "$model_family" "$frontend" "$edge_tau" "$run_tag" test "$FINAL_TEST_RUNS" "0.01" "test${FINAL_TEST_RUNS}_far1"
    rc=$?
    set -e
    test1_status=$([ "$rc" -eq 0 ] && echo "ok" || echo "failed_rc_${rc}")

    set +e
    evaluate_checkpoint "checkpoints/$run_tag/best.pt" "$model_family" "$frontend" "$edge_tau" "$run_tag" test "$FINAL_TEST_RUNS" "0.05" "test${FINAL_TEST_RUNS}_far5"
    rc=$?
    set -e
    test5_status=$([ "$rc" -eq 0 ] && echo "ok" || echo "failed_rc_${rc}")
  else
    train_status="failed_rc_${rc}"
    log "Train failed: $run_tag rc=$rc"
  fi

  finished_at="$(date '+%Y-%m-%d %H:%M:%S')"
  echo -e "${run_tag}\t${model_family}\t${frontend}\t${loss}\t${edge_tau:-}\tcap620_development\t${epochs}\t${episodes}\t${train_status}\t${dev_status}\t${test1_status}\t${test5_status}\t${started_at}\t${finished_at}" >> "$SUMMARY_FILE"
  sync_artifacts_once || true
}

prepare_teacher_for_kd() {
  if [ "$RUN_KD" != "1" ]; then
    return 0
  fi
  prepare_subset_words "$KD_MAX_WORDS" "train_words_kd_${KD_MAX_WORDS}.json"
  if [ ! -f "$TEACHER_HEAD" ]; then
    log "Training Wav2Vec2 teacher projection head for KD subset"
    python scripts/train_teacher_head.py \
      --data-dir "$DATA_DIR" \
      --train-files "train_files_cap620_flac.json" \
      --train-words-file "train_words_kd_${KD_MAX_WORDS}.json" \
      --max-words "$KD_MAX_WORDS" \
      --max-per-word "$KD_MAX_PER_WORD" \
      --epochs "$KD_HEAD_EPOCHS" \
      --extract-batch-size 16 \
      --batch-size 256 \
      --output "$TEACHER_HEAD"
    sync_artifacts_once || true
  else
    log "Existing teacher head found: $TEACHER_HEAD"
  fi

  if ! find "$TEACHER_EMB_DIR" -name 'teacher_*.pt' -print -quit 2>/dev/null | grep -q .; then
    log "Precomputing teacher embeddings for KD subset"
    python scripts/precompute_teacher_embeddings.py \
      --data-dir "$DATA_DIR" \
      --train-files "train_files_cap620_flac.json" \
      --train-words-file "train_words_kd_${KD_MAX_WORDS}.json" \
      --max-per-word "$KD_MAX_PER_WORD" \
      --head-checkpoint "$TEACHER_HEAD" \
      --output-dir "$TEACHER_EMB_DIR" \
      --batch-size 16 \
      --shard-size 4096
    sync_artifacts_once || true
  else
    log "Existing teacher embedding shards found: $TEACHER_EMB_DIR"
  fi
}

main() {
  require_drive
  cd "$PROJECT_DIR"
  mkdir -p "$LOG_DIR" "$DRIVE_OUT"
  if [ ! -s "$SUMMARY_FILE" ]; then
    echo -e "run_tag\tmodel_family\tfrontend\tloss\tedge_tau\tdata_profile\tepochs\tepisodes\ttrain_status\tdev30_far1_status\ttest100_far1_status\ttest100_far5_status\tstarted_at\tfinished_at" > "$SUMMARY_FILE"
  fi

  log "Colab cap620 development run id: $RUN_ID"
  log "Drive output: $DRIVE_OUT"
  log "Toggles: RUN_ACCURACY=$RUN_ACCURACY RUN_COMPACT=$RUN_COMPACT RUN_KD=$RUN_KD RUN_SCAF_ABLATION=$RUN_SCAF_ABLATION"
  log "GSC checkpoint selection metric: $GSC_SELECT_METRIC"

  sync_artifacts_loop &
  SYNC_PID="$!"

  install_colab_deps
  print_env
  prepare_gsc
  prepare_cap620_flac

  if [ "$RUN_ACCURACY" = "1" ]; then
    run_train_stage \
      "dscnn_pcen_ge2e_accdev_ep${ACC_EPISODES}_composite" \
      "dscnn" "mel_pcen" "ge2e" "" \
      "$ACC_EPOCHS" "$ACC_EPISODES" "$ACC_N_CLASSES" "$ACC_N_SAMPLES" \
      --hard-pairs-path "$HARD_PAIRS_PATH" \
      --hard-pair-prob "$HARD_PAIR_PROB"
  fi

  if [ "$RUN_COMPACT" = "1" ]; then
    run_train_stage \
      "edgespot_t4_pcen_triplet_hard_ep${COMPACT_EPISODES}_composite" \
      "edgespot_full" "mel_pcen" "triplet" "4" \
      "$COMPACT_EPOCHS" "$COMPACT_EPISODES" "$COMPACT_N_CLASSES" "$COMPACT_N_SAMPLES" \
      --mining hard \
      --margin 1.0 \
      --hard-pairs-path "$HARD_PAIRS_PATH" \
      --hard-pair-prob "$HARD_PAIR_PROB"

    run_train_stage \
      "edgespot_t4_pcen_ge2e_ep${COMPACT_EPISODES}_composite" \
      "edgespot_full" "mel_pcen" "ge2e" "4" \
      "$COMPACT_EPOCHS" "$COMPACT_EPISODES" "$COMPACT_N_CLASSES" "$COMPACT_N_SAMPLES" \
      --hard-pairs-path "$HARD_PAIRS_PATH" \
      --hard-pair-prob "$HARD_PAIR_PROB"
  fi

  if [ "$RUN_KD" = "1" ]; then
    prepare_teacher_for_kd
    run_train_stage \
      "edgespot_t4_pcen_kd_ge2e_w${KD_WEIGHT}_words${KD_MAX_WORDS}" \
      "edgespot_full" "mel_pcen" "kd_ge2e" "4" \
      "$KD_EPOCHS" "$KD_EPISODES" "$KD_N_CLASSES" "$KD_N_SAMPLES" \
      --train-words-file "train_words_kd_${KD_MAX_WORDS}.json" \
      --max-per-word "$KD_MAX_PER_WORD" \
      --teacher-embeddings-dir "$TEACHER_EMB_DIR" \
      --kd-weight "$KD_WEIGHT" \
      --ge2e-weight 1.0
  fi

  if [ "$RUN_SCAF_ABLATION" = "1" ]; then
    prepare_subset_words "$SCAF_SUBSET_WORDS" "train_words_scaf_${SCAF_SUBSET_WORDS}.json"
    local warmup_id="edgespot_t4_pcen_ge2e_scafwarmup_words${SCAF_SUBSET_WORDS}"
    run_train_stage \
      "$warmup_id" \
      "edgespot_full" "mel_pcen" "ge2e" "4" \
      "$SCAF_WARMUP_EPOCHS" "$SCAF_EPISODES" "$SCAF_N_CLASSES" "$SCAF_N_SAMPLES" \
      --train-words-file "train_words_scaf_${SCAF_SUBSET_WORDS}.json" \
      --max-per-word "$SCAF_MAX_PER_WORD"

    run_train_stage \
      "edgespot_t4_pcen_scaf_ge2e_recovery_w${SCAF_WEIGHT}_s${SCAF_SCALE}_m${SCAF_MARGIN}_words${SCAF_SUBSET_WORDS}" \
      "edgespot_full" "mel_pcen" "scaf_ge2e" "4" \
      "$SCAF_FINETUNE_EPOCHS" "$SCAF_EPISODES" "$SCAF_N_CLASSES" "$SCAF_N_SAMPLES" \
      --train-words-file "train_words_scaf_${SCAF_SUBSET_WORDS}.json" \
      --max-per-word "$SCAF_MAX_PER_WORD" \
      --resume "checkpoints/${warmup_id}_${RUN_ID}/best.pt" \
      --resume-encoder-only \
      --scaf-weight "$SCAF_WEIGHT" \
      --ge2e-weight 1.0 \
      --arcface-scale "$SCAF_SCALE" \
      --arcface-margin "$SCAF_MARGIN"
  fi

  log "All selected development stages finished or skipped. Summary: $SUMMARY_FILE"
  sync_artifacts_once || true
}

main "$@"
