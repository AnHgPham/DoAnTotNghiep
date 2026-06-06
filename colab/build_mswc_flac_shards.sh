#!/usr/bin/env bash
set -Eeuo pipefail

# Build MSWC FLAC cache shards for Colab.
#
# This script is for long-term cache only. It writes a small number of large
# tar files to Google Drive, not millions of loose audio files. Use a fresh
# runtime with OPUS sources when possible. Do not run this on a runtime that is
# already full of all-WAV MSWC clips.

PROJECT_DIR="${PROJECT_DIR:-/content/DoAnTotNghiep}"
DATA_DIR="${DATA_DIR:-data/mswc_en}"
DRIVE_RUN_ROOT="${DRIVE_RUN_ROOT:-/content/drive/MyDrive/DoAnTotNghiep_colab_runs}"
CACHE_OUT="${CACHE_OUT:-$DRIVE_RUN_ROOT/audio_cache/flac_shards}"
TMP_ROOT="${TMP_ROOT:-/content/mswc_flac_shard_tmp}"

SHARD_COUNT="${SHARD_COUNT:-2}"
SHARD_INDEX="${SHARD_INDEX:-0}" # 0..SHARD_COUNT-1, "all", or -1
PREPARE_FULL_OPUS="${PREPARE_FULL_OPUS:-0}"

MSWC_MIN_CLIPS="${MSWC_MIN_CLIPS:-1}"
MSWC_VAL_FRACTION="${MSWC_VAL_FRACTION:-0.02}"
MSWC_SPLIT_SEED="${MSWC_SPLIT_SEED:-42}"
MSWC_MIRROR="${MSWC_MIRROR:-cloudflare}"
MAX_FILES_PER_WORD="${MAX_FILES_PER_WORD:-0}"

CONVERT_WORKERS="${CONVERT_WORKERS:-8}"
FFMPEG_COMPRESSION_LEVEL="${FFMPEG_COMPRESSION_LEVEL:-3}"
ALLOW_WAV_SOURCE="${ALLOW_WAV_SOURCE:-0}"
DELETE_TMP_AFTER_TAR="${DELETE_TMP_AFTER_TAR:-1}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_drive() {
  if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive is not mounted. Run: from google.colab import drive; drive.mount('/content/drive')" >&2
    exit 2
  fi
}

install_deps() {
  log "Installing ffmpeg/rsync if needed"
  apt-get update -qq
  apt-get install -y -qq ffmpeg rsync
}

print_env() {
  log "Environment"
  pwd
  df -h /content || true
  nvidia-smi || true
}

prepare_full_opus() {
  if [ "$PREPARE_FULL_OPUS" != "1" ]; then
    return 0
  fi

  log "Preparing Full MSWC English OPUS sources only"
  python data/download_mswc.py \
    --min-clips "$MSWC_MIN_CLIPS" \
    --val-fraction "$MSWC_VAL_FRACTION" \
    --split-seed "$MSWC_SPLIT_SEED" \
    --max-per-word 0 \
    --mirror "$MSWC_MIRROR"

  rm -f "$DATA_DIR/en.tar.gz" 2>/dev/null || true
}

check_source_audio() {
  local clips="$PROJECT_DIR/$DATA_DIR/clips"
  if [ ! -d "$clips" ]; then
    echo "ERROR: missing source clips directory: $clips" >&2
    echo "Run with PREPARE_FULL_OPUS=1 or prepare MSWC first." >&2
    exit 2
  fi

  log "Source audio summary"
  find "$clips" -type f \( -name '*.opus' -o -name '*.wav' -o -name '*.flac' -o -name '*.ogg' \) \
    -printf '%s %f\n' 2>/dev/null \
    | awk '
      {
        n += 1; s += $1;
        if ($2 ~ /\.opus$/) { no += 1; so += $1 }
        else if ($2 ~ /\.wav$/) { nw += 1; sw += $1 }
        else if ($2 ~ /\.flac$/) { nf += 1; sf += $1 }
        else if ($2 ~ /\.ogg$/) { ng += 1; sg += $1 }
      }
      END {
        printf("total_files=%d total_size=%.2fG\n", n, s/1024/1024/1024);
        printf("opus_files=%d opus_size=%.2fG\n", no, so/1024/1024/1024);
        printf("wav_files=%d wav_size=%.2fG\n", nw, sw/1024/1024/1024);
        printf("flac_files=%d flac_size=%.2fG\n", nf, sf/1024/1024/1024);
        printf("ogg_files=%d ogg_size=%.2fG\n", ng, sg/1024/1024/1024);
      }'

  local wav_count
  wav_count="$(CLIPS_DIR="$clips" python - <<'PY'
import os
count = 0
for root, _, files in os.walk(os.environ["CLIPS_DIR"]):
    for name in files:
        if name.endswith(".wav"):
            count += 1
            if count > 1000:
                print(count)
                raise SystemExit
print(count)
PY
)"
  if [ "$wav_count" -gt 1000 ] && [ "$ALLOW_WAV_SOURCE" != "1" ]; then
    echo "ERROR: many WAV source files detected. This usually means the runtime is already using too much disk." >&2
    echo "Use a fresh runtime with OPUS sources, or set ALLOW_WAV_SOURCE=1 if you intentionally accept the disk risk." >&2
    exit 3
  fi
}

build_one_shard() {
  local idx="$1"
  local tmp_dir="$TMP_ROOT/shard_${idx}_of_${SHARD_COUNT}"
  local partial="$CACHE_OUT/mswc_flac_shard_${idx}_of_${SHARD_COUNT}.tar.partial"
  local final="$CACHE_OUT/mswc_flac_shard_${idx}_of_${SHARD_COUNT}.tar"

  log "Building FLAC shard $idx/$SHARD_COUNT"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir" "$CACHE_OUT"

  export PROJECT_DIR DATA_DIR TMP_SHARD_DIR="$tmp_dir"
  export CURRENT_SHARD_INDEX="$idx" SHARD_COUNT MAX_FILES_PER_WORD
  export CONVERT_WORKERS FFMPEG_COMPRESSION_LEVEL

  python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

project_dir = Path(os.environ["PROJECT_DIR"])
data_dir = project_dir / os.environ.get("DATA_DIR", "data/mswc_en")
clips_dir = data_dir / "clips"
tmp_dir = Path(os.environ["TMP_SHARD_DIR"])
shard_index = int(os.environ["CURRENT_SHARD_INDEX"])
shard_count = int(os.environ["SHARD_COUNT"])
max_files_per_word = int(os.environ.get("MAX_FILES_PER_WORD", "0"))
workers = int(os.environ.get("CONVERT_WORKERS", "8"))
compression = os.environ.get("FFMPEG_COMPRESSION_LEVEL", "3")

audio_priority = {".flac": 0, ".wav": 1, ".opus": 2, ".ogg": 3}
queue_limit = max(workers * 8, 16)


def load_word_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        if "words" in obj:
            obj = obj["words"]
        else:
            obj = list(obj.keys())
    return [str(x) for x in obj]


def load_words() -> list[str]:
    split_dir = data_dir / "splits"
    words: list[str] = []
    words.extend(load_word_json(split_dir / "train_words.json"))
    words.extend(load_word_json(split_dir / "val_words.json"))
    if not words:
        words = [p.name for p in clips_dir.iterdir() if p.is_dir()]
    words = sorted({w for w in words if (clips_dir / w).is_dir()})
    return words


def collect_word_files(word: str) -> list[Path]:
    word_dir = clips_dir / word
    best_by_stem: dict[str, Path] = {}
    for entry in os.scandir(word_dir):
        if not entry.is_file():
            continue
        path = Path(entry.path)
        suffix = path.suffix.lower()
        if suffix not in audio_priority:
            continue
        stem = path.stem
        current = best_by_stem.get(stem)
        if current is None or audio_priority[suffix] < audio_priority[current.suffix.lower()]:
            best_by_stem[stem] = path
    files = sorted(best_by_stem.values(), key=lambda p: p.name)
    if max_files_per_word > 0:
        files = files[:max_files_per_word]
    return files


def convert_one(src: Path, dst: Path) -> tuple[str, str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    if dst.exists() and dst.stat().st_size > 0:
        return ("skip", suffix, dst.stat().st_size)
    tmp = dst.with_suffix(".flac.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        if suffix == ".flac":
            shutil.copy2(src, tmp)
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-compression_level",
                    str(compression),
                    str(tmp),
                ],
                check=True,
            )
        tmp.replace(dst)
        return ("ok", suffix, dst.stat().st_size)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> int:
    if not clips_dir.exists():
        print(f"missing clips dir: {clips_dir}", file=sys.stderr)
        return 2
    if workers < 1:
        print("CONVERT_WORKERS must be >= 1", file=sys.stderr)
        return 2

    words = load_words()
    selected_words = [w for i, w in enumerate(words) if i % shard_count == shard_index]
    if not selected_words:
        print(f"no words selected for shard {shard_index}/{shard_count}", file=sys.stderr)
        return 2

    (tmp_dir / "clips").mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    output_bytes = 0
    queued = 0
    completed = 0
    failed = 0
    start = time.time()
    pending = set()

    def drain(pool_pending, return_when=FIRST_COMPLETED):
        nonlocal completed, failed, output_bytes
        done, pool_pending = wait(pool_pending, return_when=return_when)
        for fut in done:
            try:
                status, suffix, size = fut.result()
                status_counts[status] += 1
                source_counts[suffix] += 1
                output_bytes += size
            except Exception as exc:
                failed += 1
                print(f"[ERROR] conversion failed: {exc}", file=sys.stderr)
            completed += 1
            if completed % 5000 == 0:
                elapsed = max(time.time() - start, 1.0)
                print(
                    f"completed={completed} queued={queued} failed={failed} "
                    f"rate={completed/elapsed:.2f}/s output={output_bytes/1024/1024/1024:.2f}G",
                    flush=True,
                )
        return pool_pending

    print(
        f"shard={shard_index}/{shard_count} words={len(selected_words)} "
        f"workers={workers} max_files_per_word={max_files_per_word}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for word_i, word in enumerate(selected_words, start=1):
            files = collect_word_files(word)
            for src in files:
                dst = tmp_dir / "clips" / word / f"{src.stem}.flac"
                while len(pending) >= queue_limit:
                    pending = drain(pending, FIRST_COMPLETED)
                pending.add(pool.submit(convert_one, src, dst))
                queued += 1
            if word_i % 100 == 0:
                print(
                    f"queued_words={word_i}/{len(selected_words)} queued_files={queued} "
                    f"pending={len(pending)}",
                    flush=True,
                )

        while pending:
            pending = drain(pending, FIRST_COMPLETED)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "source_clips_dir": str(clips_dir),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "word_count": len(selected_words),
        "file_count": queued,
        "failed_count": failed,
        "output_size_gb": output_bytes / 1024 / 1024 / 1024,
        "source_suffix_counts": dict(sorted(source_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "max_files_per_word": max_files_per_word,
        "ffmpeg_compression_level": compression,
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (tmp_dir / "words.txt").write_text("\n".join(selected_words) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failed else 0


raise SystemExit(main())
PY

  log "Creating tar shard on Drive: $final"
  rm -f "$partial"
  tar -cf "$partial" -C "$tmp_dir" .
  mv "$partial" "$final"

  if [ "$DELETE_TMP_AFTER_TAR" = "1" ]; then
    rm -rf "$tmp_dir"
  fi

  log "Finished shard: $final"
  ls -lh "$final"
  df -h /content || true
}

main() {
  require_drive
  cd "$PROJECT_DIR"
  install_deps
  print_env
  prepare_full_opus
  check_source_audio

  if [ "$SHARD_INDEX" = "all" ] || [ "$SHARD_INDEX" = "-1" ]; then
    local i
    for ((i = 0; i < SHARD_COUNT; i++)); do
      build_one_shard "$i"
    done
  else
    if [ "$SHARD_INDEX" -lt 0 ] || [ "$SHARD_INDEX" -ge "$SHARD_COUNT" ]; then
      echo "ERROR: SHARD_INDEX must be 0..$((SHARD_COUNT - 1)), all, or -1" >&2
      exit 2
    fi
    build_one_shard "$SHARD_INDEX"
  fi

  log "Available FLAC shards"
  ls -lh "$CACHE_OUT" || true
}

main "$@"
