# Runbook Colab MSWC an toàn sau lỗi full WAV

Mục tiêu của runbook này là chạy tiếp thí nghiệm MSWC trên Colab Pro+ mà không lặp lại lỗi đầy `/content`. Trên runtime khoảng 236GB, không chạy lại hướng Full MSWC all-clips WAV.

## 1. Quyết định vận hành

Không dùng:

```bash
bash colab/run_full_mswc_24h.sh
```

trên Colab 236GB.

Lý do: lần chạy trước đã tạo `6,350,474` file WAV, chiếm `172.60GB`, trong khi vẫn còn `706,560` file OPUS chưa convert. Nếu tiếp tục all-clips WAV, tổng dung lượng vượt giới hạn local disk của Colab.

Hướng dùng ngay:

- Chạy `MSWC_MAX_PER_WORD=50` trước.
- Nếu disk còn an toàn, chạy tiếp `MSWC_MAX_PER_WORD=100`.
- Train/evaluate hai cấu hình chính:
  - `DSCNN-L + PCEN + GE2E`
  - `EdgeSpotFull T4 + PCEN + GE2E`
- Chỉ sync artifact nhỏ lên Drive: checkpoint, result JSON, DET curve/report, log, config, script.

Hướng cache dài hạn:

- Convert OPUS -> FLAC theo word shard.
- Đóng thành ít file tar lớn trên Drive.
- Không lưu hàng triệu file audio rời lên Drive.

## 2. Chuẩn bị Colab

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Upload hoặc copy zip code vào `/content`, sau đó giải nén. Nếu zip tạo trên Windows bị cảnh báo backslash path separator, dùng cell Python này:

```python
import zipfile
from pathlib import Path

zip_path = sorted(Path("/content").glob("*.zip"), key=lambda p: p.stat().st_mtime)[-1]
out_root = Path("/content/DoAnTotNghiep")
tmp_root = Path("/content/kws_zip")

if out_root.exists():
    import shutil
    shutil.rmtree(out_root)
if tmp_root.exists():
    import shutil
    shutil.rmtree(tmp_root)
tmp_root.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        normalized = info.filename.replace("\\", "/")
        if normalized.endswith("/"):
            continue
        target = tmp_root / normalized
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as dst:
            dst.write(src.read())

candidate = tmp_root / "DoAnTotNghiep"
if candidate.exists():
    candidate.rename(out_root)
else:
    tmp_root.rename(out_root)

print(out_root)
print(sorted(p.name for p in out_root.iterdir())[:30])
```

Preflight:

```bash
cd /content/DoAnTotNghiep
df -h /content
nvidia-smi
ls colab
```

## 3. Chạy cap50 để có kết quả nhanh

```bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_capped_train.sh

MSWC_MAX_PER_WORD=50 \
RUN_EPOCHS=10 \
RUN_EPISODES=200 \
RUN_DSCNN=1 \
RUN_EDGESPOT=1 \
RUN_EDGESPOT_HYBRID=0 \
bash colab/run_mswc_capped_train.sh
```

Kết quả mong đợi:

- `data/mswc_en/splits/train_files_cap50.json`
- `data/mswc_en/splits/val_files_cap50.json`
- `checkpoints/<run>/best.pt`
- `results/<run>/gsc_dev30`
- `results/<run>/gsc_test100`
- bản sync trên Drive trong `/content/drive/MyDrive/DoAnTotNghiep_colab_runs/<RUN_ID>/`

Monitor:

```bash
cd /content/DoAnTotNghiep
tail -f "$(find logs_colab -name run.log | sort | tail -n 1)"
```

Nếu muốn xem process:

```bash
ps -ef | grep -E "run_mswc_capped|download_mswc|convert_opus|train.py|evaluate_edgespot" | grep -v grep
```

## 4. Chạy cap100 nếu disk còn an toàn

Chỉ chạy khi `df -h /content` vẫn còn dư tốt. Nếu `/content` đã trên khoảng 85-90%, dừng ở cap50 và báo cáo cap50.

```bash
cd /content/DoAnTotNghiep

MSWC_MAX_PER_WORD=100 \
RUN_EPOCHS=10 \
RUN_EPISODES=200 \
RUN_DSCNN=1 \
RUN_EDGESPOT=1 \
RUN_EDGESPOT_HYBRID=0 \
bash colab/run_mswc_capped_train.sh
```

Ghi chú báo cáo: cap50/cap100 là ablation mở rộng từ Full MSWC, không claim là full all-clips final.

## 5. Chạy heavy FLAC gần 6 triệu clip

Nếu mục tiêu là chạy lớn hơn, gần `6M` clip, dùng runner FLAC riêng này thay vì runner WAV cap50/cap100.

Ý nghĩa:

- `max_per_word=180` là tối đa `180 clip/word`, không phải 180 word.
- Script sẽ estimate từ metadata và chọn cap nhỏ nhất trong `180, 200, 220` sao cho gần/đạt `TARGET_FILES=6,000,000`.
- Audio được convert sang `.flac`, xóa `.opus`, rồi train trực tiếp từ manifest `.flac`.
- Không dùng lại runtime đã từng đầy disk hoặc đã có audio cũ, trừ khi cố ý bật `ALLOW_EXISTING_DATA=1`.

Lệnh khuyến nghị trên fresh A100 runtime:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/run_mswc_heavy_flac_train.sh

TARGET_FILES=6000000 \
MIN_CAP=180 \
MAX_CAP=220 \
CAP_STEP=20 \
RUN_EPOCHS=15 \
RUN_EPISODES=800 \
RUN_DSCNN=1 \
RUN_EDGESPOT=1 \
CONVERT_WORKERS=12 \
CONVERT_BATCH_SIZE=16 \
FLAC_COMPRESSION_LEVEL=3 \
bash colab/run_mswc_heavy_flac_train.sh
```

Nếu muốn ép đúng một cap, ví dụ cap180:

```bash
MSWC_MAX_PER_WORD=180 \
RUN_EPOCHS=15 \
RUN_EPISODES=800 \
bash colab/run_mswc_heavy_flac_train.sh
```

Output mong đợi:

- `logs_colab/<RUN_ID>/cap_estimate.json`
- `data/mswc_en/splits/train_files_cap<N>_flac.json`
- `data/mswc_en/splits/val_files_cap<N>_flac.json`
- `checkpoints/<run>/best.pt`
- `results/<run>/gsc_dev30`
- `results/<run>/gsc_test100`
- artifact sync trên Drive.

Nếu script dừng vì disk vượt ngưỡng:

- không chạy tiếp train;
- giảm `TARGET_FILES` hoặc dùng cap thấp hơn;
- không chạy lại hướng WAV.

## 6. FLAC shard cache dài hạn

Chỉ dùng phần này nếu muốn tạo cache tái sử dụng nhiều session. Không cần chạy trước cap50/cap100.

Ý tưởng:

- Runtime fresh.
- Tải/extract Full MSWC ở dạng OPUS.
- Chọn word shard.
- Convert shard đó sang FLAC trong `/content`.
- Đóng thành một file tar lớn trên Drive.
- Xóa folder tạm sau khi tar xong.

Session 1:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/build_mswc_flac_shards.sh

PREPARE_FULL_OPUS=1 \
SHARD_COUNT=2 \
SHARD_INDEX=0 \
CONVERT_WORKERS=12 \
bash colab/build_mswc_flac_shards.sh
```

Session 2:

```bash
cd /content/DoAnTotNghiep
chmod +x colab/build_mswc_flac_shards.sh

PREPARE_FULL_OPUS=1 \
SHARD_COUNT=2 \
SHARD_INDEX=1 \
CONVERT_WORKERS=12 \
bash colab/build_mswc_flac_shards.sh
```

Output trên Drive:

```text
/content/drive/MyDrive/DoAnTotNghiep_colab_runs/audio_cache/flac_shards/
  mswc_flac_shard_0_of_2.tar
  mswc_flac_shard_1_of_2.tar
```

Không nên:

- Không copy `data/mswc_en/clips` lên Drive.
- Không tạo folder Drive chứa hàng triệu `.flac`.
- Không extract full FLAC cache vào `/content` nếu runtime vẫn chỉ có 236GB.

## 7. Tiêu chí chấp nhận

Cap50/cap100:

- Không có `df -h /content` đạt 100%.
- Có manifest cap tương ứng.
- Có `best.pt` cho ít nhất `DSCNN-L + PCEN + GE2E`.
- Có `GSC-test100` result trong `results/`.
- Artifact đã được sync lên Drive.

Heavy FLAC:

- `cap_estimate.json` ghi rõ selected cap và expected clip count.
- Manifest thật có tổng số file gần mục tiêu `6M`, nếu metadata cho phép.
- `/content` không đạt 100%; script dừng nếu vượt ngưỡng an toàn.
- Có `GSC-test100` cho ít nhất `DSCNN-L + PCEN + GE2E`.

FLAC shard:

- Mỗi shard là một file `.tar` trên Drive.
- Trong tar có `manifest.json`, `words.txt`, và `clips/<word>/*.flac`.
- Không có audio rời được lưu lên Drive.
