# Colab Pro+ 24h Full MSWC Runbook

> WARNING 2026-06-04: Khong dung runner full all-WAV nay tren Colab local disk
> 236GB. Lan chay truoc da day `/content` khi convert Full MSWC sang WAV. Hay
> dung `docs/colab/mswc_capped_and_flac_runbook_vi.md` va
> `colab/run_mswc_capped_train.sh` cho cap50/cap100; neu can cache dai han thi
> dung `colab/build_mswc_flac_shards.sh`. Neu muon chay gan 6M clip, dung
> `colab/run_mswc_heavy_flac_train.sh` thay vi runner nay.

Mục tiêu: chạy liên tục trong một phiên Colab Pro+ khoảng 24 giờ để lấy kết quả Full MSWC có thể báo cáo, đồng thời không lãng phí thời gian/unit vào việc copy hàng triệu WAV nhỏ lên Google Drive.

## 1. Quyết định quan trọng

Không lưu WAV Full MSWC vào Google Drive.

Lý do: Full MSWC sau khi extract/convert có rất nhiều file nhỏ. Copy hàng triệu WAV nhỏ lên Drive rất chậm, dễ mất nhiều giờ và không giúp train trong phiên hiện tại nhanh hơn. Pipeline Colab này chỉ sync các artifact nhỏ nhưng quan trọng:

- `checkpoints/`
- `results/`
- `reports/`
- `logs_colab/`
- `configs/`

Như vậy nếu Colab disconnect, bạn vẫn giữ được checkpoint, DET curve, result JSON, log và bảng summary. Dữ liệu WAV local trong `/content` sẽ mất khi runtime mất, nhưng đó là trade-off hợp lý để không đốt unit vào Drive I/O.

## 2. Nên chọn GPU nào

Ưu tiên đề xuất:

1. `A100`: lựa chọn cân bằng nhất cho 24h. Đủ nhanh cho train, VRAM lớn, ít lãng phí hơn H100 trong 10-12h đầu khi pipeline chủ yếu tải/giải nén/chuyển audio.
2. `H100`: nhanh nhất cho phần train, nhưng chỉ nên chọn nếu bạn chấp nhận burn unit nhanh. Không giúp nhiều ở bước tải dữ liệu.
3. `L4`: tiết kiệm hơn, phù hợp nếu muốn kéo dài unit, nhưng có thể không hoàn thành nhiều stage full all-clips trong 24h.
4. `T4`: không khuyến nghị cho Full MSWC all-clips. Chỉ dùng cho smoke test, Top500, hoặc manifest cap nhỏ.

Nếu chỉ có một phiên 24h và bạn có 490 units, chọn `A100` trước. Nếu A100 không available và bạn cần kết quả gấp, dùng `H100`. Nếu muốn tiết kiệm unit hơn và chấp nhận ít kết quả hơn, dùng `L4`.

## 3. Thứ tự chạy mặc định

Script `colab/run_full_mswc_24h.sh` chạy theo thứ tự:

1. Cài dependency Colab, không reinstall torch.
2. Tải GSC v2.
3. Tải/extract Full MSWC English local.
4. Convert OPUS sang WAV và xóa OPUS sau khi convert thành công.
5. Build manifest all-clips:
   - `data/mswc_en/splits/train_files_full.json`
   - `data/mswc_en/splits/val_files_full.json`
6. Train/evaluate Full MSWC all-clips:
   - `DSCNN-L + PCEN + GE2E`
   - `EdgeSpotFull T4 + PCEN + GE2E`
7. Nếu còn thời gian:
   - `EdgeSpotFull T4 + PCEN + SCAF+GE2E`
   - Top500 follow-up cho `DSCNN-L + PCEN + GE2E` và `EdgeSpotFull T4 + PCEN + SCAF+GE2E`

Mặc định mỗi full stage dùng:

- `10 epochs`
- `200 episodes/epoch`
- `30-way x 10 samples`
- GSC-dev selection mỗi `2 epochs`
- GSC-dev `5 runs`
- final eval: `dev30` và `test100`

Đây là cấu hình để có kết quả báo cáo trước. Nếu chạy H100/A100 và muốn train lâu hơn, tăng:

```bash
FULL_EPOCHS=20
FULL_EPISODES=300
```

## 4. Cell chạy trên Colab

### 4.1 Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4.2 Kiểm tra GPU

```bash
!nvidia-smi
!df -h /content
```

Nếu `/content` còn dưới khoảng `120GB`, không nên chạy Full MSWC all-clips. Script sẽ dừng sớm trong trường hợp này. Mức khuyến nghị là `150GB+` vì pipeline cần chỗ cho archive, extracted audio, WAV sau convert và checkpoint tạm.

### 4.3 Đưa code vào Colab

Cách A: nếu repo đã có trên GitHub:

```bash
!rm -rf /content/DoAnTotNghiep
!git clone <YOUR_REPO_URL> /content/DoAnTotNghiep
%cd /content/DoAnTotNghiep
```

Cách B: nếu dùng zip từ máy local, upload zip lên Google Drive rồi chạy:

```bash
!rm -rf /content/DoAnTotNghiep
!unzip -q /content/drive/MyDrive/DoAnTotNghiep_code_colab.zip -d /content
%cd /content/DoAnTotNghiep
```

Zip chỉ nên chứa code/config/scripts. Không zip `data/`, `checkpoints/`, `results/`, `node_modules/`.

### 4.4 Chạy pipeline mặc định 24h

```bash
!chmod +x colab/run_full_mswc_24h.sh
!MAX_SECONDS=84600 \
  FULL_EPOCHS=10 \
  FULL_EPISODES=200 \
  RUN_EXTRA_HYBRID=1 \
  RUN_TOP500_AFTER_FULL=1 \
  bash colab/run_full_mswc_24h.sh
```

Nếu dùng H100/A100 và muốn train full lâu hơn:

```bash
!MAX_SECONDS=84600 \
  FULL_EPOCHS=20 \
  FULL_EPISODES=300 \
  RUN_EXTRA_HYBRID=1 \
  RUN_TOP500_AFTER_FULL=0 \
  bash colab/run_full_mswc_24h.sh
```

Nếu chỉ muốn lấy kết quả nhanh nhất cho 2 cấu hình chính:

```bash
!MAX_SECONDS=84600 \
  FULL_EPOCHS=10 \
  FULL_EPISODES=200 \
  RUN_EXTRA_HYBRID=0 \
  RUN_TOP500_AFTER_FULL=0 \
  bash colab/run_full_mswc_24h.sh
```

## 5. Theo dõi tiến độ

Trong Colab cell khác:

```bash
!tail -n 120 /content/DoAnTotNghiep/logs_colab/*/run.log
```

Xem summary:

```bash
!cat /content/DoAnTotNghiep/logs_colab/*/stages.tsv
```

Artifact trên Drive nằm ở:

```text
/content/drive/MyDrive/DoAnTotNghiep_colab_runs/<RUN_ID>/
```

Quan trọng nhất:

- `run.log`
- `logs_colab/<RUN_ID>/stages.tsv`
- `checkpoints/<run_tag>/best.pt`
- `checkpoints/<run_tag>/latest.pt`
- `results/<RUN_ID>/<run_tag>/test100/gsc_edgespot_exact_k10_results.json`
- `results/<RUN_ID>/<run_tag>/test100/gsc_edgespot_exact_det_curve.png`

## 6. Resume

Nếu runtime vẫn còn data local và chỉ cell bị ngắt, chạy lại cùng script. Script sẽ thấy local marker và skip data preparation.

Nếu runtime mất hẳn, WAV local mất. Vì không lưu WAV lên Drive, lần sau phải tải/convert MSWC lại. Tuy nhiên checkpoint/result/log đã nằm trên Drive. Có thể restore checkpoint để train tiếp sau khi data được tạo lại:

```bash
!RESUME_FROM_DRIVE_RUN_ID=<OLD_RUN_ID> \
  RUN_ID=<NEW_RUN_ID> \
  MAX_SECONDS=84600 \
  FULL_EPOCHS=20 \
  FULL_EPISODES=300 \
  bash colab/run_full_mswc_24h.sh
```

## 7. Kỳ vọng thời gian

Ước lượng thực tế phụ thuộc mạnh vào Colab network, disk và GPU:

- GSC v2: vài phút đến khoảng 15 phút.
- Full MSWC download/extract/convert: khoảng 10-12 giờ theo estimate hiện tại của bạn.
- Build full manifest: khoảng 10-40 phút.
- Một full training stage `10 epochs x 200 episodes`: có thể từ vài giờ trên A100/H100 đến lâu hơn nhiều trên L4/T4.
- Final `dev30 + test100`: thường thêm vài chục phút đến hơn một giờ cho mỗi checkpoint.

Nếu phiên chỉ còn ít thời gian sau data preparation, kết quả báo cáo tối thiểu nên lấy:

1. `DSCNN-L + PCEN + GE2E` Full MSWC all-clips test100.
2. Nếu kịp, thêm `EdgeSpotFull T4 + PCEN + GE2E` để so sánh accuracy vs compactness.

## 8. Nội dung có thể báo cáo sau khi chạy

Nếu chỉ xong cấu hình đầu tiên:

- Dataset: Full MSWC English all-clips local, split 2% validation.
- Pipeline: `DSCNN-L + PCEN + GE2E`.
- Training: epochs, episodes, n-way, n-samples, GSC-dev selection.
- Evaluation: GSC-test100, ACC@1%FAR, ACC@5%FAR, AUC, EER, F1, DET curve.

Nếu xong hai cấu hình chính:

- So sánh `DSCNN-L + PCEN + GE2E` với `EdgeSpotFull T4 + PCEN + GE2E`.
- Kết luận theo hai mục tiêu:
  - DSCNN-L: accuracy-oriented.
  - EdgeSpotFull T4: compact edge-oriented.

Nếu xong hybrid:

- Kiểm tra lại câu hỏi nghiên cứu: trên full all-clips, `SCAF+GE2E` có còn tốt hơn `GE2E` đơn lẻ hay không.
