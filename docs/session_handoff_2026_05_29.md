# Handoff Session KWS - 2026-05-29

File này tóm tắt toàn bộ mạch làm việc để một AI khác hoặc người khác có thể tiếp tục dự án mà không cần đọc lại toàn bộ chat.

## 1. Bối Cảnh Dự Án

Dự án là hệ thống **few-shot open-set keyword spotting**:

- Người dùng enroll vài mẫu âm thanh cho mỗi từ khóa.
- Encoder biến audio thành embedding.
- Prototype/centroid được tính từ các mẫu enroll.
- Audio mới được score bằng khoảng cách L2 đến prototype.
- Nếu khoảng cách vượt ngưỡng hoặc bị policy reject thì trả `unknown`.
- Có demo web cho:
  - enroll;
  - single detection;
  - long audio detection;
  - open-set rejection test;
  - model switcher;
  - calibration;
  - streaming/state-machine.

Mục tiêu nghiên cứu hiện tại:

- Hiểu và thử nghiệm các kết hợp model/feature/loss.
- So sánh **DSCNN** và **EdgeSpotFull T4**.
- So sánh **MFCC**, **mel/PCEN**.
- So sánh loss: **Triplet**, **SCAF**, **GE2E**, **SCAF+GE2E**.
- Chạy trên MSWC Microset, Top500, và đang chuẩn bị full MSWC.

## 2. Câu Chuyện Kết Quả Hiện Tại

### 2.1. Microset

Microset là mốc chính để chọn cấu hình.

Các cấu hình đã thử:

1. `DSCNN-L + MFCC + Triplet Loss`
2. `EdgeSpotFull T4 + PCEN + SCAF`
3. `EdgeSpotFull T4 + PCEN + SCAF+GE2E`

Số liệu người dùng đã cung cấp:

| Cấu hình | AUC | EER | FRR@5%FAR | Open-set ACC | Keyword ACC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L + MFCC + Triplet | 0.9122 +/- 0.0074 | 0.1822 +/- 0.0112 | 0.4058 +/- 0.0360 | 0.8054 +/- 0.0087 | 0.6839 +/- 0.0181 | 0.6641 +/- 0.0168 | 0.8179 +/- 0.0112 | 0.7330 +/- 0.0147 |
| EdgeSpotFull T4 + PCEN + SCAF | 0.9569 +/- 0.0030 | 0.1189 +/- 0.0075 | 0.2061 +/- 0.0222 | 0.8521 +/- 0.0057 | 0.7452 +/- 0.0168 | 0.7655 +/- 0.0128 | 0.8811 +/- 0.0076 | 0.8192 +/- 0.0106 |
| EdgeSpotFull T4 + PCEN + SCAF+GE2E | 0.9561 +/- 0.0043 | 0.1154 +/- 0.0086 | 0.2139 +/- 0.0324 | 0.8612 +/- 0.0081 | 0.7766 +/- 0.0141 | 0.7715 +/- 0.0148 | 0.8845 +/- 0.0087 | 0.8241 +/- 0.0122 |

Kết luận hiện tại:

- `SCAF-only` nhỉnh hơn nhẹ ở AUC và FRR@5%FAR.
- `SCAF+GE2E` tốt hơn ở phần lớn chỉ số quan trọng cho demo/thesis: Open-set ACC, Keyword ACC, Precision, Recall, F1, EER.
- Vì vậy hướng chính là `EdgeSpotFull T4 + PCEN + SCAF+GE2E`.

### 2.2. Top500

Mục tiêu: mở rộng từ Microset sang MSWC Top500 full clips.

Các mốc đã có trong log:

| Mốc | Checkpoint | Split | Runs | k-shot | ACC@1%FAR | ACC@5%FAR / Open-set ACC | FRR@5%FAR | AUC | EER | Keyword ACC | F1 | Ghi chú |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Chọn checkpoint trong lúc train | epoch25 | GSC-dev | 3 | 10 | 87.61% | 89.20% | 20.00% | - | - | - | - | Best theo GSC-dev trong run đầu |
| Eval lại checkpoint epoch25 | epoch25 | GSC-dev | 30 | 10 | - | 89.33% | 19.57% | 95.01% | 11.37% | 89.68% | 82.66% | Eval ổn định hơn trên dev |
| Eval checkpoint epoch25 | epoch25 | GSC-test | 100 | 10 | - | 88.57% | 23.13% | 94.63% | 12.13% | 91.06% | 81.57% | Test100 |
| Train lại, checkpoint local hiện có | epoch13 | GSC-dev | 30 | 10 | 86.68% | 88.87% | 20.36% | 95.12% | 12.03% | 88.86% | 81.71% | Bị dừng do hết Colab compute units |

Lưu ý quan trọng:

- Epoch25 từng có log tốt nhưng checkpoint/artifact local không còn vì Colab mất session/chưa package kịp.
- Checkpoint chắc chắn hiện có là epoch13.
- Với email/thesis, phải viết trung thực: epoch25 là kết quả từng chạy trong log; epoch13 là artifact local hiện có.

### 2.3. Open-set Demo

Open-set UI đã được cải thiện thành flow sampled GSC.

Preset chính:

- Known/enrolled 17 từ:
  `yes, stop, happy, bird, dog, tree, marvin, four, learn, wow, sheila, zero, down, left, right, off, three`
- Unknown 17 từ:
  `no, go, up, on, one, two, five, six, seven, eight, nine, bed, cat, house, backward, forward, follow`
- Heldout:
  `visual`

Các policy đã thử:

- Per-class ON/OFF.
- Close-word guard ON/OFF.
- Accept margin.

Kết luận thực nghiệm demo-level:

- `Guard ON + Per-class OFF` đang là cấu hình cân bằng tốt nhất cho open-set demo.
- `Guard OFF` thường tăng keyword recognition nhưng làm unknown bị accept nhiều hơn, khiến false accept rate cao.
- Open-set UI là sampled/demo-level evaluation, không thay thế `gsc_edgespot_exact test100`.

## 3. Giải Thích Paper / Lý Do Chọn Hướng

Các thành phần đã được dùng trong email/thesis:

- **EdgeSpot-4 / EdgeSpotFull T4**
  - Backbone nhẹ cho KWS.
  - Có PCEN để ổn định trước khác biệt âm lượng/nhiễu.
  - Có temporal block/attention để học đặc trưng theo thời gian.
  - Paper EdgeSpot 01/2026 có mốc ACC@1%FAR 10-shot khoảng 82.0%.

- **SCAF / Sub-center ArcFace**
  - Từ paper: *Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces*, Deng et al., ECCV 2020.
  - Ý tưởng dùng margin angular/sub-center để làm embedding cùng lớp gần nhau, khác lớp xa nhau.
  - Trong project dùng như loss để tạo embedding space phân tách tốt hơn.

- **GE2E**
  - Từ paper: *GE2E-KWS: Generalized End-to-End Training and Evaluation for Zero-shot Keyword Spotting*, 2024.
  - GE2E đưa embedding gần centroid/prototype đúng lớp và xa centroid lớp khác.
  - Phù hợp với few-shot enrollment vì inference cũng dùng prototype/centroid.
  - GE2E-KWS paper có DET-AUC 0.504% so với Triplet 1.283% cùng model 419KB, tức GE2E có DET-AUC chỉ khoảng 39% của Triplet.

Ghi chú:

- GE2E không phải thành phần gốc của EdgeSpot paper.
- Đây là cải tiến/kết hợp trong project.
- Không claim reproduction đầy đủ EdgeSpot.

## 4. Các Việc UI / Backend Đã Làm

### 4.1. Model Switcher

Mục tiêu:

- Chọn giữa Top500 epoch13, Microset epoch05, legacy DSCNN.
- Có quick switcher ở thanh trên và card chi tiết trong Model Info.
- Khi đổi model, hỏi:
  - rebuild enrollment nếu có waveform cache;
  - hoặc clear enrollment.

Backend liên quan:

- `GET /api/model/profiles`
- `POST /api/model/select`

### 4.2. Long Audio UI

Vấn đề cũ:

- Giao diện long audio quá rối.
- 50+ detection card bung hết ra màn hình.
- Transcript dọc dài, khó đọc.
- Timeline nhãn bị dính.

Plan đã implement:

- Summary cards.
- Policy cards.
- Sequence chips có collapse/show all.
- Expected/detected timeline scroll ngang.
- Error review section chỉ hiện MISS/ERR/rejected trước.
- Compact table cho toàn bộ detections.
- Top-3 candidates nằm trong details.

Vấn đề đã phát hiện:

- Có lúc timing JSON không được nhận đúng, dẫn đến tất cả detection bị `EXTRA` và `Expected = -`.
- Cần kiểm tra lại form upload timing JSON và field gửi lên backend nếu lỗi tái diễn.

### 4.3. Long Audio Result Logic

Đã thêm/định nghĩa:

- All accuracy.
- Enrolled-only accuracy.
- Missed expected cards.
- Miss reasons:
  - no overlap;
  - threshold reject;
  - guard reject;
  - wrong prediction;
  - out-of-enrollment;
  - VAD/cooldown skip.

### 4.4. Detection Policy

Vấn đề cũ:

- UI tắt `Close-word guard` nhưng backend vẫn reject vì top1/top2 margin.

Plan đã implement:

- Backend là nguồn sự thật.
- `build_detection_policy(threshold, use_per_class, use_close_word_guard)`
- Nếu guard OFF thì `accept_margin = 0.0`.
- Response luôn trả:
  - threshold;
  - use_per_class;
  - close_word_guard;
  - accept_margin;
  - engine.

UI hiển thị:

- `Ngưỡng từng lớp: BẬT/TẮT`
- `Chặn từ gần nhau: BẬT/TẮT`
- `Accept margin`

### 4.5. Open-set Test V1

Endpoint mới:

- `POST /api/open-set/test`
- `POST /api/open-set/calibrate`

Metric:

- known tested;
- unknown tested;
- candidate label count;
- keyword acc;
- unknown reject acc;
- false accept rate;
- false reject rate;
- open-set acc;
- balanced score.

Quan trọng:

- Khi dùng preset 17/17, scorer phải restrict `candidate_words` chỉ còn 17 known words.
- Unknown words không được là candidate dù trong session từng có prototype.

## 5. Các File Code Quan Trọng Đã/Sắp Sửa

Worktree đang dirty, nhiều thay đổi có thể đến từ các bước trước. Không tự revert.

Các file quan trọng trong mạch hiện tại:

- `scripts/train.py`
  - Đã thêm/đang có `--feature-type {auto,mfcc,mel,mel_pcen}`.
  - `auto`: DSCNN dùng MFCC; EdgeSpot dùng mel/PCEN.
  - `mfcc`: cho phép EdgeSpot chạy ablation với MFCC.
  - `mel_pcen`: dùng mel + trainable PCEN.
  - Checkpoint lưu `feature_type`.

- `src/models/dscnn.py`
  - Đã hỗ trợ optional PCEN/mel input.
  - DSCNN có thể chạy:
    - MFCC `(47, 10)`;
    - mel/PCEN `(40, 101)`.

- `src/models/edgespot_full.py`
  - Đã sửa pooling để EdgeSpotFull chạy được input MFCC ngắn hơn.
  - Thay squeeze cứng bằng mean trên chiều frequency/time phù hợp.

- `src/models/edgespot_lite.py`
  - Tương tự EdgeSpotFull, hỗ trợ ablation MFCC.

- `data/download_mswc.py`
  - Đã sửa lỗi nghiêm trọng: extract lỗi thì phải raise, không được tiếp tục bằng partial dataset.
  - Trước đây lỗi `Transport endpoint is not connected` vẫn bị coi như partial success.

- `server/server_setup_full_mswc.sh`
  - Setup full MSWC trên server.
  - Đã thêm `--keep-archive`.
  - Đã sửa để mặc định chỉ chuẩn bị data, không tự train fixed EdgeSpotFull+SCAF+GE2E.
  - Nếu muốn train fixed run thì set `RUN_FIXED_FULL_MSWC_TRAIN=1`.

- `server/run_full_mswc_experiment_matrix.sh`
  - Matrix chạy các tổ hợp DSCNN/EdgeSpot + MFCC/PCEN + loss.

- `server/wait_then_run_full_mswc_matrix.sh`
  - Đợi setup data xong rồi chạy matrix.

- `requirements-server-cu102.txt`
  - Dùng cho env CUDA 10.2 trên ict6.

## 6. Matrix Thử Nghiệm Full MSWC Hiện Tại

Không dùng BCResNet trong matrix hiện tại.

Các tổ hợp cần chạy:

### DSCNN + MFCC

1. DSCNN + MFCC + Triplet
2. DSCNN + MFCC + SCAF
3. DSCNN + MFCC + GE2E
4. DSCNN + MFCC + SCAF+GE2E

### DSCNN + PCEN

5. DSCNN + PCEN + Triplet
6. DSCNN + PCEN + SCAF
7. DSCNN + PCEN + GE2E
8. DSCNN + PCEN + SCAF+GE2E

### EdgeSpotFull T4 + MFCC

9. EdgeSpotFull T4 + MFCC + Triplet
10. EdgeSpotFull T4 + MFCC + SCAF
11. EdgeSpotFull T4 + MFCC + GE2E
12. EdgeSpotFull T4 + MFCC + SCAF+GE2E

### EdgeSpotFull T4 + PCEN

13. EdgeSpotFull T4 + PCEN + Triplet
14. EdgeSpotFull T4 + PCEN + SCAF
15. EdgeSpotFull T4 + PCEN + GE2E
16. EdgeSpotFull T4 + PCEN + SCAF+GE2E

Mục tiêu matrix:

- Hiểu model nào tốt hơn.
- Hiểu feature nào tốt hơn.
- Hiểu loss nào tốt hơn.
- Có dữ liệu thực nghiệm để giải thích vì sao chọn EdgeSpotFull T4 + PCEN + SCAF+GE2E.

## 7. Server USTH / ict6

### 7.1. Cách Vào Server

Từ Windows:

```powershell
ssh -p <port> <user>@<lab-gateway>
```

Sau đó từ `frontend`:

```bash
ssh ict6
```

Hoặc dùng ProxyJump trực tiếp từ local:

```powershell
ssh -J <user>@<lab-gateway>:<port> <user>@ict6
```

### 7.2. Working Directory

Trên ict6:

```bash
cd /storage/<user>/an_kws/DoAnTotNghiep
```

### 7.3. Conda Env Đúng

Không dùng base env vì PyTorch trong base là CUDA mới, driver server cũ không chạy được.

Env đúng:

```bash
source /home/<user>/anaconda3/etc/profile.d/conda.sh
conda activate kws_cu102
export CUDA_VISIBLE_DEVICES=4
```

Kiểm tra:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Kết quả mong muốn:

- `torch 1.12.1+cu102`
- CUDA available = True
- GPU = Tesla K80

### 7.4. Lưu Ý Torchaudio

Đã từng gặp lỗi:

```text
OSError: libtorch_hip.so: cannot open shared object file
Bus error
```

Nguyên nhân:

- `torchaudio` bản ROCm hoặc mismatch với torch CUDA 10.2.

Đã xử lý bằng package/code no-torchaudio cho py39 và `requirements-server-cu102.txt`.

Nếu gặp lại:

- Không cài torch/torchaudio lung tung trong env.
- Kiểm tra `pip show torch torchaudio`.
- Ưu tiên tránh phụ thuộc `torchaudio` nếu code đã chuyển sang `soundfile/librosa`.

## 8. Trạng Thái Server Mới Nhất Trong Session

Thời điểm kiểm tra gần nhất: 2026-05-29 khoảng 11:32 Asia/Bangkok.

Session tmux:

- `kws_full_mswc`: còn tồn tại.
- `kws_full_matrix_wait`: còn tồn tại.

Full archive:

```text
/storage/<user>/an_kws/DoAnTotNghiep/data/mswc_en/en.tar.gz
size: 33G
downloaded at: 2026-05-29 10:58
```

Log cho thấy:

```text
Step 3: Audio download
en.tar.gz (alibaba): 100%|██████████| 34.8G/34.8G [58:08<00:00, 9.99MB/s]
Downloaded: data/mswc_en/en.tar.gz (32.5 GB)
Step 4: Extracting target words
```

Process còn sống:

```text
PID 2962205
python data/download_mswc.py --min-clips 1 --val-fraction 0.02 --split-seed 42 --max-per-word 0 --mirror alibaba --keep-archive
```

Nó đang ở bước extract. Có thể nhìn như đứng im vì log không in từng file.

Kiểm tra process:

```bash
ps -p 2962205 -o pid,ppid,stat,etime,%cpu,%mem,wchan:24,cmd
```

Kiểm tra tmux:

```bash
tmux attach -t kws_full_mswc
```

Detach đúng cách:

```text
Ctrl+B rồi D
```

Không nên dùng:

```bash
du -sh /storage/<user>/an_kws/DoAnTotNghiep/data/mswc_en/clips
```

Lý do:

- Trên NFS, `du` từng bị treo trong `fuse_lock_inode`.
- Nó làm khó kiểm tra trạng thái và có thể gây nghẽn.

## 9. Các Lỗi Server Đã Gặp

### 9.1. Không resolve `ict14`

Từ Windows:

```text
ssh: Could not resolve hostname ict14
```

Cách đúng:

```powershell
ssh -p <port> <user>@<lab-gateway>
ssh ict6
```

### 9.2. `nvidia-smi` không có ở frontend

Frontend không có GPU. Phải SSH tiếp vào `ict6`.

### 9.3. PyTorch base env không chạy CUDA

Base env:

- Python 3.13.
- Torch 2.11 + CUDA 13.
- Driver server chỉ CUDA 10.2.

Lỗi:

```text
CUDA initialization: The NVIDIA driver on your system is too old
cuda available: False
```

Dùng env:

```bash
conda activate kws_cu102
```

### 9.4. `unzip` không có

Server thiếu `unzip`. Dùng Python để extract zip:

```bash
python - <<'PY'
import zipfile
from pathlib import Path
zip_path = Path("DoAnTotNghiep_code.zip")
out_dir = Path("DoAnTotNghiep")
out_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(out_dir)
PY
```

Sau này đã dùng `.tar.gz` tốt hơn.

### 9.5. Python 3.9 Type Hint Lỗi

Lỗi:

```text
TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'torch._C._TensorMeta'
```

Nguyên nhân:

- Code dùng type hint kiểu Python 3.10: `list[int] | torch.Tensor`.
- Server env là Python 3.9.

Đã upload bản code no-torchaudio/py39 fix.

### 9.6. Full MSWC Extract Lỗi NFS

Lỗi cũ:

```text
Extraction error: [Errno 107] Transport endpoint is not connected
```

Vấn đề nghiêm trọng:

- Code cũ chỉ log error nhưng vẫn return partial clip counts.
- Setup cũ xóa archive sau đó.

Đã sửa:

- `data/download_mswc.py`: extraction error thì raise RuntimeError.
- `server_setup_full_mswc.sh`: thêm `--keep-archive`.

## 10. Lệnh Theo Dõi Server Nên Dùng

Vào server:

```powershell
ssh -p <port> <user>@<lab-gateway>
ssh ict6
```

Attach tmux:

```bash
tmux attach -t kws_full_mswc
```

Xem process:

```bash
pgrep -af "server_setup_full_mswc.sh|download_mswc.py|run_full_mswc_experiment_matrix.sh|scripts/train.py"
```

Xem trạng thái process download/extract:

```bash
ps -p 2962205 -o pid,ppid,stat,etime,%cpu,%mem,wchan:24,cmd
```

Xem tail log:

```bash
tail -n 80 /storage/<user>/an_kws/logs/kws_full_mswc.log
tail -n 40 /storage/<user>/an_kws/logs/kws_wait_then_matrix.log
```

Kiểm tra archive:

```bash
ls -lh /storage/<user>/an_kws/DoAnTotNghiep/data/mswc_en/en.tar.gz*
```

Không khuyến nghị dùng `du -sh clips` trong lúc extract.

## 11. Việc Cần Làm Tiếp Theo

### 11.1. Ngắn hạn

1. Theo dõi `kws_full_mswc`.
2. Đợi extract full MSWC xong.
3. Đợi convert OPUS -> WAV xong.
4. Đảm bảo setup log có dòng:

```text
Finished full MSWC data setup; fixed training skipped. Experiment matrix should run separately.
```

5. `kws_full_matrix_wait` sẽ detect setup xong và chạy matrix.

### 11.2. Nếu Extract Lại Lỗi

Do `data/download_mswc.py` đã sửa, nếu lỗi sẽ fail thật.

Cần kiểm tra:

```bash
tail -n 120 /storage/<user>/an_kws/logs/kws_full_mswc.log
```

Nếu lỗi NFS `Transport endpoint is not connected` tái diễn:

- Không train.
- Giữ archive `en.tar.gz`.
- Restart lại setup sau khi NFS ổn:

```bash
tmux kill-session -t kws_full_mswc 2>/dev/null || true
tmux new-session -d -s kws_full_mswc "env MSWC_MIRROR=alibaba RUN_FIXED_FULL_MSWC_TRAIN=0 bash /storage/<user>/an_kws/server_setup_full_mswc.sh"
```

### 11.3. Khi Matrix Bắt Đầu

Theo dõi:

```bash
tail -f /storage/<user>/an_kws/logs/full_mswc_phase1_matrix.log
tail -f /storage/<user>/an_kws/logs/full_mswc_phase1_matrix_runs.tsv
```

Mục tiêu phase 1 nên là chạy ngắn/smoke trước nếu script có cấu hình nhỏ, vì full 16 tổ hợp có thể rất tốn GPU.

## 12. Những Điểm Cần Nhớ Khi Viết Thesis/Email

Không overclaim.

Nên viết:

- Em đọc thêm EdgeSpot, SCAF, GE2E-KWS.
- Em thấy ba thành phần EdgeSpotFull T4, SCAF, GE2E phù hợp với bài toán.
- Em kiểm chứng trên MSWC Microset với 3 cấu hình chính.
- Microset cho thấy EdgeSpotFull T4 + PCEN + SCAF+GE2E tốt nhất ở phần lớn chỉ số.
- Từ đó em mở rộng sang Top500.
- Top500 epoch25 từng đạt kết quả tốt trong log nhưng checkpoint bị mất do Colab/session.
- Hiện artifact chắc chắn là Top500 epoch13.
- Em đang chuyển sang server `ict6` để tránh mất session Colab và mở rộng thử nghiệm.

Không nên viết:

- “Top500 epoch25 là final checkpoint” nếu không có artifact.
- “Open-set UI thay thế test100”.
- “Reproduce đầy đủ EdgeSpot paper”.

## 13. File Handoff Này Được Tạo Ở Đâu

Local path:

```text
D:\Downloads\DoAnTotNghiep\docs\session_handoff_2026_05_29.md
```

Có thể gửi nguyên file này cho AI khác.

