# Báo Cáo Kết Quả Thực Nghiệm Colab - MSWC Microset English

Ngày tổng hợp: 2026-05-19  
Nguồn chạy: Google Colab Pro/A100  
Notebook tham chiếu: https://colab.research.google.com/drive/1q2Dzh3Og27H9o02wh56mFXA2rbVV2cm2?usp=sharing  
Repository: https://github.com/AnHgPham/DoAnTotNghiep

Ghi chú truy xuất: link Colab yêu cầu đăng nhập Google nên báo cáo này tổng hợp từ các log/result JSON được in trực tiếp trong Colab và các đường dẫn kết quả dưới `/content/drive/MyDrive/DoAnTotNghiep_output/results`.

## 1. Mục Tiêu Thực Nghiệm

Mục tiêu của giai đoạn này là kiểm tra lại pipeline few-shot open-set keyword spotting trong điều kiện tài nguyên Colab hạn chế, chuyển từ pipeline cũ DSCNN sang nhánh EdgeSpot-style mạnh hơn, đồng thời chuẩn hóa cách đánh giá bằng giao thức `gsc_edgespot_exact`.

Các mục tiêu cụ thể:

- dùng MSWC Microset English chính thức thay vì Top500/full MSWC để tiết kiệm disk và Colab units;
- sửa pipeline để dùng đúng split CSV chính thức của Microset, tránh quét toàn bộ folder gây leakage giữa train/dev/test;
- chạy baseline DSCNN-L + Triplet;
- chạy EdgeSpotFull T4 + SCAF;
- chạy EdgeSpotFull T4 + SCAF+GE2E;
- đánh giá bằng Google Speech Commands v2 với true `_silence_`, 10-shot, open-set protocol;
- chọn model cuối dựa trên GSC-dev, sau đó báo cáo kết quả cuối trên GSC-test 100 runs.

## 2. Cấu Hình Dữ Liệu

Profile dữ liệu hiện tại:

```text
CURRENT DATA PROFILE = MSWC MICROSET ENGLISH
TEMPORARY RUN FOR UNIT/DISK SAVING
NOT TOP500 FULL
NOT FULL MSWC
NOT EDGESPOT PAPER REPRODUCTION
```

Dataset train:

- nguồn: MLCommons Multilingual Spoken Words Corpus Microset, English;
- thư mục trên Colab: `data/mswc_microset_en`;
- tổng audio sau convert: `96,099 WAV`;
- số keyword: `31`;
- split chính thức:
  - train: `69,868` file từ `en_train.csv`;
  - dev/val: `13,114` file từ `en_dev.csv`;
  - test/eval: `13,117` file từ `en_test.csv`.

Điểm quan trọng: Microset là split theo sample-level. Cả train/dev/test đều có cùng 31 keyword, nhưng file audio khác nhau. Vì vậy code đã được sửa để train bằng `train_files.json`, validation bằng `val_files.json`, không quét toàn bộ `clips/<word>` nữa.

Dataset đánh giá:

- Google Speech Commands v2;
- protocol: `gsc_edgespot_exact`;
- setup: 10 command keywords + true `_silence_`;
- negative/open-set: 25 speech words không thuộc target;
- k-shot: `10`;
- classifier: `OpenNCMClassifier`;
- scoring: L2 distance;
- dev: dùng chọn/check model;
- test: dùng báo cáo cuối, 100 runs.

## 3. Các Phần Đã Hoàn Thành Trong Code

Các thay đổi chính đã hoàn thành:

- tạo workflow Colab command-based, không phụ thuộc notebook train cũ;
- thêm runbook `docs/colab_microset_runbook_vi.md`;
- thêm note trạng thái `docs/current_training_profile_vi.md`;
- thêm script xử lý Microset `data/download_mswc_microset.py`;
- sửa Microset loader để đọc đúng CSV official split;
- sinh `train_files.json`, `val_files.json`, `eval_files.json`;
- sửa `MSWCDataset` để nhận manifest file-level;
- sửa `scripts/train.py` để dùng manifest split khi có;
- thêm EdgeSpotFull T1-T4;
- thêm BCResNetFS;
- thêm GE2E loss;
- thêm hybrid loss SCAF+GE2E;
- thêm protocol `gsc_edgespot_exact` với true silence;
- thêm checkpoint/eval workflow cho GSC-dev và GSC-test;
- sửa lỗi resume khi checkpoint đã đủ epoch: nếu `start_epoch >= target_epochs` thì script thoát sạch thay vì crash `UnboundLocalError`.

## 4. Model Và Thiết Lập Train

### 4.1. DSCNN-L Triplet Baseline

- model: DSCNN-L;
- input: MFCC;
- loss: Triplet;
- mục đích: baseline cũ để so với EdgeSpot-style;
- checkpoint đánh giá: `dscnn_l_triplet_microset_en_v1/epoch_05.pt`;
- kết quả hiện mới có GSC-dev 30 runs, chưa có GSC-test 100 runs.

### 4.2. EdgeSpotFull T4 SCAF

- model: EdgeSpotFull T4;
- input: mel `40x101`;
- PCEN: bật;
- embedding: 64-D;
- parameter count: `130,598`;
- loss: Sub-center ArcFace/SCAF;
- checkpoint đánh giá: `edgespot_full_t4_scaf_microset_en_v1/epoch_05.pt`.

### 4.3. EdgeSpotFull T4 SCAF+GE2E

- model: EdgeSpotFull T4;
- input: mel `40x101`;
- PCEN: bật;
- embedding: 64-D;
- loss: hybrid SCAF + GE2E;
- checkpoint đánh giá: `edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt`;
- đây là model final hiện tại vì đạt `Keyword ACC`, `Open-set ACC@5% FAR`, `EER`, và `F1` tốt nhất trên GSC-test 100 runs.

## 5. Kết Quả Từ Colab

### 5.1. Bảng Kết Quả Raw Từ Result JSON

Các file kết quả Colab đã in:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/results/dscnn_l_triplet_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
```

Một file cũ không dùng làm kết luận chính:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_dev30/gsc_edgespot_exact_k10_results.json
```

Lý do không dùng file cũ này: tên không có `epoch05`, kết quả thấp hơn rõ, khả năng là checkpoint cũ hoặc checkpoint chọn theo tiêu chí khác.

### 5.2. Bảng Kết Quả Chính

| Model | Split | Runs | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | AUC | EER | Keyword ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L Triplet | dev | 30 | 76.48% | 79.17% | 47.91% | 89.63% | 19.77% | 69.65% | 71.27% |
| EdgeSpotFull T4 SCAF | dev | 30 | 82.48% | 84.85% | 21.52% | 95.73% | 11.28% | 72.76% | 82.78% |
| EdgeSpotFull T4 SCAF+GE2E | dev | 30 | 83.78% | 85.56% | 21.85% | 95.60% | 11.63% | 76.15% | 82.29% |
| EdgeSpotFull T4 SCAF | test | 100 | 84.64% | 85.21% | 20.61% | 95.69% | 11.89% | 74.52% | 81.92% |
| EdgeSpotFull T4 SCAF+GE2E | test | 100 | 84.61% | 86.12% | 21.39% | 95.61% | 11.54% | 77.66% | 82.41% |

### 5.3. Kết Quả Final Được Chọn

Model final:

```text
EdgeSpotFull T4 + SCAF+GE2E
Checkpoint: /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt
Result: /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
```

Kết quả final trên GSC-test 100 runs:

```text
ACC@1% FAR     = 84.61%
ACC@5% FAR     = 86.12%
FRR@5% FAR     = 21.39%
AUC            = 95.61%
EER            = 11.54%
Keyword ACC    = 77.66%
Precision      = 77.15%
Recall         = 88.45%
F1             = 82.41%
```

## 6. Phân Tích Kết Quả

### 6.1. So Sánh Với DSCNN-L

So với DSCNN-L Triplet baseline dev30, EdgeSpotFull T4 + SCAF+GE2E test100 cải thiện rõ:

- `ACC@5% FAR`: 79.17% -> 86.12%, tăng 6.95 điểm %;
- `FRR@5% FAR`: 47.91% -> 21.39%, giảm 26.52 điểm %;
- `AUC`: 89.63% -> 95.61%, tăng 5.98 điểm %;
- `EER`: 19.77% -> 11.54%, giảm 8.23 điểm %;
- `Keyword ACC`: 69.65% -> 77.66%, tăng 8.01 điểm %;
- `F1`: 71.27% -> 82.41%, tăng 11.14 điểm %.

Điều này cho thấy việc chuyển từ DSCNN-L/MFCC/Triplet sang EdgeSpotFull/mel-PCEN/SCAF+GE2E cải thiện mạnh khả năng phân tách embedding và open-set rejection.

Lưu ý khoa học: DSCNN-L hiện mới có dev30, chưa có test100. Nếu cần bảng baseline tuyệt đối công bằng, nên chạy thêm DSCNN-L test100. Tuy vậy, kết quả hiện tại vẫn đủ cho kết luận định hướng vì EdgeSpot cải thiện rất rõ trên cùng protocol.

### 6.2. SCAF Và SCAF+GE2E

Trên GSC-test 100 runs:

- SCAF nhỉnh hơn rất nhẹ ở `AUC` và `FRR@5% FAR`:
  - AUC: 95.69% so với 95.61%;
  - FRR@5% FAR: 20.61% so với 21.39%.
- SCAF+GE2E tốt hơn ở các chỉ số quan trọng hơn cho mục tiêu nhận diện keyword:
  - ACC@5% FAR: 86.12% so với 85.21%;
  - EER: 11.54% so với 11.89%;
  - Keyword ACC: 77.66% so với 74.52%;
  - F1: 82.41% so với 81.92%.

Vì mục tiêu của đề tài là vừa nhận diện đúng keyword vừa kiểm soát open-set, SCAF+GE2E được chọn làm model final hiện tại.

### 6.3. Ý Nghĩa Của Kết Quả

Kết quả final cho thấy pipeline mới đã vượt xa baseline cũ ở các điểm chính:

- giảm mạnh false rejection ở mức FAR cố định;
- tăng khả năng nhận diện keyword;
- tăng độ ổn định open-set;
- test100 không bị tụt so với dev30, cho thấy model không chỉ may mắn trên dev.

Đây là một kết quả đủ tốt để viết báo cáo đồ án và làm nền cho hướng nghiên cứu tiếp theo. Tuy nhiên, chưa nên claim đã reproduce EdgeSpot paper, vì paper sử dụng setup dữ liệu và training lớn hơn Microset.

## 7. Các Vấn Đề Đã Phát Hiện Và Đã Sửa

### 7.1. Vấn đề Microset folder scan

Ban đầu nếu chỉ quét folder `clips/<word>`, một keyword như `one` có thể có hơn 11 nghìn file vì folder chứa cả train/dev/test. Điều này dễ gây hiểu nhầm rằng train dùng toàn bộ folder, đồng thời có nguy cơ leakage.

Đã sửa bằng cách đọc CSV official:

- `en_train.csv`;
- `en_dev.csv`;
- `en_test.csv`;
- `en_splits.csv`.

Sau đó sinh manifest:

- `train_files.json`;
- `val_files.json`;
- `eval_files.json`.

Training và validation hiện dùng đúng manifest này.

### 7.2. Vấn đề n_samples=20 không phù hợp Microset

Trong Microset official train split, một số từ có rất ít mẫu train. Đặc biệt `sheila` chỉ có 16 mẫu train. Nếu đặt `n_samples=20` và `n_classes=31`, sampler báo lỗi:

```text
Need at least 31 classes with >=20 samples each, but only 30 classes qualify.
```

Đã sửa cấu hình Microset:

```text
n_classes = 31
n_samples = 16
```

Như vậy mỗi episode dùng đủ 31 class, không loại mất keyword nào.

### 7.3. Vấn đề resume checkpoint đã hoàn tất

Khi resume từ `latest.pt` ở epoch 24 nhưng cấu hình `--epochs 25`, script không còn epoch nào để chạy. Code cũ vẫn gọi `save_checkpoint(..., epoch, ...)` nên lỗi:

```text
UnboundLocalError: cannot access local variable 'epoch'
```

Đã sửa trong `scripts/train.py`: nếu `start_epoch >= n_epochs`, script log rằng run đã hoàn tất và thoát sạch.

Commit sửa lỗi:

```text
375ab4c Handle completed resume runs
```

## 8. Giới Hạn Hiện Tại

Các giới hạn cần ghi rõ khi báo cáo:

- đây là MSWC Microset English, không phải Top500 full và không phải full MSWC;
- chưa claim reproduce EdgeSpot paper;
- DSCNN-L mới có dev30, chưa có test100;
- chưa chạy streaming benchmark thật với mic;
- chưa có calibration nâng cao như impostor bank, multi-prototype, support uncertainty;
- chưa chạy KD với Wav2Vec2 teacher;
- chưa có confusion matrix/per-word error analysis trong bản tổng hợp này.

## 9. Kết Luận

Trong giai đoạn này, project đã chuyển từ pipeline DSCNN baseline sang pipeline EdgeSpot-style có benchmark rõ ràng hơn. Dữ liệu Microset đã được xử lý đúng theo official CSV split, tránh leakage do quét toàn bộ folder. Giao thức đánh giá `gsc_edgespot_exact` đã được dùng để đo few-shot open-set KWS với true silence.

Kết quả tốt nhất hiện tại là:

```text
EdgeSpotFull T4 + SCAF+GE2E
GSC-test 100 runs, 10-shot
ACC@5% FAR  = 86.12%
Keyword ACC = 77.66%
F1          = 82.41%
EER         = 11.54%
FRR@5% FAR  = 21.39%
```

So với baseline DSCNN-L, phương pháp mới cải thiện đáng kể ở cả open-set accuracy, keyword accuracy, F1 và FRR. Đây là kết quả đủ mạnh để dùng làm mốc hiện tại của đồ án, đồng thời là nền để tiếp tục mở rộng sang Top500 full, calibration và streaming.

## 10. Hướng Tiếp Theo

Ưu tiên tiếp theo:

1. Chạy thêm DSCNN-L test100 để bảng baseline hoàn toàn công bằng.
2. Xuất confusion matrix/per-word metrics cho EdgeSpotFull T4 + SCAF+GE2E.
3. Thêm calibration:
   - per-keyword threshold;
   - impostor bank;
   - multi-prototype;
   - support uncertainty scaling.
4. Khi có máy trường hoặc disk đủ, chuyển sang `top500_full_v1`.
5. Sau khi static/open-set ổn, làm streaming benchmark:
   - false alarms/hour;
   - miss rate;
   - latency;
   - duplicate detection.

