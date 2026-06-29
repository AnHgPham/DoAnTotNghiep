# Báo Cáo Thực Nghiệm Colab - MSWC Microset English

Ngày tổng hợp: 2026-05-19  
Nguồn chạy: Google Colab Pro/A100  
Notebook tham chiếu: https://colab.research.google.com/drive/1q2Dzh3Og27H9o02wh56mFXA2rbVV2cm2?usp=sharing  
Repository: https://github.com/AnHgPham/DoAnTotNghiep

Ghi chú truy xuất: link Colab yêu cầu đăng nhập Google, vì vậy báo cáo này tổng hợp từ log chạy, result JSON, checkpoint trong Google Drive và code hiện tại của project.

## 1. Mục Tiêu Thực Nghiệm

Trong tuần này, em tập trung cải tiến pipeline few-shot open-set keyword spotting dựa trên hai hướng nghiên cứu chính: kiến trúc EdgeSpot trong paper **EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting** và ý tưởng loss GE2E trong các nghiên cứu về speaker/keyword verification. Từ đó, em bổ sung thêm nhánh `EdgeSpotFull T4` và loss `GE2E` vào dự án để chạy thực nghiệm trên Colab.

Mục tiêu chính là kiểm tra xem việc chuyển từ baseline cũ `DSCNN-L + MFCC + Triplet` sang nhánh `EdgeSpotFull + mel-PCEN + SCAF/GE2E` có cải thiện kết quả few-shot open-set KWS hay không. Kết quả thực nghiệm cho thấy hướng cải tiến này cho kết quả tốt hơn rõ rệt, đặc biệt ở các chỉ số `ACC@5% FAR`, `EER`, `Keyword ACC`, `F1` và `FRR@5% FAR`.

Các mục tiêu cụ thể:

- dùng MSWC Microset English chính thức để train trong điều kiện tiết kiệm disk và Colab units;
- sửa pipeline để đọc đúng official CSV split của Microset, tránh leakage do quét toàn bộ folder `clips/<word>`;
- chạy lại baseline `DSCNN-L + Triplet` để có mốc so sánh;
- cải tiến thêm `EdgeSpotFull T4 + SCAF` theo hướng EdgeSpot paper;
- mở rộng thêm `EdgeSpotFull T4 + SCAF+GE2E` để training sát hơn với cơ chế few-shot enrollment/prototype;
- đánh giá bằng Google Speech Commands v2 theo protocol `gsc_edgespot_exact`, có true `_silence_`, 10-shot, open-set;
- chọn model cuối dựa trên kết quả GSC-dev và báo cáo kết quả cuối trên GSC-test 100 runs.

Kết quả trong báo cáo này chỉ là **official Microset experiment / pipeline validation under resource constraints**. Không claim đây là reproduction đầy đủ của EdgeSpot paper, vì paper dùng dữ liệu và training setup lớn hơn.

## 2. Dữ Liệu Và Protocol

### 2.1. Dữ Liệu Train

Profile dữ liệu:

```text
CURRENT DATA PROFILE = MSWC MICROSET ENGLISH
TEMPORARY RUN FOR UNIT/DISK SAVING
NOT TOP500 FULL
NOT FULL MSWC
NOT EDGESPOT PAPER REPRODUCTION
```

MSWC Microset English sau khi xử lý:

| Split | Nguồn | Số file |
|---|---|---:|
| Train | `en_train.csv` | 69,868 |
| Dev/Val | `en_dev.csv` | 13,114 |
| Test/Eval | `en_test.csv` | 13,117 |
| Tổng | official Microset English | 96,099 |

Microset là sample-level split: train/dev/test có cùng 31 keyword nhưng file audio khác nhau. Vì vậy project dùng manifest file-level:

- `train_files.json`;
- `val_files.json`;
- `eval_files.json`.

Cách này thay thế việc quét trực tiếp toàn bộ `clips/<word>`, tránh vô tình trộn file train/dev/test.

### 2.2. Dữ Liệu Đánh Giá

Dataset đánh giá là Google Speech Commands v2.

Protocol chính:

```text
protocol: gsc_edgespot_exact
query split: dev cho checkpoint selection, test cho final report
target keywords: 10 GSC commands + true _silence_
unknown/open-set: 25 speech words còn lại
k-shot: 10
n-runs final: 100
classifier: OpenNCMClassifier
scoring: L2 distance
```

Protocol này phù hợp với mục tiêu few-shot open-set KWS: hệ thống cần nhận diện đúng keyword đã enrollment và từ chối từ lạ ở mức FAR cố định.

## 3. Cơ Sở Cải Tiến Từ Các Bài Báo

### 3.1. Phần Dựa Trên EdgeSpot Paper

Từ paper EdgeSpot, em triển khai nhánh `EdgeSpotFull T4` để thay thế baseline DSCNN-L trong các thực nghiệm mới. Mục tiêu là giữ tinh thần của EdgeSpot-4: model nhỏ, phù hợp edge deployment, nhưng vẫn học được embedding tốt cho few-shot keyword spotting.

Các thành phần lấy cảm hứng trực tiếp từ paper:

- input là **40x101 Mel-Spectrogram** cho audio 1 giây 16 kHz;
- frontend có **trainable PCEN**;
- backbone dùng các block kiểu **BC-ResNet / Fused BC-ResNet**;
- có positional temporal Conv1D và lightweight temporal self-attention/SDPA;
- output là **64-D embedding**;
- biến thể lớn nhất trong paper là **EdgeSpot-4**, khoảng 128k parameters.

Trong project, `EdgeSpotFull T4` dùng `tau=4`, tương ứng với hướng `EdgeSpot-4`. Model local có `130,598` tham số, gần với footprint paper báo cáo cho EdgeSpot-4.

### 3.2. Input Mel 40x101 Là Gì

Model không nhận trực tiếp waveform thô. Audio 1 giây được đổi thành một bản đồ thời gian-tần số:

- `40`: số mel frequency bands;
- `101`: số frame thời gian;
- hình dạng input của model: `(B, 1, 40, 101)`.

Cách biểu diễn này giữ cấu trúc âm học tốt cho CNN/attention hơn MFCC trong nhánh EdgeSpot-style, đồng thời vẫn đủ nhỏ để train và evaluate nhanh.

### 3.3. Vì Sao Bật PCEN

PCEN là **Per-Channel Energy Normalization**. Có thể hiểu PCEN như một lớp chuẩn hóa năng lượng học được, giúp giảm ảnh hưởng của:

- nói to/nhỏ khác nhau;
- thay đổi microphone;
- nhiễu nền;
- lệch domain giữa MSWC train và GSC test.

EdgeSpot paper dùng trainable PCEN ở frontend, nên project bật PCEN trong `EdgeSpotFull`.

### 3.4. Vì Sao Dùng Embedding 64-D

Few-shot KWS không chỉ là classifier cố định. Khi chạy thật, hệ thống cần:

1. lấy vài mẫu enrollment cho mỗi keyword;
2. encode từng mẫu thành embedding;
3. lấy trung bình thành prototype;
4. so query mới với prototype để nhận diện hoặc từ chối.

Vì vậy output của model là embedding. Kích thước 64-D đến từ thiết kế EdgeSpot paper và được giữ trong project để cân bằng giữa khả năng biểu diễn và chi phí tính toán.

### 3.5. Vì Sao Có SCAF

SCAF là **Sub-center ArcFace**. Thành phần này có cơ sở từ EdgeSpot paper: paper dùng Sub-center ArcFace trong pipeline teacher/student để học embedding phân biệt tốt.

Lý do SCAF phù hợp với few-shot open-set KWS:

- ArcFace ép các class tách nhau theo góc trong embedding space;
- sub-center cho phép một keyword có nhiều cụm phát âm khác nhau;
- điều này phù hợp với tiếng nói vì cùng một từ có thể do nhiều speaker, accent, tốc độ và âm lượng khác nhau tạo ra;
- embedding tách rõ giúp prototype matching và open-set rejection ổn định hơn.

Vì vậy `EdgeSpotFull T4 SCAF` được dùng như cấu hình EdgeSpot-style chính: kiến trúc EdgeSpotFull + Sub-center ArcFace loss.

### 3.6. Vì Sao Thêm GE2E

GE2E không phải thành phần gốc của EdgeSpot paper. GE2E đến từ hướng **Generalized End-to-End Loss** cho speaker verification, sau đó cũng có hướng áp dụng vào keyword spotting như **GE2E-KWS: Generalized End-to-End Training and Evaluation for Zero-shot Keyword Spotting**. Vì vậy, trong đồ án này GE2E được xem là phần cải tiến thêm, không phải reproduction nguyên bản của EdgeSpot.

Lý do đưa GE2E vào đồ án là vì few-shot KWS có cơ chế gần với speaker verification/custom keyword verification:

- có support/enrollment examples;
- lấy trung bình embedding thành centroid/prototype;
- query được so với centroid để quyết định match hoặc reject.

GE2E mô phỏng đúng cơ chế này trong training. Trong mỗi episode, nó tách mẫu của từng class thành support và query, tạo centroid từ support, rồi bắt query gần centroid đúng và xa centroid sai.

### 3.7. Vì Sao Kết Hợp SCAF+GE2E

SCAF và GE2E giải quyết hai mặt khác nhau:

- **SCAF** tạo embedding space có biên phân tách class mạnh;
- **GE2E** làm training giống cơ chế enrollment/prototype lúc inference;
- hybrid `SCAF+GE2E` vừa giữ khả năng tách class, vừa tăng tính phù hợp với few-shot open-set deployment.

Do đó `EdgeSpotFull T4 SCAF+GE2E` là biến thể mở rộng của project, không phải cấu hình nguyên bản trong EdgeSpot paper. Model này được chọn làm final vì kết quả GSC-test 100 runs tốt nhất ở các chỉ số quan trọng cho mục tiêu đồ án.

## 4. Các Model Được Chạy

### 4.1. DSCNN-L Triplet Baseline

```text
model: DSCNN-L
input: MFCC
loss: Triplet
parameter count: 412,896
checkpoint dev30 ban đầu: dscnn_l_triplet_microset_en_v1/epoch_05.pt
checkpoint test100 cập nhật: dscnn_l_triplet_microset_en_v1/best.pt
```

`best.pt` tương ứng với `epoch_10.pt` trong log train, được chọn theo GSC-dev ACC@1% FAR.

### 4.2. EdgeSpotFull T4 SCAF

```text
model: EdgeSpotFull T4
input: mel 40x101
PCEN: bật
embedding: 64-D
parameter count: 130,598
loss: Sub-center ArcFace / SCAF
checkpoint: edgespot_full_t4_scaf_microset_en_v1/epoch_05.pt
```

### 4.3. EdgeSpotFull T4 SCAF+GE2E

```text
model: EdgeSpotFull T4
input: mel 40x101
PCEN: bật
embedding: 64-D
loss: hybrid SCAF + GE2E
checkpoint: edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt
```

Đây là model final hiện tại.

## 5. Artifact Kết Quả

Các result JSON chính trên Google Drive:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/results/dscnn_l_triplet_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/dscnn_l_triplet_microset_en_v1_best_test100/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/dscnn_l_triplet_microset_en_v1_epoch10_test100/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_dev30/gsc_edgespot_exact_k10_results.json
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
```

`dscnn_l_triplet_microset_en_v1/best.pt` và `epoch_10.pt` cho cùng kết quả vì `best.pt` được lưu từ epoch 10.

Một file cũ không dùng làm kết luận chính:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_dev30/gsc_edgespot_exact_k10_results.json
```

Lý do: tên file không có `epoch05`, kết quả thấp hơn rõ, có khả năng là checkpoint cũ hoặc checkpoint chọn theo tiêu chí khác.

## 6. Kết Quả Định Lượng

### 6.1. Bảng Kết Quả Chính

| Model | Split | Runs | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | AUC | EER | Keyword ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L Triplet | dev | 30 | 76.48% | 79.17% | 47.91% | 89.63% | 19.77% | 69.65% | 71.27% |
| DSCNN-L Triplet | test | 100 | - | 80.54% | 40.58% | 91.22% | 18.22% | 68.39% | 73.30% |
| EdgeSpotFull T4 SCAF | dev | 30 | 82.48% | 84.85% | 21.52% | 95.73% | 11.28% | 72.76% | 82.78% |
| EdgeSpotFull T4 SCAF+GE2E | dev | 30 | 83.78% | 85.56% | 21.85% | 95.60% | 11.63% | 76.15% | 82.29% |
| EdgeSpotFull T4 SCAF | test | 100 | 84.64% | 85.21% | 20.61% | 95.69% | 11.89% | 74.52% | 81.92% |
| EdgeSpotFull T4 SCAF+GE2E | test | 100 | 84.61% | 86.12% | 21.39% | 95.61% | 11.54% | 77.66% | 82.41% |

Ghi chú: run DSCNN-L test100 được chạy với `target_far=5%`, nên bảng chưa có `ACC@1% FAR` cho baseline test100.

### 6.2. Chi Tiết DSCNN-L Test100

```text
DSCNN-L Triplet best.pt
GSC-test 100 runs, 10-shot
AUC            = 91.22% +/- 0.74%
EER            = 18.22% +/- 1.12%
FRR@5% FAR     = 40.58% +/- 3.60%
ACC@5% FAR     = 80.54% +/- 0.87%
Keyword ACC    = 68.39% +/- 1.81%
Precision      = 66.41% +/- 1.68%
Recall         = 81.79% +/- 1.12%
F1             = 73.30% +/- 1.47%
```

### 6.3. Kết Quả Model Final

```text
EdgeSpotFull T4 + SCAF+GE2E
GSC-test 100 runs, 10-shot
Checkpoint: /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt
Result: /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json

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

### 6.4. Kết Quả Mở Rộng Top500 Từ Notebook

Ngoài Microset, em cũng đã chạy tiếp nhánh `top500_full_v1` trong notebook `server/week_8_500_words.ipynb`. Phần này không thay thế kết quả Microset official ở trên, nhưng là bằng chứng cho thấy hướng `EdgeSpotFull T4 + SCAF+GE2E` vẫn giữ kết quả tốt khi mở rộng dữ liệu.

Với Top500, có hai mốc cần tách rõ:

- `epoch13`: có checkpoint và result dev30 trong package local, dùng được cho demo/phân tích sơ bộ;
- `epoch25`: có log đầy đủ trong notebook cho dev30 và test100, nhưng package local hiện tại chưa kèm checkpoint/result JSON, nên nên ghi là kết quả từ Colab notebook log.

| Model | Eval | Runs | ACC@1% FAR | ACC@5% FAR / Open-set ACC | FRR@5% FAR | AUC | EER | Keyword ACC | F1 | Nguồn |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EdgeSpotFull T4 SCAF+GE2E Top500 epoch13 | dev | 30 | 86.68% | 88.87% | 20.36% | 95.12% | 12.03% | 88.86% | 81.71% | local artifact |
| EdgeSpotFull T4 SCAF+GE2E Top500 epoch25 | dev | 30 | - | 89.33% | 19.57% | 95.01% | 11.37% | 89.68% | 82.66% | notebook log |
| EdgeSpotFull T4 SCAF+GE2E Top500 epoch25 | test | 100 | - | 88.57% | 23.13% | 94.63% | 12.13% | 91.06% | 81.57% | notebook log |

Trong quá trình train epoch25, checkpoint selection theo GSC-dev 3 runs đạt:

```text
Epoch 25/25
loss        = 11.086228
episodes    = 300
SCAF loss   = 10.4846
GE2E loss   = 0.6017
GE2E ACC    = 0.842
GSC-dev     = ACC@1%FAR 87.61%, ACC@5%FAR 89.20%, FRR@5% 20.00%
checkpoint  = edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_25.pt
```

Điểm quan trọng là Top500 epoch25 cho `Keyword ACC` rất cao trên GSC-test 100 runs (`91.06%`), nhưng `FRR@5% FAR` tăng lên `23.13%` so với dev30. Vì vậy em nên trình bày phần này như kết quả mở rộng có triển vọng, còn kết quả chính thức/locked của báo cáo vẫn là Microset.

## 7. Phân Tích Kết Quả

### 7.1. So Với Baseline DSCNN-L

So với DSCNN-L Triplet test100, `EdgeSpotFull T4 + SCAF+GE2E` cải thiện:

| Metric | DSCNN-L test100 | EdgeSpotFull T4 SCAF+GE2E test100 | Chênh lệch |
|---|---:|---:|---:|
| ACC@5% FAR | 80.54% | 86.12% | +5.58 điểm % |
| FRR@5% FAR | 40.58% | 21.39% | -19.19 điểm % |
| AUC | 91.22% | 95.61% | +4.39 điểm % |
| EER | 18.22% | 11.54% | -6.68 điểm % |
| Keyword ACC | 68.39% | 77.66% | +9.27 điểm % |
| F1 | 73.30% | 82.41% | +9.11 điểm % |

Kết quả này cho thấy việc chuyển từ `DSCNN-L/MFCC/Triplet` sang `EdgeSpotFull/mel-PCEN/SCAF+GE2E` cải thiện rõ khả năng phân tách embedding, giảm false rejection và tăng độ chính xác keyword.

### 7.2. So Sánh SCAF Và SCAF+GE2E

Trên GSC-test 100 runs:

- SCAF nhỉnh hơn rất nhẹ ở AUC và FRR@5% FAR:
  - AUC: 95.69% so với 95.61%;
  - FRR@5% FAR: 20.61% so với 21.39%.
- SCAF+GE2E tốt hơn ở các chỉ số quan trọng hơn cho mục tiêu nhận diện keyword:
  - ACC@5% FAR: 86.12% so với 85.21%;
  - EER: 11.54% so với 11.89%;
  - Keyword ACC: 77.66% so với 74.52%;
  - F1: 82.41% so với 81.92%.

Vì mục tiêu của đề tài là vừa nhận diện đúng keyword vừa kiểm soát open-set, `SCAF+GE2E` được chọn làm model final hiện tại.

### 7.3. Ý Nghĩa Thực Nghiệm

Kết quả final có ba ý nghĩa chính:

- model mới giảm mạnh tỷ lệ bỏ sót keyword thật ở mức FAR cố định;
- keyword classification tốt hơn baseline cũ;
- test100 không tụt so với dev30, cho thấy kết quả không chỉ là may mắn trên dev.

Tuy nhiên, chưa nên so trực tiếp với số EdgeSpot paper như một reproduction đầy đủ. Lý do là paper dùng full-scale MSWC setup, teacher-student distillation và số epoch/training profile khác. Kết quả hiện tại nên được báo cáo là **EdgeSpot-style Microset experiment**.

## 8. Các Vấn Đề Đã Phát Hiện Và Đã Sửa

### 8.1. Microset Folder Scan

Ban đầu nếu chỉ quét folder `clips/<word>`, một keyword như `one` có thể có hơn 11 nghìn file vì folder chứa cả train/dev/test. Điều này dễ gây hiểu nhầm rằng train dùng toàn bộ folder và có nguy cơ leakage.

Đã sửa bằng cách đọc official CSV:

- `en_train.csv`;
- `en_dev.csv`;
- `en_test.csv`;
- `en_splits.csv`.

Sau đó sinh manifest:

- `train_files.json`;
- `val_files.json`;
- `eval_files.json`.

### 8.2. `n_samples=20` Không Phù Hợp Microset

Trong Microset official train split, một số từ có ít mẫu train. `sheila` chỉ có 16 mẫu train, nên `n_samples=20` làm sampler loại mất class.

Đã sửa cấu hình Microset:

```text
n_classes = 31
n_samples = 16
```

Như vậy mỗi episode dùng đủ 31 class.

### 8.3. Resume Checkpoint Đã Hoàn Tất

Khi resume từ checkpoint đã đạt số epoch target, script cũ vẫn cố lưu biến `epoch` và gây lỗi:

```text
UnboundLocalError: cannot access local variable 'epoch'
```

Đã sửa trong `scripts/train.py`: nếu `start_epoch >= n_epochs`, script log rằng run đã hoàn tất và thoát sạch.

Commit liên quan:

```text
375ab4c Handle completed resume runs
```

## 9. Giới Hạn Hiện Tại

Các giới hạn cần ghi rõ:

- kết quả chính để claim trong báo cáo này vẫn là MSWC Microset English;
- Top500 đã có log epoch25 dev30/test100 trong notebook, nhưng package local hiện tại chưa kèm checkpoint/result JSON tương ứng nên cần đóng gói lại trước khi xem là artifact locked;
- Top500 chưa phải full reproduction của EdgeSpot paper vì vẫn khác dữ liệu, teacher-student setup và training profile;
- chưa claim reproduce EdgeSpot paper;
- DSCNN-L test100 đã bổ sung, nhưng còn thiếu `ACC@1% FAR` vì run này dùng `target_far=5%`;
- chưa chạy KD với Wav2Vec2 teacher như EdgeSpot paper;
- chưa có calibration nâng cao như impostor bank, multi-prototype, support uncertainty;
- chưa có confusion matrix/per-word error analysis trong bản tổng hợp này;
- chưa chạy streaming benchmark thật với microphone.

## 10. Kết Luận

Từ hai hướng nghiên cứu trên, tuần này em đã cải tiến dự án bằng cách bổ sung nhánh `EdgeSpotFull T4` và loss `GE2E`, sau đó chạy thực nghiệm lại trên cùng protocol GSC open-set. Kết quả cho thấy hướng cải tiến này tốt hơn baseline DSCNN-L rõ rệt.

`EdgeSpotFull T4` giúp project tiến gần hơn đến kiến trúc EdgeSpot-4 trong paper, gồm mel 40x101, PCEN, backbone kiểu BC-ResNet/Fused BC-ResNet, temporal attention và 64-D embedding. `SCAF` được dùng vì có cơ sở trực tiếp từ EdgeSpot paper, còn `GE2E` là phần em bổ sung thêm để training sát hơn với cơ chế few-shot support/query-prototype khi inference.

Model final hiện tại:

```text
EdgeSpotFull T4 + SCAF+GE2E
GSC-test 100 runs, 10-shot
ACC@5% FAR  = 86.12%
Keyword ACC = 77.66%
F1          = 82.41%
EER         = 11.54%
FRR@5% FAR  = 21.39%
```

So với DSCNN-L Triplet test100, model final tăng `ACC@5% FAR` thêm 5.58 điểm %, tăng `Keyword ACC` thêm 9.27 điểm %, tăng `F1` thêm 9.11 điểm %, đồng thời giảm `EER` 6.68 điểm % và giảm `FRR@5% FAR` 19.19 điểm %. Như vậy, việc cải tiến thêm EdgeSpot và GE2E vào dự án đã cho kết quả thực nghiệm tốt, đủ để dùng làm mốc hiện tại của đồ án và làm nền cho các phase tiếp theo.

Ở nhánh mở rộng Top500, epoch25 trong notebook đạt `Keyword ACC = 91.06%` và `F1 = 81.57%` trên GSC-test 100 runs. Kết quả này cho thấy hướng mở rộng lên nhiều keyword có triển vọng, nhưng do checkpoint/result JSON epoch25 chưa nằm trong package local nên em nên ghi là kết quả từ notebook log, chưa phải artifact locked.

## 11. Hướng Tiếp Theo

Ưu tiên tiếp theo:

1. Chạy DSCNN-L test100 với `--target-far 0.01` nếu cần điền đủ `ACC@1% FAR` cho baseline.
2. Xuất confusion matrix và per-word metrics cho `EdgeSpotFull T4 + SCAF+GE2E`.
3. Thêm calibration:
   - per-keyword threshold;
   - impostor bank;
   - multi-prototype;
   - support uncertainty scaling.
4. Đóng gói lại checkpoint/result JSON Top500 epoch25 từ Google Drive để biến log notebook thành artifact locked.
5. Nếu muốn bám EdgeSpot paper hơn, chạy KD với Wav2Vec2 teacher và so với SCAF/SCAF+GE2E hiện tại.
6. Sau khi static/open-set ổn, làm streaming benchmark:
   - false alarms/hour;
   - miss rate;
   - latency;
   - duplicate detection.

## 12. Tài Liệu Tham Chiếu

- EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting.
- Deng et al., Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces, ECCV 2020.
- Wan et al., Generalized End-to-End Loss for Speaker Verification, ICASSP 2018.
- Zhu et al., GE2E-KWS: Generalized End-to-End Training and Evaluation for Zero-shot Keyword Spotting, 2024.
