# Chương thực nghiệm mở rộng: MSWC cap620 FLAC, 16 pipeline cố định và so sánh EdgeSpot-4

Ngày tổng hợp: 2026-06-12

Nguồn số liệu chính:

- `results/cap620_16_pipeline_metrics_long.csv`
- `results/cap620_16_pipeline_test100_summary.md`
- Colab run id: `colab_mswc_cap620_flac_16pipe_e40_ep150_20260611_154517`
- Drive artifact gốc: `/content/drive/MyDrive/DoAnTotNghiep_colab_runs/colab_mswc_cap620_flac_16pipe_e40_ep150_20260611_154517`
- Script protocol: `colab/run_mswc_cap620_16_pipeline_e40_fixed.sh`

## 1. Tóm tắt kết quả chính

Thí nghiệm MSWC cap620 FLAC đã chạy đủ 16 pipeline, mỗi pipeline được huấn luyện trong cùng một điều kiện: 40 epoch, 150 episode mỗi epoch, 30 lớp mỗi episode và 10 mẫu mỗi lớp. Tất cả các pipeline đều có trạng thái `train=ok`, `dev30_far1=ok`, `test100_far1=ok`, `test100_far5=ok`. Do đó, đây là mốc ablation chuẩn nhất hiện tại để so sánh đồng thời kiến trúc, frontend và loss trong cùng một cấu hình dữ liệu lớn.

Kết quả tốt nhất trên GSC-test100 ở operating point nghiêm ngặt `FAR=1%` là:

| Pipeline | ACC@1%FAR | AUC | EER | FRR@1%FAR | Keyword ACC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSCNN-L + PCEN + GE2E | 82.34 +/- 1.19 | 92.42 +/- 0.54 | 14.89 +/- 0.84 | 54.55 +/- 4.01 | 88.81 +/- 1.10 | 77.75 +/- 1.15 |
| DSCNN-L + PCEN + Triplet | 79.98 +/- 1.11 | 90.65 +/- 0.62 | 17.57 +/- 0.93 | 62.37 +/- 3.81 | 86.10 +/- 1.63 | 74.14 +/- 1.23 |
| EdgeSpotFull T4 + PCEN + GE2E | 79.98 +/- 0.98 | 87.23 +/- 0.75 | 20.23 +/- 0.96 | 61.26 +/- 3.39 | 83.00 +/- 1.32 | 70.68 +/- 1.23 |
| EdgeSpotFull T4 + PCEN + Triplet | 79.58 +/- 1.35 | 89.85 +/- 0.63 | 18.22 +/- 0.78 | 62.21 +/- 4.82 | 80.99 +/- 1.43 | 73.29 +/- 1.02 |

Ở operating point dễ hơn `FAR=5%`, ranking vẫn cho thấy cùng xu hướng:

| Pipeline | ACC@5%FAR | AUC | EER | FRR@5%FAR | Keyword ACC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSCNN-L + PCEN + GE2E | 86.57 +/- 0.75 | 92.42 +/- 0.54 | 14.89 +/- 0.84 | 29.18 +/- 2.60 | 88.81 +/- 1.10 | 77.75 +/- 1.15 |
| DSCNN-L + PCEN + Triplet | 84.34 +/- 0.83 | 90.65 +/- 0.62 | 17.57 +/- 0.93 | 36.09 +/- 2.80 | 86.10 +/- 1.63 | 74.14 +/- 1.23 |
| EdgeSpotFull T4 + PCEN + Triplet | 83.26 +/- 0.77 | 89.85 +/- 0.63 | 18.22 +/- 0.78 | 37.31 +/- 2.67 | 80.99 +/- 1.43 | 73.29 +/- 1.02 |
| EdgeSpotFull T4 + PCEN + GE2E | 83.16 +/- 0.82 | 87.23 +/- 0.75 | 20.23 +/- 0.96 | 39.09 +/- 2.80 | 83.00 +/- 1.32 | 70.68 +/- 1.23 |

Kết luận ngắn: `DSCNN-L + PCEN + GE2E` là cấu hình mạnh nhất nếu tối ưu mục tiêu accuracy. `EdgeSpotFull T4 + PCEN + Triplet` và `EdgeSpotFull T4 + PCEN + GE2E` là hai cấu hình compact tốt nhất, trong đó Triplet tốt hơn về AUC, EER và F1, còn GE2E nhỉnh nhẹ ở ACC@1%FAR.

## 2. Cấu hình dữ liệu huấn luyện

### 2.1. MSWC cap620 FLAC

Dữ liệu huấn luyện là MSWC English với giới hạn tối đa 620 clip mỗi từ. Dữ liệu gốc được tải ở định dạng OPUS, sau đó chuyển sang FLAC và xóa OPUS để giảm dung lượng local disk trong Colab. Profile này được gọi là `cap620 FLAC`.

Cấu hình dataset:

| Trường | Giá trị |
| --- | ---: |
| Data root | `data/mswc_en` |
| Ngôn ngữ | English |
| Tổng số từ có dữ liệu | 38,150 |
| Train words | 37,387 |
| Validation words | 763 |
| Train files | 2,989,780 |
| Validation files | 52,399 |
| Max files per word | 620 |
| Audio local format | FLAC |
| Drive sync | Chỉ sync artifacts, không sync audio |

Việc dùng FLAC thay vì WAV giúp giảm áp lực dung lượng so với full WAV. Trong log Colab, sau khi chuyển OPUS sang FLAC, `/content` dùng khoảng 135 GB trên tổng 236 GB. Đây là mức chấp nhận được cho một run, nhưng không đủ để chạy lặp nhiều run mới trong cùng runtime. Vì vậy, mỗi lần chạy lại phải giữ đúng `RUN_ID` nếu muốn resume, hoặc phải dọn local artifacts trước khi bắt đầu run mới.

### 2.2. File manifest

Pipeline không quét trực tiếp toàn bộ thư mục audio mỗi lần train. Thay vào đó, nó dùng manifest:

- `train_files_cap620_flac.json`: 2,989,780 file train.
- `val_files_cap620_flac.json`: 52,399 file validation.

Manifest có ba lợi ích chính. Thứ nhất, nó cố định chính xác tập dữ liệu dùng cho thí nghiệm, giúp tái lập kết quả. Thứ hai, nó tránh việc mỗi lần chạy lại bị thay đổi thứ tự hoặc thay đổi số file do trạng thái folder. Thứ ba, nó giảm nguy cơ trộn lẫn train/validation khi mở rộng data profile.

## 3. Cấu hình huấn luyện

Tất cả 16 pipeline dùng cùng một lịch huấn luyện:

| Trường | Giá trị |
| --- | ---: |
| Epochs | 40 |
| Episodes per epoch | 150 |
| Classes per episode | 30 |
| Samples per class | 10 |
| Num workers | 8 |
| K-shot evaluation | 10 |
| Checkpoint selection | GSC-dev ACC@1%FAR |
| Selection frequency | Mỗi 5 epoch |
| Selection runs | 3 |
| Save policy | Lưu mỗi epoch và `latest.pt` |
| Final dev evaluation | GSC-dev, 30 runs, FAR=1% |
| Final test evaluation | GSC-test, 100 runs, FAR=1% và FAR=5% |

Mỗi episode lấy 30 từ, mỗi từ 10 mẫu. Với 150 episode mỗi epoch, một epoch nhìn thấy 45,000 mẫu episodic. Lịch 40 epoch tương ứng khoảng 1.8 triệu sample occurrence theo episode, nhưng không có nghĩa là đã đi qua toàn bộ 2.99 triệu file train theo kiểu supervised epoch truyền thống. Đây là điểm quan trọng khi giải thích vì sao tăng dữ liệu từ cap220 lên cap620 không nhất thiết tăng mạnh accuracy: ngân sách episodic cố định làm số lần model thật sự quan sát dữ liệu không tăng tuyến tính theo kích thước manifest.

Checkpoint tốt nhất không được chọn theo training loss. Nó được chọn theo GSC-dev `ACC@1%FAR`, đánh giá 3 runs mỗi 5 epoch. Cách chọn này phù hợp với mục tiêu open-set KWS, vì training loss thấp chưa chắc tương ứng với threshold calibration tốt ở FAR thấp.

## 4. Cấu hình 16 pipeline

Thí nghiệm biến thiên ba trục:

- Kiến trúc: `DSCNN-L`, `EdgeSpotFull T4`.
- Frontend: `MFCC`, `PCEN`.
- Loss: `Triplet`, `SCAF`, `GE2E`, `SCAF+GE2E`.

Ma trận đầy đủ:

| # | Architecture | Frontend | Loss |
| ---: | --- | --- | --- |
| 1 | DSCNN-L | MFCC | Triplet |
| 2 | DSCNN-L | MFCC | SCAF |
| 3 | DSCNN-L | MFCC | GE2E |
| 4 | DSCNN-L | MFCC | SCAF+GE2E |
| 5 | DSCNN-L | PCEN | Triplet |
| 6 | DSCNN-L | PCEN | SCAF |
| 7 | DSCNN-L | PCEN | GE2E |
| 8 | DSCNN-L | PCEN | SCAF+GE2E |
| 9 | EdgeSpotFull T4 | MFCC | Triplet |
| 10 | EdgeSpotFull T4 | MFCC | SCAF |
| 11 | EdgeSpotFull T4 | MFCC | GE2E |
| 12 | EdgeSpotFull T4 | MFCC | SCAF+GE2E |
| 13 | EdgeSpotFull T4 | PCEN | Triplet |
| 14 | EdgeSpotFull T4 | PCEN | SCAF |
| 15 | EdgeSpotFull T4 | PCEN | GE2E |
| 16 | EdgeSpotFull T4 | PCEN | SCAF+GE2E |

## 5. Cấu hình evaluation trên GSC

GSC v2 không dùng để train các mô hình trong thí nghiệm cap620. Nó được dùng làm benchmark cross-dataset cho few-shot open-set KWS.

Protocol evaluation:

- Protocol: `gsc_edgespot_exact`.
- K-shot: 10 support samples mỗi keyword.
- Query split:
  - `dev`: dùng cho checkpoint selection và final dev30.
  - `test`: dùng cho final test100.
- Classifier: OpenNCM với L2 distance.
- Scoring: score càng cao càng có khả năng là known keyword.
- Final metrics:
  - AUC.
  - EER.
  - FRR tại target FAR.
  - Open-set ACC tại target FAR.
  - Keyword ACC.
  - Precision, recall, F1.
  - DET curve.

Hai operating point chính:

- `FAR=1%`: nghiêm ngặt, dùng làm metric chính khi so sánh với paper EdgeSpot-4 và khi cần hạn chế false accept.
- `FAR=5%`: dễ hơn, thể hiện performance nếu hệ thống cho phép nhiều false accept hơn để giảm false reject.

Test100 nghĩa là trung bình trên 100 repeated few-shot episodes. Mỗi episode có tập support/query khác nhau, do đó test100 giảm nhiễu sampling so với chỉ đánh giá một lần.

## 6. Bảng kết quả test100 đầy đủ

### 6.1. GSC-test100, FAR=1%

| Pipeline | AUC | EER | FRR@1%FAR | ACC@1%FAR | Keyword ACC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSCNN-L + MFCC + Triplet | 75.55 | 31.41 | 89.67 | 71.52 | 63.47 | 57.17 |
| DSCNN-L + MFCC + SCAF | 55.04 | 46.41 | 94.87 | 70.08 | 20.62 | 41.38 |
| DSCNN-L + MFCC + GE2E | 86.46 | 21.95 | 70.49 | 77.08 | 78.32 | 68.50 |
| DSCNN-L + MFCC + SCAF+GE2E | 47.78 | 52.15 | 98.54 | 69.04 | 13.60 | 35.93 |
| DSCNN-L + PCEN + Triplet | 90.65 | 17.57 | 62.37 | 79.98 | 86.10 | 74.14 |
| DSCNN-L + PCEN + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| DSCNN-L + PCEN + GE2E | 92.42 | 14.89 | 54.55 | 82.34 | 88.81 | 77.75 |
| DSCNN-L + PCEN + SCAF+GE2E | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + MFCC + Triplet | 52.84 | 48.05 | 96.92 | 69.63 | 15.87 | 39.79 |
| EdgeSpotFull T4 + MFCC + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + MFCC + GE2E | 65.30 | 39.05 | 92.29 | 70.76 | 42.28 | 48.82 |
| EdgeSpotFull T4 + MFCC + SCAF+GE2E | 50.88 | 50.39 | 96.10 | 69.67 | 12.89 | 37.57 |
| EdgeSpotFull T4 + PCEN + Triplet | 89.85 | 18.22 | 62.21 | 79.58 | 80.99 | 73.29 |
| EdgeSpotFull T4 + PCEN + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + PCEN + GE2E | 87.23 | 20.23 | 61.26 | 79.98 | 83.00 | 70.68 |
| EdgeSpotFull T4 + PCEN + SCAF+GE2E | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |

### 6.2. GSC-test100, FAR=5%

| Pipeline | AUC | EER | FRR@5%FAR | ACC@5%FAR | Keyword ACC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSCNN-L + MFCC + Triplet | 75.55 | 31.41 | 74.67 | 72.31 | 63.47 | 57.17 |
| DSCNN-L + MFCC + SCAF | 55.04 | 46.41 | 88.78 | 67.98 | 20.62 | 41.38 |
| DSCNN-L + MFCC + GE2E | 86.46 | 21.95 | 47.66 | 80.33 | 78.32 | 68.50 |
| DSCNN-L + MFCC + SCAF+GE2E | 47.78 | 52.15 | 94.71 | 66.54 | 13.60 | 35.93 |
| DSCNN-L + PCEN + Triplet | 90.65 | 17.57 | 36.09 | 84.34 | 86.10 | 74.14 |
| DSCNN-L + PCEN + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| DSCNN-L + PCEN + GE2E | 92.42 | 14.89 | 29.18 | 86.57 | 88.81 | 77.75 |
| DSCNN-L + PCEN + SCAF+GE2E | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + MFCC + Triplet | 52.84 | 48.05 | 90.57 | 67.62 | 15.87 | 39.79 |
| EdgeSpotFull T4 + MFCC + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + MFCC + GE2E | 65.30 | 39.05 | 82.49 | 69.84 | 42.28 | 48.82 |
| EdgeSpotFull T4 + MFCC + SCAF+GE2E | 50.88 | 50.39 | 90.70 | 67.77 | 12.89 | 37.57 |
| EdgeSpotFull T4 + PCEN + Triplet | 89.85 | 18.22 | 37.31 | 83.26 | 80.99 | 73.29 |
| EdgeSpotFull T4 + PCEN + SCAF | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |
| EdgeSpotFull T4 + PCEN + GE2E | 87.23 | 20.23 | 39.09 | 83.16 | 83.00 | 70.68 |
| EdgeSpotFull T4 + PCEN + SCAF+GE2E | 50.00 | 50.00 | 100.00 | 69.44 | 9.09 | 0.00 |

## 7. Phân tích vì sao một số tổ hợp tốt

### 7.1. PCEN là frontend ổn định nhất

PCEN cải thiện mạnh khi so với MFCC, đặc biệt trong các cấu hình có GE2E:

| So sánh trên test100@1%FAR | Delta ACC | Delta AUC | Delta F1 |
| --- | ---: | ---: | ---: |
| DSCNN-L + GE2E: PCEN so với MFCC | +5.26 | +5.96 | +9.25 |
| EdgeSpotFull T4 + GE2E: PCEN so với MFCC | +9.22 | +21.93 | +21.86 |

Lý do hợp lý là PCEN chuẩn hóa năng lượng theo kênh và thích nghi tốt hơn với khác biệt âm lượng, speaker và background. Trong few-shot KWS, support và query có thể đến từ speaker hoặc điều kiện thu khác nhau; nếu frontend quá nhạy với năng lượng tuyệt đối, distance trong embedding space dễ phản ánh điều kiện thu hơn là nội dung từ. PCEN giảm vấn đề này bằng cách làm đặc trưng ổn định hơn trước biến thiên biên độ và noise.

EdgeSpotFull T4 đặc biệt phụ thuộc vào PCEN. Khi dùng MFCC, EdgeSpotFull T4 gần như không phát huy được lợi thế của backbone. Điều này phù hợp với trực giác kiến trúc: EdgeSpot-style model được thiết kế quanh mel/PCEN frontend và xử lý cấu trúc thời gian-tần số dài hơn. MFCC nén phổ mạnh hơn và trục thời gian ngắn hơn, làm giảm lượng thông tin mà các block temporal có thể khai thác.

Hướng cải thiện: giữ PCEN làm frontend mặc định; sau đó thử so sánh PCEN trainable với PCEN fixed, tune tham số PCEN, và tăng augmentation theo âm lượng/noise để kiểm tra robustness.

### 7.2. GE2E phù hợp với prototype inference

GE2E là loss có liên hệ trực tiếp với cơ chế inference của hệ thống. Khi inference, mỗi keyword được biểu diễn bằng prototype là trung bình embedding của support samples; query được so với prototype bằng distance. GE2E huấn luyện embedding theo hướng gần centroid đúng lớp và xa centroid sai lớp, vì vậy objective huấn luyện gần với cách hệ thống thật sự dùng model.

Trên DSCNN-L, PCEN + GE2E là cấu hình mạnh nhất ở tất cả metric chính:

- ACC@1%FAR: 82.34%.
- ACC@5%FAR: 86.57%.
- AUC: 92.42%.
- EER: 14.89%.
- F1: 77.75%.

So với DSCNN-L + PCEN + Triplet, GE2E tăng:

- +2.36 điểm ACC@1%FAR.
- +2.23 điểm ACC@5%FAR.
- +1.77 điểm AUC.
- +3.61 điểm F1.

Vì DSCNN-L có dung lượng lớn hơn, mô hình có đủ capacity để tận dụng GE2E và tạo embedding space có cấu trúc centroid tốt hơn. Đây là lý do `DSCNN-L + PCEN + GE2E` trở thành cấu hình accuracy-oriented tốt nhất.

Hướng cải thiện: tăng số episode hoặc số class mỗi episode để GE2E thấy nhiều centroid đa dạng hơn; thử hard episode mining, tăng số query per class, hoặc dùng schedule lựa chọn checkpoint theo nhiều metric thay vì chỉ ACC@1%FAR.

### 7.3. Triplet vẫn rất mạnh với EdgeSpotFull T4

Với EdgeSpotFull T4, `PCEN + Triplet` có ACC@1%FAR thấp hơn `PCEN + GE2E` rất nhẹ, 79.58% so với 79.98%, nhưng lại tốt hơn về AUC, EER và F1:

| EdgeSpotFull T4 PCEN | ACC@1%FAR | ACC@5%FAR | AUC | EER | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triplet | 79.58 | 83.26 | 89.85 | 18.22 | 73.29 |
| GE2E | 79.98 | 83.16 | 87.23 | 20.23 | 70.68 |

Điều này cho thấy Triplet tạo ranking score tổng quát tốt hơn cho EdgeSpotFull T4, trong khi GE2E có thể tối ưu tốt hơn tại đúng ngưỡng FAR=1%. AUC/EER phản ánh toàn bộ đường trade-off threshold, còn ACC@1%FAR phản ánh một operating point cụ thể. Do đó, nếu mục tiêu là một hệ thống compact có calibration linh hoạt, Triplet là ứng viên đáng giữ lại chứ không nên loại bỏ chỉ vì ACC@1%FAR thấp hơn 0.40 điểm.

Hướng cải thiện: thử hybrid Triplet+GE2E cho EdgeSpot nhưng với trọng số nhỏ và có warmup; tune triplet margin; thử mining khó hơn nhưng cần kiểm soát collapse.

## 8. Phân tích vì sao một số tổ hợp chưa tốt

### 8.1. SCAF collapse trên cap620

Nhiều cấu hình dùng SCAF hoặc SCAF+GE2E bị collapse. Dấu hiệu rõ nhất là:

- AUC gần 50%.
- EER gần 50%.
- FRR@FAR = 100%.
- F1 = 0%.
- Keyword ACC = 9.09%.

Các cấu hình collapse gồm:

| Pipeline | AUC | FRR@1%FAR | ACC@1%FAR | F1 |
| --- | ---: | ---: | ---: | ---: |
| DSCNN-L + PCEN + SCAF | 50.00 | 100.00 | 69.44 | 0.00 |
| DSCNN-L + PCEN + SCAF+GE2E | 50.00 | 100.00 | 69.44 | 0.00 |
| EdgeSpotFull T4 + MFCC + SCAF | 50.00 | 100.00 | 69.44 | 0.00 |
| EdgeSpotFull T4 + PCEN + SCAF | 50.00 | 100.00 | 69.44 | 0.00 |
| EdgeSpotFull T4 + PCEN + SCAF+GE2E | 50.00 | 100.00 | 69.44 | 0.00 |

Không được diễn giải `ACC@1%FAR = 69.44%` trong các dòng này là kết quả tốt. Trong open-set KWS, nếu model reject gần như toàn bộ sample, nó có thể đúng với nhiều unknown/negative samples nhưng miss toàn bộ positive keywords. Khi F1 bằng 0 và FRR bằng 100%, hệ thống không nhận được keyword thật. Vì vậy, AUC, EER, FRR và F1 quan trọng hơn open-set ACC đơn lẻ trong trường hợp này.

Nguyên nhân khả dĩ:

1. Số lớp train rất lớn. MSWC cap620 có 37,387 train words. SCAF là angular classification loss, cần quản lý classifier head với rất nhiều class. Nếu scale, margin hoặc sub-center không phù hợp, gradient có thể quá nhiễu hoặc làm embedding co cụm sai.
2. Episodic sampling chỉ dùng 30 class mỗi episode. Trong khi SCAF có ý nghĩa global class separation, mỗi batch episodic chỉ nhìn thấy một phần cực nhỏ của không gian class.
3. Hybrid SCAF+GE2E có thể bị mismatch thang đo loss. Nếu SCAF dominating trong giai đoạn đầu, embedding có thể collapse trước khi GE2E kịp tạo cấu trúc centroid tốt.
4. Hyperparameter SCAF có thể phù hợp với Microset hoặc Top500 nhưng không còn phù hợp với 37k-class cap620.

Hướng cải thiện:

- Giảm trọng số SCAF trong hybrid loss, ví dụ thử `0.05`, `0.1`, `0.2` thay vì dùng trọng số ngang với GE2E.
- Warmup bằng Triplet hoặc GE2E trước, sau đó mới bật SCAF.
- Giảm margin hoặc scale của SCAF.
- Dùng class-balanced mini-batch lớn hơn hoặc memory bank cho angular classifier.
- Thử SCAF trên Top500 hoặc subset trước khi áp dụng cho full 37k words.
- Log embedding norm, class-center norm, loss component riêng và tỉ lệ gradient của từng loss để phát hiện collapse sớm.

### 8.2. MFCC không phù hợp với EdgeSpotFull T4

EdgeSpotFull T4 + MFCC cho kết quả thấp hơn rõ rệt so với EdgeSpotFull T4 + PCEN. Với GE2E, đổi từ MFCC sang PCEN tăng ACC@1%FAR từ 70.76% lên 79.98% và tăng F1 từ 48.82% lên 70.68%.

Lý do là EdgeSpotFull T4 được thiết kế theo hướng xử lý mel/PCEN time-frequency map. MFCC là đặc trưng đã nén cepstral, có thể hữu ích cho một số baseline cổ điển nhưng làm mất nhiều chi tiết phổ và động học thời gian mà backbone nhỏ cần để phân tách keyword. Với mô hình compact, mất thông tin ở frontend khó được bù lại bằng capacity ở backbone.

Hướng cải thiện: không nên dùng MFCC cho EdgeSpotFull T4 trong final system. MFCC chỉ nên giữ trong ablation để chứng minh vai trò của PCEN.

### 8.3. Tăng dữ liệu không tự động giải quyết mọi vấn đề

cap620 có gần 3 triệu train files, lớn hơn nhiều so với manifest20/cap20, nhưng một số pipeline vẫn collapse. Điều này cho thấy vấn đề không chỉ là thiếu dữ liệu. Với loss hoặc frontend không phù hợp, thêm dữ liệu có thể không giúp, thậm chí làm training khó hơn vì class space lớn hơn và nhiễu hơn.

Kết luận thực nghiệm là: tăng dữ liệu chỉ có ích khi objective và architecture đã ổn định. Với cap620, cấu hình ổn định là PCEN + GE2E hoặc PCEN + Triplet. Các loss angular/hybrid cần tuning lại trước khi dùng trên full vocabulary.

## 9. So sánh với paper EdgeSpot-4

Paper EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting, arXiv:2601.16316, được nộp ngày 2026-01-22. Paper báo cáo EdgeSpot-4 đạt 10-shot ACC@1%FAR = 82.0%, với khoảng 128k parameters và 29.4M MACs. Nguồn: https://arxiv.org/abs/2601.16316.

So sánh chính với kết quả cap620 fixed:

| Hệ thống | Data/profile trong project | Model size | ACC@1%FAR | Nhận xét |
| --- | --- | ---: | ---: | --- |
| EdgeSpot-4 paper | Theo paper EdgeSpot | 128k params, 29.4M MACs | 82.0 | Mốc paper |
| DSCNN-L + PCEN + GE2E | MSWC cap620 FLAC | khoảng 412.9k params | 82.34 +/- 1.19 | Cao hơn rất nhẹ, nhưng model lớn hơn và chênh lệch nằm trong std |
| EdgeSpotFull T4 + PCEN + GE2E | MSWC cap620 FLAC | khoảng 130.6k params | 79.98 +/- 0.98 | Chưa vượt EdgeSpot-4 paper |
| EdgeSpotFull T4 + PCEN + Triplet | MSWC cap620 FLAC | khoảng 130.6k params | 79.58 +/- 1.35 | Chưa vượt EdgeSpot-4 paper, nhưng AUC/F1 tốt hơn GE2E trong EdgeSpot group |

Claim an toàn:

> Trong protocol cap620 fixed 16-pipeline, cấu hình tốt nhất của project là DSCNN-L + PCEN + GE2E, đạt 82.34% ACC@1%FAR trên GSC-test100, xấp xỉ và nhỉnh nhẹ so với mốc 82.0% của EdgeSpot-4 paper. Tuy nhiên, cấu hình EdgeSpotFull T4 compact của project chỉ đạt tối đa 79.98% ACC@1%FAR trong protocol này, nên chưa vượt EdgeSpot-4 paper ở cùng metric chính.

Không nên viết:

- "Project đã vượt EdgeSpot-4" nếu không nói rõ đó là DSCNN-L, không phải EdgeSpotFull T4.
- "EdgeSpotFull T4 của project tốt hơn paper" dựa trên cap620 fixed run.
- "ACC@5%FAR cao hơn 82%" để claim vượt paper, vì paper so sánh ở 1% FAR.

Một kết quả khác cần phân biệt: project có artifact Top500 epoch13 của EdgeSpotFull T4 + PCEN + SCAF+GE2E đạt 85.62% ACC@1%FAR trên GSC-test100. Kết quả này cao hơn mốc 82.0% của EdgeSpot-4 paper, nhưng nó thuộc một profile huấn luyện khác, không phải protocol cap620 fixed 16-pipeline. Vì vậy, có thể trình bày như "separate artifact evidence", không dùng để thay thế kết luận của cap620 fixed matrix.

## 10. Diễn giải khoa học theo kiểu bài báo

### 10.1. Research question 1: Frontend nào phù hợp?

Kết quả cho thấy PCEN nhất quán tốt hơn MFCC, đặc biệt khi kết hợp với GE2E. Điều này củng cố giả thuyết rằng few-shot open-set KWS cần frontend ổn định với biến thiên âm lượng và noise. Trong môi trường cross-dataset MSWC -> GSC, PCEN giúp giảm domain shift tốt hơn MFCC.

### 10.2. Research question 2: Loss nào phù hợp?

GE2E tốt nhất với DSCNN-L vì nó khớp trực tiếp với prototype inference. Triplet lại rất cạnh tranh với EdgeSpotFull T4, đặc biệt ở AUC/EER/F1. SCAF và SCAF+GE2E chưa ổn định trong cap620 fixed, dù từng có tín hiệu tốt ở Microset/Top500. Do đó, loss tốt không thể kết luận độc lập khỏi data scale và backbone.

### 10.3. Research question 3: Backbone nào tốt hơn?

DSCNN-L tốt hơn về accuracy tuyệt đối. EdgeSpotFull T4 vẫn có giá trị vì số tham số thấp hơn khoảng 3 lần. Với metric test100@1%FAR, best DSCNN đạt 82.34%, best EdgeSpot đạt 79.98%, chênh 2.36 điểm. Với test100@5%FAR, chênh giữa best DSCNN và best EdgeSpot là 3.31 điểm nếu chọn best EdgeSpot theo ACC@5%FAR.

### 10.4. Research question 4: Có thể nâng cao tiếp không?

Có. Kết quả chỉ ra các hướng cụ thể thay vì thử ngẫu nhiên:

1. Với mục tiêu accuracy: tiếp tục tối ưu `DSCNN-L + PCEN + GE2E` bằng tăng episode budget, hard episode mining, và lựa chọn checkpoint theo metric tổng hợp ACC@1%FAR + AUC + F1.
2. Với mục tiêu compact model: tối ưu `EdgeSpotFull T4 + PCEN + Triplet` và `EdgeSpotFull T4 + PCEN + GE2E` thay vì SCAF. Triplet đáng ưu tiên vì AUC/EER/F1 tốt hơn trong EdgeSpot group.
3. Với mục tiêu vượt EdgeSpot-4 bằng EdgeSpotFull T4: cần thêm KD hoặc teacher-guided objective, vì paper EdgeSpot có sử dụng knowledge distillation từ self-supervised teacher. Chỉ dùng GE2E/Triplet trên cap620 hiện chưa đủ để vượt 82.0% ACC@1%FAR.
4. Với SCAF/SCAF+GE2E: cần ablation riêng về loss weight, margin, scale, warmup và class subset. Không nên tiếp tục chạy full cap620 SCAF nếu chưa sửa collapse.

## 11. Cách viết vào thesis

Đoạn đề xuất:

> Để đánh giá có hệ thống ảnh hưởng của kiến trúc, frontend và objective huấn luyện, đồ án thực hiện một thí nghiệm 16 pipeline cố định trên MSWC English cap620 FLAC. Tất cả pipeline dùng cùng dữ liệu, cùng số epoch, cùng số episode và cùng protocol GSC few-shot open-set. Kết quả cho thấy frontend PCEN là thành phần quan trọng nhất để tăng độ ổn định cross-dataset, đặc biệt khi kết hợp với GE2E. Cấu hình tốt nhất là DSCNN-L + PCEN + GE2E, đạt 82.34% ACC@1%FAR và 86.57% ACC@5%FAR trên GSC-test100. Trong nhóm model nhỏ, EdgeSpotFull T4 + PCEN + GE2E đạt ACC@1%FAR cao nhất, 79.98%, trong khi EdgeSpotFull T4 + PCEN + Triplet tốt hơn về AUC, EER và F1. Điều này cho thấy lựa chọn loss cho model compact cần xét nhiều metric, không chỉ một operating point.

Đoạn so sánh paper:

> So với EdgeSpot-4 paper, mốc được báo cáo là 82.0% ACC@1%FAR ở setting 10-shot với 128k parameters và 29.4M MACs. Cấu hình DSCNN-L + PCEN + GE2E của đồ án đạt 82.34% ACC@1%FAR, tức xấp xỉ và nhỉnh nhẹ mốc paper, nhưng dùng backbone lớn hơn. Ngược lại, EdgeSpotFull T4 compact trong protocol cap620 đạt tối đa 79.98% ACC@1%FAR, thấp hơn paper khoảng 2.02 điểm. Vì vậy, kết luận đúng là project đã đạt mức cạnh tranh với EdgeSpot-4 về metric bằng cấu hình accuracy-oriented, nhưng phiên bản compact EdgeSpotFull T4 vẫn cần cải thiện để vượt paper trong điều kiện so sánh nghiêm ngặt.

Đoạn về SCAF collapse:

> Một quan sát quan trọng là SCAF và SCAF+GE2E không ổn định trên profile cap620. Nhiều cấu hình có AUC gần 50%, EER gần 50%, FRR@FAR bằng 100% và F1 bằng 0, nghĩa là model gần như reject toàn bộ positive queries. Các dòng này không được xem là tốt dù open-set ACC khoảng 69.44%, vì open-set ACC bị ảnh hưởng bởi tỉ lệ unknown/negative. Kết quả này cho thấy angular classification loss cần được tune lại khi số lớp train tăng lên hàng chục nghìn.

## 12. Threats to validity

1. So sánh với EdgeSpot-4 paper chỉ dựa trên metric được báo cáo công khai, chưa phải reproduction đầy đủ cùng code, cùng data split và cùng training recipe.
2. Colab cap620 dùng A100 và PyTorch 2.11, trong khi server ict6 dùng CUDA 10.2/K80. Điều này không ảnh hưởng trực tiếp đến kết quả final đã ghi, nhưng cần ghi rõ môi trường.
3. Checkpoint selection dùng GSC-dev 3 runs mỗi 5 epoch. Vì selection runs ít hơn final test100, vẫn có khả năng nhiễu trong chọn checkpoint.
4. Số episode cố định làm cho cap620 chưa khai thác hết toàn bộ 2.99M train files. Kết quả phản ánh hiệu quả trong ngân sách train hiện tại, không phải upper bound của dataset.
5. Một số kết quả Top500 cũ tốt hơn cap620 nhưng thuộc profile khác; không được trộn vào ranking cap620 fixed.

## 13. Kết luận cho ba đầu việc hiện tại

1. Thực nghiệm 16 pipeline đã hoàn tất và đủ điều kiện báo cáo. Kết quả chính là PCEN + GE2E tốt nhất cho DSCNN, còn PCEN + Triplet/GE2E là hai hướng tốt nhất cho EdgeSpotFull T4. SCAF cần được coi là chưa ổn định trên cap620.
2. So với paper EdgeSpot-4, best overall của project xấp xỉ hoặc nhỉnh nhẹ paper nhưng là DSCNN lớn hơn. EdgeSpotFull T4 compact trong cap620 chưa vượt paper.
3. Khi viết thesis, nên trình bày theo logic: protocol cố định -> data/train/eval config -> bảng test100 -> phân tích từng trục -> so sánh paper -> giới hạn và hướng cải thiện. Không nên chỉ đưa bảng số liệu mà thiếu giải thích collapse và claim boundary.
