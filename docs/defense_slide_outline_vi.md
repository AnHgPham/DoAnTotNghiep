# Slide Outline Bảo Vệ

## 1. Title
- Few-Shot Open-Set Keyword Spotting.
- Mục tiêu: thêm keyword mới bằng vài mẫu enrollment và từ chối unknown speech.

## 2. Problem
- Keyword spotting truyền thống thường train fixed classes.
- Bài toán thực tế cần thêm keyword mới nhanh, ít mẫu, không retrain toàn bộ.
- Thách thức chính: speaker variation, noise, false alarm, unknown words.

## 3. System Pipeline
- Enrollment audio -> feature extractor -> encoder embedding -> prototype.
- Query audio -> embedding -> OpenNCM L2 scoring.
- Decision: accept keyword hoặc reject unknown theo threshold.

## 4. Dataset
- Train: MSWC Microset English official CSV split.
- Evaluation: Google Speech Commands v2.
- Protocol: `gsc_edgespot_exact`, 10-shot, true `_silence_`, 25 unknown words.
- Nhấn mạnh: Microset tạm thời, không phải Top500 full/full MSWC.

## 5. Baseline
- DSCNN-L + MFCC + Triplet.
- Ưu điểm: ổn định, dễ chạy.
- Hạn chế: embedding kém hơn khi chuyển domain MSWC -> GSC.

## 6. Proposed Method
- EdgeSpotFull T4 + mel-PCEN + 64-D embedding.
- SCAF để tách class tốt hơn.
- GE2E để training giống support/query-prototype ở inference.

## 7. Main Results
- Bảng so sánh DSCNN-L, EdgeSpotFull SCAF, EdgeSpotFull SCAF+GE2E.
- Final test100:
  - ACC@5% FAR: 86.12%.
  - Keyword ACC: 77.66%.
  - F1: 82.41%.
  - EER: 11.54%.

## 8. Interpretation
- EdgeSpotFull giảm FRR mạnh so với DSCNN.
- SCAF+GE2E cải thiện Keyword ACC và F1 so với SCAF-only.
- Kết quả ổn định từ dev30 sang test100.

## 9. Demo
- Enroll keyword samples.
- Hiển thị confidence, threshold, margin.
- Detection history và streaming view.

## 10. Limitations
- Chưa chạy Top500 full.
- Chưa full MSWC.
- Chưa KD teacher như EdgeSpot paper.
- Chưa có streaming benchmark chính thức.

## 11. Next Work
- Top500 full run.
- Per-word error/confusion analysis.
- Threshold calibration với impostor bank.
- Streaming benchmark: false alarms/hour, miss rate, latency.

## 12. Closing
- Kết quả hiện tại là mốc Microset đã khóa.
- Hướng EdgeSpotFull + SCAF+GE2E có tiềm năng để nâng lên experiment quy mô lớn hơn.
