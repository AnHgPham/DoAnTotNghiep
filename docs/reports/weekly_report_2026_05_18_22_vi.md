# Báo Cáo Tuần 18/05 - 22/05/2026

## 1. Tổng Quan Công Việc

Trong tuần này, em tập trung hoàn thiện ba hướng chính của dự án few-shot open-set keyword spotting:

1. Chạy và tổng hợp thực nghiệm Microset để chọn cấu hình model tốt nhất.
2. Mở rộng cấu hình đã chọn sang Top500 và xử lý vấn đề Colab/artifact.
3. Nâng cấp demo để giải thích rõ single detection, long audio, open-set rejection và policy calibration.

Kết luận kỹ thuật quan trọng nhất của tuần: qua các thử nghiệm trên Microset với nhiều cấu hình, hướng **EdgeSpotFull T4 + SCAF+GE2E** là cấu hình phù hợp nhất trong các cấu hình đã thử. Cấu hình này được dùng làm nền để train Top500.

## 2. Microset Experiments

### 2.1. Mục Tiêu

Microset được dùng làm mốc chính để kiểm tra nhanh các hướng kiến trúc/loss trước khi mở rộng sang Top500. Lý do chọn Microset:

- có official CSV split;
- kích thước vừa đủ để chạy nhiều cấu hình;
- giảm rủi ro leakage nếu dùng manifest đúng;
- phù hợp để so sánh baseline và hướng cải tiến.

### 2.2. Các Cấu Hình Đã Thử

Các hướng chính:

- baseline DSCNN-L + MFCC + Triplet;
- EdgeSpotFull T4 + mel-PCEN + SCAF;
- EdgeSpotFull T4 + mel-PCEN + SCAF+GE2E.

Trong đó SCAF giúp tách class trong embedding space, còn GE2E giúp training sát hơn với cơ chế support/query/prototype khi inference.

### 2.3. Kết Quả Chính

Cấu hình Microset tốt nhất hiện tại:

- model: EdgeSpotFull T4;
- feature: mel-PCEN;
- loss: SCAF+GE2E;
- checkpoint: `epoch_05.pt`;
- GSC test100:
  - ACC@5%FAR: 86.12%;
  - KW-ACC: 77.66%;
  - F1: 82.41%;
  - AUC: 95.61%;
  - EER: 11.54%.

Ý nghĩa: Microset cho thấy hướng EdgeSpotFull T4 + SCAF+GE2E tốt hơn pipeline cũ và đủ hợp lý để chuyển sang Top500.

## 3. Top500 Training

### 3.1. Lý Do Mở Rộng Sang Top500

Microset chỉ có 31 keyword, nên dù hữu ích cho chọn cấu hình, nó vẫn hạn chế về độ đa dạng từ và speaker. Top500 được dùng để mở rộng phạm vi huấn luyện:

- nhiều từ hơn;
- nhiều biến thể phát âm hơn;
- kỳ vọng embedding generalize tốt hơn khi demo/open-set.

### 3.2. Run Top500 Lần 1

Run Top500 trước đó có log/kết quả hứa hẹn, nhưng lúc đó chưa tải/package artifact đầy đủ về local ngay sau khi chạy. Do session/Colab bị mất, checkpoint/result đầy đủ không còn chắc chắn trong local package. Vì vậy kết quả này chỉ nên ghi là lịch sử tiến độ, không dùng làm claim chính nếu thiếu artifact.

Bài học rút ra:

- không chờ đến cuối mới tải;
- phải lưu checkpoint từng epoch;
- phải package vào Drive trước khi đóng Colab;
- local report chỉ claim thứ có file evidence.

### 3.3. Run Top500 Lần 2

Sau đó pipeline được sửa để an toàn hơn:

- `--save-every 1`;
- `--save-latest-every-epoch`;
- checkpoint lưu vào Drive sau mỗi epoch;
- dataset dùng session-first để tránh copy cache WAV lớn lên Drive.

Run này bị dừng ở epoch 13 do giới hạn Colab/session/units. Hiện checkpoint chắc chắn đang có local là:

```text
server/final_kws_artifacts_package/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_13.pt
```

Kết quả dev30 của epoch13:

- ACC@1%FAR: 86.68%;
- ACC@5%FAR: 88.87%;
- FRR@5%FAR: 20.36%;
- AUC: 95.12%;
- F1: 81.71%.

Kết luận: Top500 có tín hiệu tốt, nhưng hiện nên mô tả là kết quả sơ bộ/demo vì chưa có final test100 artifact đầy đủ.

## 4. Demo/UI Improvements

### 4.1. Model Switcher

Đã thiết kế lại flow chọn model:

- Microset epoch05;
- Top500 epoch13;
- legacy DSCNN nếu còn checkpoint.

Khi đổi model, UI hỏi người dùng rebuild enrollment hoặc clear enrollment. Đây là cần thiết vì prototype phụ thuộc encoder hiện tại.

### 4.2. Long Audio

Đã cải thiện cách hiển thị long-audio:

- không dùng bảng rộng làm view chính;
- có summary cards;
- có expected timeline;
- có detected timeline;
- có detection cards;
- có missed expected cards;
- giải thích lý do miss: no overlap, threshold reject, guard reject, wrong prediction, outside enrollment.

Điều này giúp demo dễ hiểu hơn khi file dài 50 từ, thay vì chỉ nhìn một bảng khó đọc.

### 4.3. Open-Set Test

Đã chuyển open-set từ placeholder sang flow thật:

- GSC 17 known / 17 unknown;
- heldout `visual`;
- candidate label restriction;
- keyword accuracy;
- unknown reject accuracy;
- false accept rate;
- false reject rate;
- balanced score.

Qua thử nghiệm UI, policy cân bằng hơn hiện tại là:

- Guard ON;
- Per-class OFF;
- accept margin 0.05.

Guard OFF nhận keyword dễ hơn nhưng false accept unknown cao hơn, nên không phù hợp làm setting open-set cân bằng.

### 4.4. React/Vite UI

Đã scaffold UI mới tại:

```text
src/demo/ui
```

Stack:

- React;
- TypeScript;
- Vite;
- CSS design tokens;
- typed API client.

Build đã pass:

```text
npm run typecheck
npm run build
```

FastAPI sẽ serve React build nếu `src/demo/ui/dist` tồn tại, fallback về UI cũ nếu chưa build.

## 5. Problems Encountered

### 5.1. Colab Session And Units

Top500 tốn tài nguyên, run có thể dừng giữa chừng. Vấn đề này ảnh hưởng trực tiếp đến việc có artifact cuối cùng hay không.

### 5.2. Drive Copy Time

Copy toàn bộ WAV cache lên Drive quá chậm. Hướng xử lý là session-first dataset, Drive-first artifacts.

### 5.3. Open-Set Tradeoff

Nếu threshold/guard quá thoáng, unknown dễ bị nhận nhầm thành keyword. Nếu quá chặt, keyword đúng bị reject. Vì vậy cần báo cáo balanced score thay vì chỉ open-set accuracy.

### 5.4. Long Audio Timing

Long audio dễ bị lệch vì segmentation. Khi số label và số detection không khớp, cần timing JSON và miss explanation.

## 6. Current Artifact Story

| Artifact | Status | Use |
|---|---|---|
| Microset EdgeSpotFull T4 + SCAF+GE2E epoch05 | official locked | mốc thesis chính |
| Top500 EdgeSpotFull T4 + SCAF+GE2E epoch13 | local available | demo/sơ bộ |
| Top500 epoch25 historical | progress only | ghi lịch sử nếu không có artifact đầy đủ |

Nguyên tắc báo cáo: ghi đúng những gì có file evidence. Microset là mốc chính. Top500 epoch13 là checkpoint chắc chắn đang có. Top500 epoch25 chỉ ghi là run từng có log tốt nếu chưa có artifact local đầy đủ.

## 7. Next Work

1. Hoàn thiện UI React đến mức demo chính.
2. Chạy Playwright screenshots desktop/mobile.
3. Hoàn thiện technical manual, API reference, troubleshooting.
4. Viết thesis draft song ngữ.
5. Khi có tài nguyên, resume/rerun Top500 để có final checkpoint/test100.
6. Thêm streaming benchmark: latency, false alarm/hour, miss rate.
7. Chuẩn hóa export report để copy trực tiếp vào báo cáo/thesis.

## 8. Email-Ready Summary

Tuần này em đã thử nhiều cấu hình trên MSWC Microset để chọn hướng model phù hợp cho bài toán few-shot open-set keyword spotting. Qua các thử nghiệm, cấu hình EdgeSpotFull T4 kết hợp SCAF+GE2E cho kết quả tốt nhất trong các cấu hình đã chạy, nên em dùng cấu hình này làm nền để train tiếp trên Top500.

Kết quả Microset đã khóa hiện tại đạt ACC@5%FAR 86.12%, KW-ACC 77.66%, F1 82.41% trên GSC test100. Với Top500, hướng train cho tín hiệu tốt hơn nhưng bị gián đoạn bởi Colab/session/units; checkpoint chắc chắn em đang có hiện tại là epoch13, với dev30 ACC@5%FAR 88.87%. Em đang dùng checkpoint này cho demo và sẽ chạy tiếp Top500 khi có tài nguyên.

Ngoài phần training, em cũng nâng cấp demo: chọn model Microset/Top500, long-audio timing, giải thích miss, open-set 17/17 và calibration threshold/guard.
