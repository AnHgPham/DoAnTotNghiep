# Thesis Draft VI - Few-Shot Open-Set Keyword Spotting

## Abstract

Đồ án nghiên cứu bài toán **few-shot open-set keyword spotting**: hệ thống cần nhận diện các từ khóa người dùng đăng ký chỉ từ một số mẫu rất nhỏ, đồng thời từ chối các từ hoặc âm thanh không thuộc tập keyword. Bài toán này khác với keyword spotting closed-set truyền thống vì model không chỉ chọn một trong các lớp cố định, mà còn phải đưa ra quyết định `unknown` khi input không đủ gần với bất kỳ keyword nào.

Pipeline ban đầu của dự án dùng DSCNN-L, MFCC và Triplet loss. Qua các thử nghiệm trên MSWC Microset English, dự án chuyển sang hướng EdgeSpot-style encoder: **EdgeSpotFull T4**, input mel-PCEN, embedding 64-D và loss hybrid **SCAF+GE2E**. SCAF giúp tăng độ tách biệt class trong embedding space; GE2E mô phỏng gần hơn cơ chế support-query/prototype khi inference few-shot. Cấu hình này được chọn làm mốc chính cho thesis và tiếp tục mở rộng sang MSWC Top500.

Kết quả Microset đã khóa hiện tại đạt `ACC@5%FAR = 86.12%`, `KW-ACC = 77.66%`, `F1 = 82.41%`, `AUC = 95.61%` trên GSC test100. Top500 hiện có checkpoint local chắc chắn ở epoch 13, cho tín hiệu sơ bộ tốt trên GSC-dev (`ACC@5%FAR = 88.87%`), nhưng chưa đặt là final vì run bị gián đoạn bởi giới hạn Colab/session/units. Demo web bổ sung model switcher, long-audio timing analysis, open-set 17/17 sampled evaluation và calibration để minh họa tradeoff giữa nhận đúng keyword và từ chối unknown.

## 1. Introduction

Keyword Spotting là bài toán phát hiện từ khóa trong tín hiệu âm thanh. Các ứng dụng thường gặp gồm wake word, điều khiển thiết bị thông minh, trợ lý giọng nói và giao diện rảnh tay. Tuy nhiên, nhiều hệ thống KWS truyền thống giả định tập keyword cố định và cần nhiều dữ liệu huấn luyện cho mỗi từ. Điều này chưa phù hợp với tình huống người dùng muốn thêm từ khóa cá nhân hóa bằng vài mẫu giọng nói.

Đồ án này tập trung vào ba yêu cầu:

1. **Few-shot personalization:** người dùng chỉ cần một số mẫu nhỏ để tạo keyword mới.
2. **Open-set rejection:** hệ thống phải reject từ không thuộc tập enroll thay vì đoán bừa.
3. **Demo-level usability:** hệ thống phải có UI giải thích rõ vì sao một từ được nhận, bị reject, hoặc bị miss.

## 2. Problem Statement

Cho một tập keyword đã enroll `K = {k1, k2, ..., kn}`, mỗi keyword có một số mẫu support. Với audio query `x`, hệ thống cần trả về:

- một keyword `ki` nếu `x` đủ giống prototype của `ki`;
- hoặc `unknown` nếu `x` không thuộc keyword nào đã enroll.

Khó khăn chính:

- keyword ngắn, dễ gần âm nhau;
- speaker/accent/noise khác giữa train và demo;
- số mẫu enrollment nhỏ;
- threshold quá thấp gây false reject, quá cao gây false accept;
- long audio có lỗi segmentation và timing;
- streaming có window lệch, noise và cooldown.

## 3. Research Questions

1. Chuyển từ DSCNN-L/MFCC/Triplet sang EdgeSpot-style mel-PCEN encoder có cải thiện few-shot open-set KWS không?
2. SCAF và GE2E đóng vai trò gì trong việc học embedding phù hợp với prototype inference?
3. Cấu hình tốt trên Microset có mở rộng được sang Top500 không?
4. Trong demo open-set, policy threshold/per-class/guard ảnh hưởng thế nào đến keyword accuracy và unknown rejection?

## 4. Related Work

### Keyword Spotting

KWS truyền thống thường dùng CNN/DSCNN/TC-ResNet để phân loại từ khóa cố định. Các mô hình này hiệu quả trên thiết bị edge, nhưng closed-set classifier có xu hướng bắt buộc chọn một nhãn, không tự nhiên reject unknown.

### Few-Shot And Prototype Methods

Few-shot learning thường dùng embedding và prototype. Ý tưởng chính là học một không gian vector nơi mẫu cùng class gần nhau, mẫu khác class xa nhau. Khi có class mới, chỉ cần vài support samples để tạo prototype.

### EdgeSpot-Style Models

EdgeSpot hướng đến model nhỏ, hiệu quả cho KWS. Dự án dùng cảm hứng EdgeSpotFull T4 với mel-PCEN, block kiểu BC-ResNet/Fused BC-ResNet và embedding 64-D. Đồ án không claim reproduction đầy đủ EdgeSpot paper.

### SCAF And GE2E

SCAF/Sub-center ArcFace dùng angular margin và nhiều sub-center để class có biên tách tốt hơn. GE2E đến từ hướng speaker verification, tạo centroid support và tối ưu query gần centroid đúng, xa centroid sai. Trong đồ án, GE2E được thêm vì inference few-shot KWS cũng dùng support/prototype/query.

## 5. Proposed Method

### Feature Extraction

Audio được chuẩn hóa 16 kHz, mono, trim/pad về 1 giây. EdgeSpotFull dùng mel spectrogram 40x101 và PCEN. Tensor input chính là `(B, 1, 40, 101)`.

### Encoder

Encoder chính:

- EdgeSpotFull T4;
- `tau=4`;
- khoảng 130,598 tham số;
- output embedding 64-D.

### Training Loss

Loss hybrid:

```text
L = L_scaf + L_ge2e
```

Trong log training, `kd=0.0` nếu không dùng teacher distillation. SCAF tách class ở embedding space; GE2E buộc query gần centroid support đúng.

### Inference

Enrollment:

```text
prototype(keyword) = mean(encoder(feature(sample_i)))
```

Detection:

```text
top1 = argmin_keyword L2(query_embedding, prototype(keyword))
top2 = second nearest
margin = dist(top2) - dist(top1)
accept if dist(top1) <= threshold and margin >= accept_margin
```

Open-set rejection dựa trên threshold và margin. Với open-set demo hiện tại, policy cân bằng tốt nhất là Guard ON, Per-class OFF, accept margin 0.05.

## 6. Dataset And Protocol

### MSWC Microset English

Microset dùng official CSV split:

- train khoảng 69,868 WAV;
- dev khoảng 13,114 WAV;
- test khoảng 13,117 WAV.

Project dùng manifest file-level để tránh leakage. Đây là mốc thực nghiệm chính dùng để so sánh cấu hình.

### Google Speech Commands v2

GSC dùng cho đánh giá cross-dataset và demo. Protocol chính là `gsc_edgespot_exact` với k-shot 10, true silence và unknown/negative words.

### Top500

Top500 mở rộng từ hướng Microset:

- 450 train words;
- 50 validation words;
- full clips với `max_per_word=0`;
- dữ liệu session-first trên Colab;
- checkpoint/result lưu Drive.

Top500 hiện có artifact local chắc chắn ở epoch 13. Epoch 25 chỉ nên ghi như historical/progress nếu không có checkpoint/result local đầy đủ.

### DEMAND

DEMAND được dùng cho noise augmentation trong training. Evaluation GSC không phụ thuộc DEMAND.

## 7. Experimental Setup

### Models Compared

| Model | Feature | Loss | Role |
|---|---|---|---|
| DSCNN-L | MFCC | Triplet | baseline |
| EdgeSpotFull T4 | mel-PCEN | SCAF | ablation |
| EdgeSpotFull T4 | mel-PCEN | SCAF+GE2E | selected |

### Training

Training dùng episodic sampler. Mỗi episode chọn `n_classes` keyword và `n_samples` mẫu mỗi keyword. Với Top500, số episode/epoch lớn hơn Microset và cần checkpoint-save chặt hơn do Colab dễ gián đoạn.

### Evaluation

Metric chính:

- ACC@1%FAR;
- ACC@5%FAR;
- FRR@5%FAR;
- AUC;
- EER;
- KW-ACC;
- F1.

Dev dùng để chọn checkpoint, test100 dùng để báo cáo kết quả khóa.

## 8. Results

### Microset Main Result

| Configuration | ACC@5%FAR | KW-ACC | F1 | AUC | EER |
|---|---:|---:|---:|---:|---:|
| DSCNN-L + Triplet | lower baseline | lower baseline | lower baseline | lower baseline | higher EER |
| EdgeSpotFull T4 + SCAF | 85.21% | 74.52% | 81.92% | available in result logs | available in result logs |
| EdgeSpotFull T4 + SCAF+GE2E | **86.12%** | **77.66%** | **82.41%** | **95.61%** | **11.54%** |

Cấu hình SCAF+GE2E được chọn vì cân bằng tốt hơn giữa keyword recognition và open-set behavior. Nó không chỉ tăng keyword accuracy, mà còn phù hợp hơn với cơ chế prototype inference.

### Top500 Progress

Top500 epoch13 dev30:

| Metric | Value |
|---|---:|
| ACC@1%FAR | 86.68% |
| ACC@5%FAR | 88.87% |
| FRR@5%FAR | 20.36% |
| AUC | 95.12% |
| F1 | 81.71% |

Kết quả này cho thấy hướng Top500 có tín hiệu tốt. Tuy nhiên, vì run bị gián đoạn ở epoch 13 do Colab/session/units, checkpoint này nên được mô tả là **preliminary/local artifact** thay vì final Top500 result.

### Open-Set Demo

UI dùng split GSC 17 known / 17 unknown / heldout visual. Metric chính cho demo là balanced score:

```text
balanced_score = 0.5 * keyword_acc + 0.5 * unknown_reject_acc
```

Qua thử nghiệm UI, Guard ON + Per-class OFF + accept margin 0.05 cân bằng hơn. Guard OFF có thể làm keyword ACC cao hơn nhưng unknown reject ACC giảm mạnh, dẫn đến false accept nhiều.

## 9. Demo System

Demo web gồm:

- model switcher giữa Microset epoch05, Top500 epoch13 và legacy nếu có;
- GSC enrollment preset;
- single detection với top-3, L2, threshold, margin;
- long audio với label/timing upload, timeline, detection cards, missed expected cards;
- open-set 17/17 test và calibration;
- streaming microphone/WebSocket;
- export session report.

React/Vite UI mới thay thế dần UI static cũ. FastAPI serve React build nếu `src/demo/ui/dist` tồn tại.

## 10. Discussion

SCAF+GE2E tốt hơn vì hai cơ chế bổ sung nhau:

- SCAF giúp class tách rõ trong embedding space.
- GE2E làm training giống inference: support tạo centroid, query phải gần centroid đúng.

Top500 cần thiết vì Microset nhỏ và chỉ có 31 keyword. Top500 tăng đa dạng từ và speaker, giúp model học embedding rộng hơn. Tuy nhiên Top500 đắt tài nguyên, nên artifact policy quan trọng.

Open-set khó vì unknown có thể rất gần keyword. Các từ ngắn như `three/tree`, `four/forward`, `no/go/on` dễ tạo top-1/top-2 margin nhỏ. Vì vậy guard giúp giảm false accept nhưng tăng false reject.

Streaming khó hơn static vì window không đảm bảo trùng đúng 1 giây keyword, VAD có thể split/merge, và cooldown phải cân bằng giữa chống lặp và không bỏ sót.

## 11. Limitations

- Chưa train full MSWC quy mô lớn như paper.
- Không claim reproduction đầy đủ EdgeSpot.
- Top500 final test100 chưa có artifact local đầy đủ, hiện dùng epoch13 sơ bộ.
- Streaming chưa có benchmark chính thức về false alarm/hour và latency.
- Open-set UI là sampled demo-level evaluation, không thay thế GSC test100.
- Demo long audio ghép GSC clips dễ hơn speech tự nhiên liên tục.

## 12. Future Work

1. Rerun Top500 khi có Colab/GPU resources, lưu artifact đúng policy.
2. Chạy Top500 test100 final với checkpoint đầy đủ.
3. Thêm streaming benchmark: latency, false alarm/hour, miss rate.
4. Tối ưu calibration theo từng model profile.
5. Export report tự động thành appendix/tables.
6. Kiểm thử UI bằng Playwright screenshot desktop/mobile.
7. Thử teacher distillation nếu muốn bám EdgeSpot paper hơn.

## 13. Conclusion

Đồ án đã chuyển từ baseline DSCNN-L + MFCC + Triplet sang hướng EdgeSpotFull T4 + mel-PCEN + SCAF+GE2E. Các thử nghiệm Microset cho thấy cấu hình SCAF+GE2E là lựa chọn tốt nhất trong các cấu hình đã thử, đủ làm mốc thesis hiện tại. Hướng này được mở rộng sang Top500 và cho tín hiệu sơ bộ tốt ở epoch13, dù cần chạy tiếp để có final artifact. Demo web đã bổ sung các flow quan trọng để giải thích inference, long-audio miss và open-set calibration, giúp hệ thống không chỉ có metric mà còn có khả năng trình bày/debug rõ ràng.
