# Báo cáo tiến độ tuần 10/06 - 14/06/2026

Kính gửi thầy,

Trong tuần này, em tập trung hoàn thiện ba nhóm công việc chính của đồ án: chạy và tổng hợp thực nghiệm lớn trên MSWC cap620, đánh giá lại khả năng so sánh với EdgeSpot-4, và tái cấu trúc bản thesis theo hướng giống một báo cáo khoa học hoàn chỉnh hơn.

## 1. Công việc đã thực hiện

### 1.1. Hoàn tất thí nghiệm fixed 16-pipeline trên MSWC cap620

Em đã hoàn tất thí nghiệm fixed 16-pipeline trên cấu hình MSWC English cap620 FLAC. Thí nghiệm này dùng cùng một data profile, cùng số epoch, cùng số episode và cùng protocol evaluation để so sánh công bằng giữa các pipeline.

Cấu hình dữ liệu và huấn luyện:

- Dữ liệu train: khoảng 2,989,780 audio files.
- Dữ liệu validation: khoảng 52,399 audio files.
- Số train words: 37,387.
- Số validation words: 763.
- Audio được chuyển sang FLAC để giảm dung lượng lưu trữ trên Colab.
- Training: 40 epochs, 150 episodes/epoch.
- Episode: 30 classes x 10 samples.
- Evaluation chính: GSC `gsc_edgespot_exact`, gồm `dev30@1%FAR`, `test100@1%FAR` và `test100@5%FAR`.

Ma trận 16 pipeline gồm:

- Backbone: DSCNN-L và EdgeSpotFull T4.
- Frontend: MFCC và PCEN.
- Loss/objective: Triplet, SCAF, GE2E, SCAF+GE2E.

Kết quả fixed run cho thấy cấu hình tốt nhất theo accuracy là:

- `DSCNN-L + PCEN + GE2E`
- `ACC@1%FAR = 82.34 ± 1.19%`
- `AUC = 92.42 ± 0.54%`
- `EER = 14.89 ± 0.84%`
- `F1 = 77.75 ± 1.15%`

Trong nhóm compact EdgeSpotFull T4, cấu hình tốt nhất theo ACC@1%FAR là:

- `EdgeSpotFull T4 + PCEN + GE2E`
- `ACC@1%FAR = 79.98 ± 0.98%`
- `AUC = 87.23 ± 0.75%`
- `EER = 20.23 ± 0.96%`
- `F1 = 70.68 ± 1.23%`

Ngoài ra, `EdgeSpotFull T4 + PCEN + Triplet` tuy có ACC@1%FAR thấp hơn nhẹ, nhưng có AUC, EER và F1 tốt hơn trong nhóm EdgeSpot fixed run:

- `ACC@1%FAR = 79.58 ± 1.35%`
- `AUC = 89.85 ± 0.63%`
- `EER = 18.22 ± 0.78%`
- `F1 = 73.29 ± 1.02%`

### 1.2. Chạy development run để tối ưu cấu hình mạnh nhất

Sau fixed 16-pipeline, em chạy thêm development run với budget lớn hơn để tối ưu hai hướng:

- Accuracy-oriented: `DSCNN-L + PCEN + GE2E`.
- Compact-oriented: `EdgeSpotFull T4 + PCEN + GE2E` và `EdgeSpotFull T4 + PCEN + Triplet hard`.

Cấu hình development run:

- 60 epochs.
- 300 episodes/epoch.
- Episode lớn hơn fixed run.
- Checkpoint selection dùng composite metric:

```text
Composite = mean(ACC@1%FAR, AUC, F1)
```

Kết quả tốt nhất hiện tại:

| Cấu hình | ACC@1%FAR | AUC | EER | F1 | Nhận xét |
| --- | ---: | ---: | ---: | ---: | --- |
| DSCNN-L + PCEN + GE2E | 86.36 ± 1.29 | 95.21 ± 0.45 | 11.32 ± 0.78 | 82.73 ± 1.11 | Best accuracy |
| EdgeSpotFull T4 + PCEN + GE2E | 82.87 ± 1.22 | 92.41 ± 0.44 | 14.82 ± 0.70 | 77.85 ± 0.97 | Best compact |
| EdgeSpotFull T4 + PCEN + Triplet hard | 69.10 ± 0.15 | 53.40 ± 0.48 | 47.84 ± 0.62 | 39.99 ± 0.60 | Bị collapse |

Như vậy, development run cải thiện rõ so với fixed run:

- `DSCNN-L + PCEN + GE2E`: tăng từ 82.34% lên 86.36% ACC@1%FAR.
- `EdgeSpotFull T4 + PCEN + GE2E`: tăng từ 79.98% lên 82.87% ACC@1%FAR.

### 1.3. Đánh giá lại so sánh với EdgeSpot-4

Em đã kiểm tra lại claim so với mốc EdgeSpot-4 paper. Mốc tham khảo đang dùng là:

- EdgeSpot-4 paper: `ACC@1%FAR = 82.0%`.
- Kích thước khoảng 128k parameters.
- Khoảng 29.4M MACs.

Kết luận hiện tại cần viết thận trọng:

- `DSCNN-L + PCEN + GE2E` đạt 86.36%, vượt rõ mean 82.0%, nhưng model lớn hơn EdgeSpot-4.
- `EdgeSpotFull T4 + PCEN + GE2E` đạt 82.87%, nhỉnh hơn mean 82.0%, nhưng chênh lệch nhỏ và nằm trong độ lệch chuẩn.
- Development run chưa bật knowledge distillation, trong khi EdgeSpot paper có dùng teacher-guided/distillation objective.

Do đó, em không claim là đã reproduce đầy đủ EdgeSpot-4. Cách diễn đạt đúng hơn là:

> Mô hình compact EdgeSpotFull T4 + PCEN + GE2E của đồ án đã đạt mức cạnh tranh và hơi cao hơn mean EdgeSpot-4 paper dưới protocol của project, nhưng chưa phải reproduction đầy đủ do khác recipe và chưa chạy KD.

### 1.4. Phân tích các failure modes

Em cũng phân tích các trường hợp thất bại để làm rõ hơn trong thesis:

- Nhiều cấu hình SCAF hoặc SCAF+GE2E bị collapse trên cap620, với AUC khoảng 50%, EER khoảng 50%, FRR@FAR 100% và F1 bằng 0.
- Điều này không chứng minh SCAF sai về bản chất, vì SCAF từng tốt ở Microset/Top500. Khả năng cao là các tham số scale, margin, loss weight và warmup chưa phù hợp khi số class train rất lớn.
- Branch `EdgeSpotFull T4 + PCEN + Triplet hard` trong development run cũng collapse. Em không kết luận Triplet kém, mà chỉ kết luận hard mining hiện tại quá gắt và cần thử semi-hard mining hoặc giảm hard-pair probability.

### 1.5. Tái cấu trúc thesis theo hướng reference master

Em đã đọc lại reference master thesis và thấy bản thesis cũ của em còn giống báo cáo kỹ thuật theo timeline hơn là một luận văn khoa học. Vì vậy, em đã tái cấu trúc bản thesis theo format gần hơn với reference:

1. Chương 1: Giới thiệu.
2. Chương 2: Nền tảng.
3. Chương 3: Phương pháp luận và thiết kế thực nghiệm.
4. Chương 4: Kết quả và thảo luận.
5. Chương 5: Kết luận và hướng phát triển.

Các phần đã bổ sung vào bản mới:

- Công thức ProtoNet và prototype.
- Công thức episodic support/query set.
- Công thức direct L2 decision.
- Công thức MFCC.
- Công thức PCEN.
- Công thức Triplet loss.
- Công thức GE2E.
- Công thức SCAF/Sub-center ArcFace.
- Công thức FAR, FRR, ACC@FAR và DET.
- Mapping rõ giữa reference master và hướng xử lý của đồ án.

Em cũng đã làm rõ điểm khác giữa reference và project:

- Reference dùng MSWC như một evaluation mở rộng.
- Project hiện dùng MSWC cap620 chủ yếu làm nguồn training lớn, còn final benchmark chính là GSC `gsc_edgespot_exact test100`.
- Code của project có hỗ trợ MSWC randomized evaluation 5-positive/50-negative với support/query 1:9, nhưng nhánh này chưa phải evidence chính trong thesis hiện tại.

## 2. Kết quả chính trong tuần

Kết quả quan trọng nhất trong tuần là em đã có được hai cấu hình chính cho thesis:

| Mục tiêu | Cấu hình | Kết quả chính |
| --- | --- | --- |
| Best accuracy | DSCNN-L + PCEN + GE2E | 86.36 ± 1.29% ACC@1%FAR |
| Best compact | EdgeSpotFull T4 + PCEN + GE2E | 82.87 ± 1.22% ACC@1%FAR |

Các kết luận kỹ thuật chính:

- PCEN ổn định hơn MFCC trong cross-dataset setting MSWC -> GSC.
- GE2E phù hợp với prototype inference vì cả training và inference đều dựa trên centroid/prototype.
- SCAF cần ablation riêng, không nên tiếp tục chạy full cap620 SCAF nếu chưa tune.
- EdgeSpotFull T4 + PCEN + GE2E đã cạnh tranh với EdgeSpot-4, nhưng claim cần giữ thận trọng vì chưa có KD.

## 3. Vấn đề gặp phải

Một số vấn đề chính trong tuần:

- Colab báo gần đầy disk khi chạy cap620, do dữ liệu và artifact rất lớn. Em xử lý bằng cách ưu tiên sync artifact lên Drive và không chạy duplicate run trong cùng runtime.
- Một số cấu hình SCAF bị reject gần như toàn bộ positive samples, làm open-set ACC có thể nhìn không quá thấp do nhiều negative được reject đúng, nhưng F1 bằng 0 và FRR bằng 100%, nên không thể xem là kết quả tốt.
- Hard Triplet trong development run collapse, cho thấy mining strategy cần được kiểm soát cẩn thận.
- Bản thesis cũ còn thiếu lớp công thức formal và chưa theo đúng format của reference master, nên em đã tạo lại bản reference-style.

## 4. Kế hoạch tuần tới

Tuần tới, em dự định tập trung vào các việc sau:

1. Hoàn thiện bản thesis reference-style:
   - chỉnh lại câu chữ tiếng Việt;
   - chuẩn hóa citation;
   - bổ sung hình/bảng nếu cần;
   - kiểm tra lại các claim không overstate.

2. Nếu còn thời gian GPU, chạy thêm một nhánh compact:
   - `EdgeSpotFull T4 + PCEN + GE2E + KD`;
   - mục tiêu là kiểm tra liệu KD có giúp vượt EdgeSpot-4 thuyết phục hơn không.

3. Chạy ablation nhỏ cho SCAF:
   - giảm scale/margin;
   - giảm SCAF weight;
   - warmup bằng GE2E trước;
   - thử subset nhỏ trước khi quay lại cap620.

4. Rà soát demo UI:
   - giữ open-set calibration là workflow chính;
   - để per-class threshold và close-word guard ở dạng advanced/experimental;
   - không dùng UI sampled result làm evidence chính thay cho GSC test100.

5. Chuẩn bị phần trình bày với thầy:
   - best accuracy vs best compact;
   - lý do PCEN và GE2E tốt;
   - vì sao SCAF collapse;
   - claim đúng khi so với EdgeSpot-4;
   - hướng KD/future work.

## 5. Kết luận ngắn

Trong tuần này, em đã hoàn thành được phần thực nghiệm chính và đã có kết quả đủ mạnh để định hình câu chuyện thesis. Cấu hình `DSCNN-L + PCEN + GE2E` là hướng accuracy tốt nhất, còn `EdgeSpotFull T4 + PCEN + GE2E` là hướng compact tốt nhất và đã đạt mức cạnh tranh với EdgeSpot-4. Em cũng đã tái cấu trúc thesis theo format gần với reference master hơn, bổ sung các công thức và phần phương pháp luận để báo cáo có tính khoa học rõ ràng hơn.
