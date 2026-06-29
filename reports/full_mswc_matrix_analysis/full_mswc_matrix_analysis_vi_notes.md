# Phân tích ma trận Full MSWC phase-1

## Phạm vi và cách đọc kết quả

- Đây là **phase-1 screening** để so sánh xu hướng giữa các tổ hợp model/frontend/loss. Tiêu chí chọn checkpoint là **GSC-dev ACC@1%FAR**; chưa được xem là kết luận cuối trên GSC-test100.

- Thiết lập train: Full MSWC English, manifest giới hạn tối đa 20 file/từ, 5 epochs, 150 episodes/epoch, 30-way x 10 samples.

- Thiết lập đánh giá trong lúc train: GSC-dev, 10-shot, 3 repeated runs/checkpoint. Các số dưới đây lấy từ log thật trong thư mục `raw/`.


## Ranking chính

- Tổ hợp tốt nhất trong phase-1: **DSCNN-L + PCEN + GE2E**, ACC@1%FAR=76.67%, ACC@5%FAR=79.98%, F1=67.68%.

- Tổ hợp EdgeSpot tốt nhất: **EdgeSpotFull T4 + PCEN + GE2E**, ACC@1%FAR=72.94%, ACC@5%FAR=73.35%, F1=57.29%.

- Số tham số đọc từ log: DSCNN-L khoảng **412,900** tham số, EdgeSpotFull T4 khoảng **130,598** tham số. Vì vậy cần đọc kết quả EdgeSpot cùng với yếu tố kích thước/triển khai, không chỉ raw accuracy.

| combo                            | best_epoch | acc1_far | acc5_far | frr5_far | f1_mean |
| -------------------------------- | ---------- | -------- | -------- | -------- | ------- |
| DSCNN-L + PCEN + GE2E            | 5          | 76.67%   | 79.98%   | 48.97%   | 67.68%  |
| EdgeSpotFull T4 + PCEN + GE2E    | 4          | 72.94%   | 73.35%   | 71.03%   | 57.29%  |
| DSCNN-L + MFCC + GE2E            | 5          | 72.30%   | 73.78%   | 66.24%   | 60.69%  |
| DSCNN-L + PCEN + Triplet         | 5          | 72.24%   | 72.67%   | 74.55%   | 54.98%  |
| EdgeSpotFull T4 + PCEN + Triplet | 5          | 70.76%   | 68.81%   | 87.27%   | 44.36%  |


## Bảng delta quan trọng

| So sánh                                     | Delta ACC@1%FAR | Delta F1  | Ý nghĩa                                                        |
| ------------------------------------------- | --------------- | --------- | -------------------------------------------------------------- |
| DSCNN-L: PCEN vs MFCC khi dùng GE2E         | +4.37 pp        | +6.99 pp  | PCEN giúp rõ nhất khi loss đã khớp với prototype.              |
| EdgeSpotFull T4: PCEN vs MFCC khi dùng GE2E | +3.63 pp        | +17.91 pp | MFCC làm mất một phần lợi thế time-frequency của EdgeSpot.     |
| DSCNN-L + PCEN: GE2E vs Triplet             | +4.43 pp        | +12.70 pp | GE2E ăn khớp hơn với inference theo centroid.                  |
| EdgeSpotFull T4 + PCEN: GE2E vs Triplet     | +2.18 pp        | +12.93 pp | GE2E cũng là loss mạnh nhất trong nhánh EdgeSpot.              |
| DSCNN-L + PCEN: SCAF+GE2E vs GE2E           | -6.56 pp        | -29.81 pp | Hybrid hiện tại bị SCAF kéo xấu, cần tune trọng số/lịch train. |
| EdgeSpotFull T4 + PCEN: SCAF+GE2E vs GE2E   | -3.70 pp        | -19.07 pp | Trong full-MSWC phase-1, GE2E đơn lẻ tốt hơn hybrid.           |


## Diễn giải theo hướng nghiên cứu

- **GE2E tốt nhất trong phase-1 vì objective khớp với cơ chế inference.** Hệ thống few-shot sau cùng dùng enrollment samples để tính prototype/centroid, rồi so khoảng cách L2 của query với các prototype. GE2E cũng huấn luyện embedding theo hướng gần centroid đúng lớp và xa centroid lớp khác, nên tín hiệu train và cách test nhất quán hơn Triplet.

- **PCEN tốt hơn MFCC khi đi với GE2E**, đặc biệt ở DSCNN-L. PCEN giữ biểu diễn mel/time-frequency giàu hơn và có cơ chế nén động, chuẩn hóa biên độ theo thời gian. Điều này hợp với full MSWC vì dữ liệu nhiều người nói, mức âm lượng và nhiễu khác nhau. MFCC nén phổ mạnh hơn, hữu ích cho baseline cổ điển nhưng có thể làm mất chi tiết mà CNN/EdgeSpot cần khai thác.

- **SCAF và SCAF+GE2E chưa thắng trong full-MSWC phase-1.** Lý do hợp lý nhất không phải là SCAF vô dụng, mà là cấu hình phase-1 quá ngắn và quá rộng: 37k+ train words, mỗi word tối đa 20 file, chỉ 5 epochs. SCAF là margin/angular classification loss, thường cần lịch train ổn định hơn, class sampling/tuning tốt hơn, và cần cân bằng trọng số khi ghép với GE2E. Trong log hiện tại, SCAF+GE2E bị thấp hơn GE2E đơn lẻ, cho thấy trọng số hybrid hiện tại có thể làm embedding lệch khỏi tiêu chí centroid của bước inference.

- **EdgeSpotFull T4 + MFCC là ablation kiểm tra, không phải cấu hình chính.** EdgeSpot được thiết kế để tận dụng biểu diễn time-frequency giàu hơn; khi đổi sang MFCC, lợi thế kiến trúc giảm rõ. Nhánh EdgeSpot tốt nhất vẫn là **EdgeSpotFull T4 + PCEN + GE2E**.

- **DSCNN-L + PCEN + GE2E thắng phase-1 nhưng chưa tự động thay kết luận Microset.** Microset và Top500/Full-MSWC khác nhau về dữ liệu, số từ, sampling và độ dài train. Do đó kết quả này nên dùng để chọn shortlist cho train dài hơn/test100, không dùng để phủ định mốc Microset nếu chưa chạy cùng protocol cuối.


## Cách dùng biểu đồ trong báo cáo

- `all_metric_heatmap.png`: đặt nhiều giá trị vào cùng một biểu đồ; phù hợp nhất để nhìn tổng thể ACC@1%FAR, ACC@5%FAR, AUC, open-set ACC, keyword ACC, F1, 1-EER và 1-FRR@5%.

- `research_metric_dashboard.png`: dashboard 4 metric chính; dễ trình bày khi muốn so sánh nhanh ACC@1%FAR, ACC@5%FAR, recall tại 5% FAR và F1.

- `acc1far_interaction_heatmap.png`: biểu đồ tương tác model/frontend/loss; dùng để giải thích vì sao GE2E và PCEN nổi bật hơn trong phase-1.

- `key_effect_delta_bars.png`: biểu đồ delta; dùng để nói rõ thêm PCEN hơn MFCC bao nhiêu, GE2E hơn Triplet bao nhiêu, và SCAF+GE2E đang kém GE2E bao nhiêu.

- `det_curve_summary.md`: bảng DET summary theo EER, dùng để đọc trực tiếp AUC/EER/FRR tại các operating points 1% FAR và 5% FAR.

- `det_summary_heatmap.png` và `det_operating_points.png`: hai hình DET summary; hình operating-points nối hai điểm FAR=1% và FAR=5% vì log phase-1 không lưu raw scores để dựng đường DET liên tục.


## Kết luận thực nghiệm phase-1

Cấu hình nên ưu tiên train/evaluate tiếp là **DSCNN-L + PCEN + GE2E** và **EdgeSpotFull T4 + PCEN + GE2E**. Nếu mục tiêu luận văn là hệ thống gọn cho edge/device, EdgeSpotFull T4 vẫn có giá trị vì nhỏ hơn đáng kể; nếu mục tiêu là điểm dev phase-1 cao nhất, DSCNN-L + PCEN + GE2E đang dẫn đầu. Bước chuẩn nghiên cứu tiếp theo là train dài hơn cho shortlist, sau đó chạy GSC-test100 và báo cáo mean/std thay vì chỉ dùng phase-1 dev.


## File đầu ra

- `matrix_best_epoch_metrics.csv/md`: bảng metric tốt nhất theo từng combo.

- `matrix_effect_deltas.csv/md`: bảng delta giữa các lựa chọn frontend/loss/model.

- `det_curve_summary.csv/md`: bảng DET summary theo EER và FRR operating points.

- `acc1far_ranked_bar.png`: ranked single-metric comparison.

- `acc1far_interaction_heatmap.png`: model/frontend/loss interaction.

- `all_metric_heatmap.png`: all comparable metrics in one view.

- `loss_effect_lines.png`: loss behavior under each model/frontend.

- `research_metric_dashboard.png`: 4 metric chính trong cùng dashboard.

- `key_effect_delta_bars.png`: delta ACC@1%FAR cho các ablation chính.

- `det_summary_heatmap.png`: bảng DET summary dạng ảnh.

- `det_operating_points.png`: hình operating-point DET tại FAR=1% và FAR=5%.
