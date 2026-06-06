# Báo cáo tuần - Đồ án KWS Few-Shot Open-Set (cập nhật 2026-06-06)

Người thực hiện: (sinh viên). Hệ thống tính toán: server ict6 (Tesla K80, CUDA 10.2) và Google Colab Pro+ (A100-40GB).

Tất cả số liệu trong báo cáo lấy từ log/JSON thật. Giao thức đánh giá thống nhất: `gsc_edgespot_exact` trên Google Speech Commands v2, k=10 (10-shot), test100 = 100 lần lặp, mỗi lần 11 từ khóa + 25 từ lạ + lớp `_silence_` thật.

## 1. Tóm tắt tuần

Tuần này tập trung vào ba việc: (1) mở rộng huấn luyện trên MSWC ở nhiều quy mô dữ liệu để kiểm tra giới hạn, (2) hoàn tất và kiểm chứng các thí nghiệm Top500 trên server, (3) phân tích vì sao kết quả bão hòa khi tăng dữ liệu và đề xuất hướng đi tiếp.

Kết quả nổi bật:

- Mô hình chính xác nhất hiện tại: `DSCNN-L + PCEN + GE2E` huấn luyện trên ~3M clip MSWC, đạt GSC-test100 ACC@5%FAR = 88.56%, AUC = 94.04%, F1 = 81.02%.
- Mô hình gọn cho thiết bị: `EdgeSpotFull T4 + PCEN + GE2E` (130k tham số, nhỏ hơn ~3.2 lần) đạt ACC@5%FAR = 86.01%.
- Phát hiện quan trọng: tăng dữ liệu từ ~2M lên ~3M clip gần như không cải thiện (bão hòa). Giới hạn không nằm ở số lượng clip mà ở ngân sách huấn luyện episodic, dung lượng mô hình, và trần chuyển giao MSWC -> GSC.

## 2. Công việc đã thực hiện

### 2.1. Pipeline dữ liệu

- Hoàn thiện quy trình MSWC trên Colab dùng FLAC thay vì WAV để tiết kiệm đĩa: tải -> trích xuất theo cap -> chuyển OPUS sang FLAC -> xoá OPUS -> tạo manifest. Trước đây bản WAV đầy đủ không khả thi vì tràn đĩa 236GB.
- Bổ sung công cụ ước lượng cap (`data/estimate_mswc_cap.py`) để chọn số clip/từ đạt mục tiêu, và bộ chuyển OPUS->FLAC song song (`data/convert_opus_to_flac.py`).
- ưđể kiểm soát chi phí.

### 2.2. Thí nghiệm trên server ict6

- Hoàn tất recheck Top500 (500 từ phổ biến, ~3.3M clip) với 3 nhánh, đã kiểm chứng trực tiếp trên máy (train/dev/test đều `ok`):
  - Re-eval checkpoint cũ EdgeSpotFull T4 + SCAF+GE2E (epoch13): test100 ACC@1%FAR = 85.62%, ACC@5%FAR = 88.79%, AUC = 95.34%, EER = 11.51%, F1 = 82.45%.
  - Train mới DSCNN-L + PCEN + GE2E: test100 ACC@5%FAR = 86.56%, AUC = 93.17%, EER = 14.00%.
  - Train mới EdgeSpotFull T4 + SCAF+GE2E (20 epoch): test100 ACC@5%FAR = 86.18%; đánh giá thêm tại 1%FAR: ACC@1%FAR = 83.50%, FRR@1%FAR = 50.03%.
- Trước đó đã chạy shortlist trên server ở hai quy mô cap20 và cap50 để xác nhận thứ hạng mô hình.

### 2.3. Thí nghiệm trên Colab A100

- cap220 FLAC (~2.05M clip): DSCNN-L test100 ACC@5%FAR = 88.23%; EdgeSpotFull T4 = 86.03%.
- cap620 FLAC (~2.99M clip, 20 epoch x 1000 episode): DSCNN-L test100 ACC@5%FAR = 88.56%; EdgeSpotFull T4 = 86.01%. Đây là kết quả cao nhất hiện tại cho nhánh GE2E.

### 2.4. Phân tích, demo và tài liệu

- Tổng hợp toàn bộ số liệu thành bảng so sánh theo quy mô dữ liệu (mục 3).
- Demo few-shot open-set (FastAPI + giao diện React, song ngữ): enroll keyword, nhận diện audio ngắn/dài, streaming, kiểm tra open-set.
- Duy trì bản nháp thesis, tài liệu kỹ thuật và báo cáo.

## 3. Bảng kết quả tổng hợp (GSC test100)


| Dữ liệu (clip / số từ) | Mô hình                              | ACC@5%FAR | AUC   | EER   | F1    | Tham số |
| ---------------------- | ------------------------------------ | --------- | ----- | ----- | ----- | ------- |
| Microset (31 từ)       | DSCNN-L Triplet                      | 80.54     | 91.22 | 18.22 | 73.30 | 413k    |
| Microset (31 từ)       | EdgeSpotFull T4 SCAF+GE2E (mốc khóa) | 86.12     | 95.61 | 11.54 | 82.41 | 131k    |
| cap20 (~0.53M / 38k)   | DSCNN-L PCEN GE2E                    | 86.05     | 91.57 | 16.25 | 75.90 | 413k    |
| cap20 (~0.53M / 38k)   | EdgeSpotFull T4 PCEN GE2E            | 83.06     | 87.22 | 20.40 | 70.46 | 131k    |
| cap50 (~0.94M / 38k)   | DSCNN-L PCEN GE2E                    | 84.68     | 90.45 | 17.42 | 74.34 | 413k    |
| cap50 (~0.94M / 38k)   | EdgeSpotFull T4 PCEN GE2E            | 82.24     | 87.74 | 20.19 | 70.73 | 131k    |
| cap220 (~2.05M / 38k)  | DSCNN-L PCEN GE2E                    | 88.23     | 93.87 | 12.78 | 80.67 | 413k    |
| cap220 (~2.05M / 38k)  | EdgeSpotFull T4 PCEN GE2E            | 86.03     | 91.31 | 16.47 | 75.61 | 131k    |
| cap620 (~2.99M / 38k)  | DSCNN-L PCEN GE2E                    | 88.56     | 94.04 | 12.53 | 81.02 | 413k    |
| cap620 (~2.99M / 38k)  | EdgeSpotFull T4 PCEN GE2E            | 86.01     | 91.34 | 16.64 | 75.38 | 131k    |
| Top500 (500 từ)        | DSCNN-L PCEN GE2E                    | 86.56     | 93.17 | 14.00 | 78.97 | 413k    |
| Top500 (500 từ)        | EdgeSpotFull T4 SCAF+GE2E (epoch13)  | 88.79     | 95.34 | 11.51 | 82.45 | 131k    |


## 4. Phân tích: vì sao tăng dữ liệu lại bão hòa

Quan sát định lượng (GSC test100, ACC@5%FAR):

- cap50 -> cap220 (~0.94M -> ~2.05M): DSCNN tăng mạnh +3.55 điểm phần trăm (84.68 -> 88.23).
- cap220 -> cap620 (~2.05M -> ~2.99M): DSCNN gần như đứng yên +0.33 điểm (88.23 -> 88.56); EdgeSpot không tăng (86.03 -> 86.01) dù cap620 train nhiều hơn (1000 vs 800 episode).

Nói cách khác, đường cong "accuracy theo lượng dữ liệu" tăng nhanh đến khoảng 2 triệu clip rồi đi ngang. Bốn nguyên nhân chính:

1. Ngân sách huấn luyện episodic là yếu tố giới hạn, không phải tổng số clip. Mỗi epoch chỉ lấy mẫu một số episode cố định (ví dụ 1000 episode x 30 lớp x 10 mẫu = 300,000 mẫu/epoch) bất kể bể dữ liệu lớn cỡ nào. Khi bể đã đủ đa dạng, thêm clip chỉ làm tăng phần dữ liệu mà sampler không kịp khai thác trong lịch train cố định.
2. Tăng cap là thêm clip của CÙNG các từ, không thêm từ mới. Số từ vẫn là 37,387 train words ở cả cap220 lẫn cap620; cap chỉ tăng số mẫu cho các từ phổ biến. Few-shot KWS hưởng lợi từ đa dạng từ và người nói hơn là từ việc lặp lại nhiều mẫu của cùng một từ.
3. Trần chuyển giao MSWC -> GSC. Mô hình học embedding trên MSWC nhưng được đánh giá trên 35 từ cố định của GSC. Khi embedding đã đủ tốt để tách 35 từ này theo cơ chế few-shot, việc thêm dữ liệu MSWC gần như không cải thiện thêm trên tập đích, vì giới hạn nằm ở khoảng cách miền MSWC/GSC chứ không ở lượng dữ liệu nguồn.
4. Dung lượng mô hình và tối ưu hội tụ. DSCNN-L (413k tham số) và EdgeSpotFull T4 (131k) có trần biểu diễn riêng; với cùng lịch lr và 20-25 epoch, mô hình hội tụ về vùng nghiệm tương tự nên thêm dữ liệu chỉ làm mịn nhẹ chứ không phá trần.

Hệ quả thực tế: để vượt mức ~88% ACC@5%FAR thì giải pháp KHÔNG phải là thêm clip, mà là một trong các hướng:

- Tăng ngân sách huấn luyện (nhiều episode/epoch hơn, nhiều epoch hơn) và/hoặc lịch học tốt hơn.
- Dùng loss mạnh hơn: SCAF+GE2E đã thắng ở Microset và Top500 nhưng chưa thử ở quy mô full-MSWC. Đây là thí nghiệm có giá trị nhất còn lại.
- Tăng đa dạng từ/người nói và khai thác hard-negative trong episodic sampling.
- Tăng dung lượng mô hình nếu bài toán cho phép (đánh đổi với mục tiêu gọn cho thiết bị).

## 5. Kết luận

- Hai ứng viên rõ ràng: DSCNN-L (chính xác cao nhất) và EdgeSpotFull T4 (gọn cho thiết bị, chỉ 131k tham số nhưng đạt AUC/EER tốt nhất khi dùng SCAF+GE2E).
- Microset vẫn là mốc kiến trúc đã khóa (EdgeSpotFull T4 + SCAF+GE2E, AUC 95.61, EER 11.54). Các run quy mô lớn là bằng chứng mở rộng bổ trợ.
- Đã xác nhận thực nghiệm rằng tăng dữ liệu không phải là hướng cải thiện tiếp theo; cần đổi sang loss/lịch huấn luyện.

## 6. Kế hoạch tuần tới (dùng cả server và Colab)

1. Đánh giá lại checkpoint cap620 ở 1%FAR (chỉ cần GSC, chạy nhanh) để có số liệu ACC@1%FAR cuối cùng so sánh với Top500 epoch13.
2. Thí nghiệm trọng tâm: huấn luyện `EdgeSpotFull T4 + PCEN + SCAF+GE2E` và `DSCNN-L + PCEN + SCAF+GE2E` ở quy mô ~2M clip (cap220), lịch dài hơn (25 epoch), đánh giá tại cả 1% và 5% FAR. Mục tiêu: kiểm tra liệu SCAF+GE2E có vượt GE2E ở quy mô lớn không.
  - Phân công tài nguyên: Colab A100 cho data-prep nặng và train nhanh; server ict6 cho các run dài chạy nền song song.
3. Chốt một bảng kết quả thesis thống nhất kèm đường DET, và lưu artifact về kho local.

## 7. Lưu ý trung thực số liệu

- Microset: kết quả chính thức đã khóa, không tinh chỉnh theo test100.
- Top500 epoch13: có checkpoint local, tái lập được. Top500 epoch25 chỉ tồn tại trong log Colab cũ (mất checkpoint), nên chỉ ghi là run đã hoàn tất theo log, không dùng làm bằng chứng artifact.
- cap220/cap620: số liệu lấy từ log Colab; phần test100 hiện đo tại 5%FAR, số 1%FAR cuối đang được bổ sung.

