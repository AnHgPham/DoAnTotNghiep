# Email Gửi Thầy - 22/05/2026

Thầy ơi,

Tuần này em đã tập trung thử nghiệm lại các cấu hình cho bài toán few-shot open-set keyword spotting. Em chạy các hướng trên MSWC Microset trước để chọn cấu hình ổn định, gồm baseline DSCNN-L + Triplet, EdgeSpotFull T4 + SCAF, và EdgeSpotFull T4 + SCAF+GE2E.

Qua các thử nghiệm Microset, cấu hình **EdgeSpotFull T4 + SCAF+GE2E** đang cho kết quả tốt nhất trong các cấu hình em đã chạy. Kết quả đã khóa hiện tại trên GSC test100 là:

- ACC@5%FAR: 86.12%;
- Keyword ACC: 77.66%;
- F1: 82.41%;
- AUC: 95.61%;
- EER: 11.54%.

Từ kết quả đó, em dùng cùng hướng model/loss để train tiếp trên bộ Top500. Top500 cho tín hiệu tốt hơn ở các checkpoint hiện có, nhưng phần này bị ảnh hưởng bởi Colab/session/units. Lần chạy trước có log tốt nhưng chưa kịp tải/package artifact đầy đủ trước khi session bị mất, nên em không dùng nó làm kết quả chính. Lần chạy sau em đã sửa pipeline để lưu checkpoint mỗi epoch vào Drive, nhưng bị dừng ở epoch 13 do hết tài nguyên. Checkpoint chắc chắn em đang có hiện tại là `epoch_13.pt`; kết quả dev30 của checkpoint này đạt ACC@5%FAR khoảng 88.87%.

Ngoài phần training, em cũng đang hoàn thiện demo web để giải thích rõ kết quả hơn:

- chọn model Microset/Top500;
- enroll keyword từ GSC;
- test single audio;
- test audio dài kèm file label/timing;
- hiển thị lý do miss;
- open-set test 17 từ known và 17 từ unknown;
- calibration threshold/guard để cân bằng giữa nhận đúng keyword và từ chối unknown.

Hiện tại em sẽ dùng Microset làm mốc kết quả chính trong thesis, còn Top500 epoch13 dùng cho demo và báo cáo tiến độ mở rộng. Khi có thêm tài nguyên Colab/GPU, em sẽ chạy tiếp Top500 để có checkpoint/test100 đầy đủ hơn.

Em cảm ơn thầy.
