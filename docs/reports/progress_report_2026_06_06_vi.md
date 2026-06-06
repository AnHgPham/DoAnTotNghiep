# Báo cáo tiến độ KWS - 2026-06-06

Phạm vi: tổng hợp trạng thái server ict6 (đã verify trực tiếp) và run Colab cap620 FLAC (~3M clip) mới hoàn tất. Tất cả số liệu trích từ log thật.

## 1. Trạng thái hệ thống

- Server ict6 (`ictserver6`): không còn job train/eval KWS nào chạy. GPU 5/6/7 trống.
- Hai nhánh recheck Top500 đã đóng:
  - `top500_full_recheck_e20_ep200_edgespot_resume`: train=ok, dev=ok, test=ok (xong 2026-06-05 04:58:56).
  - `top500_full_recheck_e20_ep200_far1` (eval best.pt @ FAR 1%): dev=ok, test=ok (xong 2026-06-06 ~01:10).
- Colab A100: run `colab_mswc_heavy_flac_target3000000_20260605_175317` đã chạy hết 2 stage (xong 2026-06-05 23:14).

## 2. Run mới: cap620 FLAC (~3.04M clip) - kết quả tốt nhất hiện tại

Thiết lập:

- Dữ liệu: MSWC English capped `max_per_word=620` -> 2,989,780 file train, 52,399 file val, 37,387 train words / 763 val words. Định dạng FLAC.
- Lịch train: 20 epoch x 1000 episode/epoch, episodic 30 lớp x 10 mẫu, loss GE2E, scheduler CosineAnnealingWarmRestarts.
- Chọn checkpoint: GSC-dev `gsc_edgespot_exact`, 5 run, k=10, theo ACC@1%FAR.
- Eval cuối: GSC-dev30 và GSC-test100, target FAR = 5% (nên "Open-set ACC" trong log = ACC@5%FAR).

### DSCNN-L + PCEN + GE2E (412,900 tham số)

- Best checkpoint: epoch 17 (GSC-dev ACC@1%FAR = 0.8769).
- GSC-dev30 @5%FAR: ACC 89.26, KW-ACC 91.07, AUC 94.41, EER 12.24, F1 81.43, FRR@5%FAR 20.70.
- GSC-test100 @5%FAR: ACC 88.56, KW-ACC 91.52, AUC 94.04, EER 12.53, F1 81.02, FRR@5%FAR 23.30.

### EdgeSpotFull T4 + PCEN + GE2E (130,598 tham số)

- Best checkpoint: epoch 11 (GSC-dev ACC@1%FAR = 0.8543).
- GSC-dev30 @5%FAR: ACC 86.81, KW-ACC 86.93, AUC 92.36, EER 15.49, F1 76.92, FRR@5%FAR 27.84.
- GSC-test100 @5%FAR: ACC 86.01, KW-ACC 85.90, AUC 91.34, EER 16.64, F1 75.38, FRR@5%FAR 30.06.

## 3. So sánh theo quy mô dữ liệu (GSC-test100 ACC@5%FAR)

| Quy mô dữ liệu (cap/từ) | ~Số clip | DSCNN-L + PCEN + GE2E | EdgeSpotFull T4 + PCEN + GE2E |
|---|---:|---:|---:|
| cap20 (manifest20) | ~0.53M | 86.05 | 83.06 |
| cap50 (manifest50) | ~0.94M | 84.68 | 82.24 |
| Top500 full recheck | ~3.3M (500 từ) | 86.56 | 86.18* |
| cap620 FLAC | ~2.99M (38k từ) | 88.56 | 86.01 |

`*` Nhánh Top500 EdgeSpot dùng SCAF+GE2E, không phải GE2E thuần.

Quan sát:

- Tăng dữ liệu lên ~3M clip qua nhiều từ (cap620) cho kết quả tốt nhất cho pipeline GE2E:
  - DSCNN: 86.05 -> 88.56 (+2.51 pp) so với cap20.
  - EdgeSpot: 83.06 -> 86.01 (+2.95 pp) so với cap20.
- cap50 thấp hơn cap20 vì cùng lịch 20ep x 200ep nhưng nhiều biến thể hơn; cap620 vượt nhờ episodes/epoch tăng từ 200 lên 1000 và phủ rộng 38k từ.
- DSCNN-L vẫn dẫn EdgeSpotFull T4 về accuracy thô (+2.55 pp ACC@5%FAR test100 trên cap620), nhưng EdgeSpot nhỏ hơn ~3.2 lần (130k vs 413k tham số).

## 4. Mốc Top500 (FAR 1%) - bằng chứng cũ vẫn giữ

- EdgeSpotFull T4 + PCEN + SCAF+GE2E, checkpoint epoch13 legacy, test100: ACC@1%FAR 85.62, ACC@5%FAR 88.79, AUC 95.34, EER 11.51, F1 82.45 (vẫn là mốc 1%FAR cao nhất).
- EdgeSpot SCAF+GE2E train-new 20ep (resume), test100 @5%FAR: ACC 86.18, AUC 92.73, EER 15.11, F1 77.45; @1%FAR: ACC 83.50, FRR@1%FAR 50.03.

## 5. Diễn giải cho thesis

- Chốt thông điệp dữ liệu: tăng quy mô MSWC (cap620, ~3M clip, 38k từ) cải thiện rõ cả hai mô hình -> ủng hộ luận điểm "more data + GE2E + PCEN giúp few-shot open-set KWS tốt hơn".
- Hai ứng viên rõ ràng:
  - Accuracy cao nhất: DSCNN-L + PCEN + GE2E (cap620), test100 ACC@5%FAR 88.56, F1 81.02.
  - Edge/deploy: EdgeSpotFull T4 + PCEN + GE2E (cap620), test100 ACC@5%FAR 86.01, F1 75.38, chỉ 130k tham số.
- Microset vẫn là mốc kiến trúc đã khóa (EdgeSpotFull T4 + SCAF+GE2E). cap620 là bằng chứng scale-up bổ trợ, không phải để phủ định Microset.

## 6. Lỗ hổng số liệu cần bổ sung

- cap620 mới có test100 ở ACC@5%FAR. Để so trực tiếp với Top500 epoch13 (85.62 @1%FAR), cần chạy lại eval `best.pt` của cap620 với `--target-far 0.01` trên cả DSCNN và EdgeSpot.
- Chưa có nhánh SCAF+GE2E cho cap620; nếu muốn so loss công bằng ở quy mô lớn, cần thêm cap620 + SCAF+GE2E.
- Lỗi log đã biết: phần summary in `words_with_audio=0`/`audio_files=0` do bộ đếm summary không đếm `.flac`; không ảnh hưởng train (đã nạp đủ 2.99M file).
- Artifact cap620 hiện chỉ ở Drive Colab; cần copy JSON/DET-curve về `reports/` để khóa bằng chứng local.

## 7. Việc nên làm tiếp

1. Eval lại cap620 `best.pt` (DSCNN + EdgeSpot) ở `--target-far 0.01` để có ACC@1%FAR test100 so sánh với Top500 epoch13.
2. Copy artifact cap620 (results JSON, DET PNG, checkpoint best.pt) từ Drive về `reports/colab_mswc_cap620_flac/`.
3. (Tuỳ chọn) chạy cap620 + SCAF+GE2E để hoàn thiện ma trận loss ở quy mô lớn.
4. Cập nhật bảng kết quả thesis với cột "cap620 ~3M".
