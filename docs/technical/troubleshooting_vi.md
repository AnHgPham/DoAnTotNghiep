# Troubleshooting VI

## Git Không Có Trong PATH

**Triệu chứng:** lệnh git fail hoặc tool không tìm thấy git.

**Nguyên nhân:** Git chưa cài, hoặc terminal không load PATH.

**Kiểm tra:** chạy `git --version`.

**Sửa:** cài Git for Windows, mở terminal mới, hoặc dùng terminal đã cấu hình.

**Phòng tránh:** kiểm tra trước khi làm branch/commit/package.

## Colab Reset

**Triệu chứng:** `/content` mất dataset, training dừng, notebook reconnect.

**Nguyên nhân:** runtime reset hoặc timeout.

**Kiểm tra:** checkpoint Drive còn không, `latest.pt`/`epoch_XX.pt` có không.

**Sửa:** setup dataset session lại, resume từ Drive checkpoint.

**Phòng tránh:** `--save-every 1`, `--save-latest-every-epoch`, package kết quả lên Drive.

## Hết Colab Units

**Triệu chứng:** run dừng ở epoch giữa chừng, không thể tiếp tục GPU.

**Nguyên nhân:** hết quota.

**Sửa:** dùng checkpoint epoch cuối đã lưu, ví dụ Top500 hiện có `epoch_13.pt`; chạy tiếp khi có tài nguyên.

**Phòng tránh:** chia run ngắn, evaluate checkpoint giữa chừng.

## Checkpoint Missing

**Triệu chứng:** model card báo missing, API không load profile.

**Nguyên nhân:** checkpoint chưa tải về local hoặc đường dẫn khác manifest.

**Kiểm tra:** xem `GET /api/artifacts/status`.

**Sửa:** copy checkpoint vào `server/final_kws_artifacts_package/checkpoints/...` hoặc cập nhật path profile.

## GSC Unknown Audio Not Found

**Triệu chứng:** Open-set báo skipped unknown words.

**Nguyên nhân:** local GSC thiếu folder word hoặc chưa setup dataset.

**Sửa:** tải/setup GSC v2 trong `data/gsc_v2`.

**Phòng tránh:** trước demo open-set, enroll/test nhanh một GSC preset.

## No Enrolled Keywords

**Triệu chứng:** detect trả lỗi hoặc luôn unknown.

**Nguyên nhân:** chưa enroll hoặc vừa switch model với `clear`.

**Sửa:** dùng Enrollment tab, chọn GSC 17 known, enroll lại.

## Labels Count Mismatch

**Triệu chứng:** long audio có số expected khác số detections.

**Nguyên nhân:** segmentation/VAD bỏ sót hoặc split khác số từ.

**Sửa:** upload timing JSON để matching theo overlap; đọc missed expected cards.

## Miss Due Threshold

**Triệu chứng:** top-1 đúng nhưng status rejected, distance > threshold.

**Sửa:** calibrate threshold hoặc bật per-class threshold nếu phù hợp.

**Lưu ý:** tăng threshold quá cao sẽ tăng false accept.

## Miss Due Guard

**Triệu chứng:** distance đạt nhưng margin thấp, bị close-word guard reject.

**Sửa:** dùng calibration; chỉ tắt guard khi demo keyword-only, không dùng làm open-set balanced result.

## Low Unknown Rejection

**Triệu chứng:** Open-set FAR cao, unknown reject ACC thấp.

**Nguyên nhân:** threshold quá thoáng, guard tắt, per-class threshold quá rộng.

**Sửa:** Guard ON, per-class OFF, accept margin 0.05 là default demo hiện tại; chạy calibration.

## Top500 Cache Incomplete

**Triệu chứng:** word dirs thấp hơn 490, coverage thấp.

**Nguyên nhân:** Drive cache/session download partial.

**Sửa:** xóa cache partial, chạy session-first setup lại.

## Worker 100 Chậm Hoặc Freeze

**Triệu chứng:** DataLoader warning, training chậm hoặc đứng.

**Nguyên nhân:** số worker vượt khuyến nghị runtime.

**Sửa:** giảm về 12 hoặc 20 nếu cần. Worker 100 chỉ dành cho runbook Top500 Colab theo yêu cầu, không đổi default global.

## Drive Copy Quá Chậm

**Triệu chứng:** setup mất rất lâu khi copy WAV lên Drive.

**Sửa:** dùng session-first dataset. Chỉ lưu checkpoint/result/package lên Drive.
