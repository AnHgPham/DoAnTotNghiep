# Reproducibility Checklist VI

## Trước Khi Train

- [ ] Xác nhận branch/code đúng.
- [ ] Xác nhận config YAML đúng dataset/model/loss.
- [ ] Xác nhận dataset path tồn tại.
- [ ] Với Microset: dùng official CSV manifest, không scan folder trực tiếp.
- [ ] Với Top500: xác nhận 450 train words + 50 val words.
- [ ] Xác nhận DEMAND nếu bật noise augmentation.
- [ ] Xác nhận output checkpoint nằm trên Drive nếu chạy Colab.

## Khi Train

- [ ] Ghi lại command đầy đủ.
- [ ] Dùng `--save-every 1` cho Top500.
- [ ] Dùng `--save-latest-every-epoch` cho Top500.
- [ ] Theo dõi loss, GE2E acc, validation AUC/ACC.
- [ ] Theo dõi GSC-dev nếu dùng checkpoint selection.
- [ ] Không bỏ qua DataLoader warning nếu runtime chậm/freeze.

## Sau Khi Train

- [ ] Kiểm tra `epoch_XX.pt` tồn tại.
- [ ] Kiểm tra `latest.pt` tồn tại.
- [ ] Kiểm tra `best.pt` nếu có selection.
- [ ] Chạy dev30 hoặc test100 phù hợp.
- [ ] Lưu result JSON.
- [ ] Lưu DET curve.
- [ ] Package artifact lên Drive trước khi download.

## Local Artifact

- [ ] Copy checkpoint vào `server`.
- [ ] Copy result JSON vào `server`.
- [ ] Copy DET curve nếu có.
- [ ] Chạy `python scripts/make_project_status.py`.
- [ ] Mở `reports/project_status/claim_matrix.md`.
- [ ] Chỉ claim những kết quả có evidence rõ.

## Demo

- [ ] Start FastAPI.
- [ ] React UI đã build bằng `npm run build`.
- [ ] Model card hiển thị ready.
- [ ] Enroll GSC 17 known.
- [ ] Run single detect.
- [ ] Run long audio với label/timing.
- [ ] Run open-set 17/17.
- [ ] Run calibration.
- [ ] Export session report.

## Thesis/Report

- [ ] Microset ghi là mốc chính.
- [ ] Top500 epoch13 ghi là artifact local/sơ bộ.
- [ ] Top500 epoch25 chỉ ghi historical nếu thiếu artifact.
- [ ] Open-set UI ghi là sampled demo-level evaluation.
- [ ] Không claim reproduction đầy đủ EdgeSpot paper.
