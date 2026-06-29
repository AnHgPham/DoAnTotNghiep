# UI Demo Guide VI

## Khởi Động

1. Build UI:

```bash
cd src/demo/ui
npm install
npm run build
```

2. Start backend:

```bash
python -m src.demo.api_server
```

3. Mở:

```text
http://127.0.0.1:8000/
```

Nếu `src/demo/ui/dist` tồn tại, FastAPI sẽ serve React UI mới. Nếu chưa build, backend fallback về UI legacy trong `src/demo/web`.

## Flow Demo Khuyến Nghị

### 1. Chọn Model

Ở thanh model trên cùng, chọn:

- `Microset epoch05` nếu muốn dùng mốc thesis chính;
- `Top500 epoch13` nếu muốn demo coverage rộng hơn với checkpoint local hiện có.

Khi đổi model, chọn:

- `Đổi và dựng lại enrollment` nếu đã enroll bằng waveform cache;
- `Đổi và xóa enrollment` nếu muốn enroll lại sạch.

### 2. Enroll GSC 17 Known

Vào tab `Ghi danh`, bấm preset `GSC 17 known`, chọn `k=5`, rồi enroll. Bộ này phù hợp với open-set 17/17:

```text
yes, stop, happy, bird, dog, tree, marvin, four, learn, wow, sheila, zero, down, left, right, off, three
```

### 3. Single Detect

Upload một file WAV ngắn. Kết quả cần đọc:

- predicted keyword hoặc unknown;
- L2 distance;
- threshold;
- margin;
- top-3 candidates;
- policy backend đang dùng.

### 4. Long Audio

Upload:

- audio dài;
- optional `labels.txt`;
- optional `timings.json`.

UI sẽ hiển thị:

- summary cards;
- expected timeline;
- detected timeline;
- missed expected cards;
- detection cards.

Nếu có miss, đọc reason trước khi kết luận model sai:

- no overlap;
- threshold reject;
- guard reject;
- wrong prediction;
- outside enrollment;
- segmentation skip.

### 5. Open-Set

Preset chính:

- known 17 từ đã enroll;
- unknown 17 từ cần reject;
- heldout `visual`.

Setting khuyến nghị hiện tại:

- Guard ON;
- Per-class OFF;
- accept margin 0.05;
- threshold bắt đầu 0.30, sau đó chạy calibration.

Metric cần báo cáo:

- balanced score;
- keyword ACC;
- unknown reject ACC;
- FAR;
- false reject rate.

Không dùng open-set UI sampled result thay cho GSC test100 trong thesis. Nó là demo-level evaluation.

### 6. Reports

Vào tab `Reports`, export session report để copy vào báo cáo hoặc gửi debug note. Artifact status cũng hiển thị tại đây.

## Lưu Ý Khi Demo

- Nếu port 8000 đang chạy UI cũ, restart server sau khi build React.
- Nếu model card missing, kiểm tra `server/final_kws_artifacts_package`.
- Nếu open-set báo thiếu GSC audio, kiểm tra `data/gsc_v2`.
- Nếu long audio accuracy thấp, xem missed cards và timing overlap trước.
