# API Reference VI

Backend chính: `src/demo/api_server.py`. Tất cả endpoint demo chạy trên FastAPI tại `http://127.0.0.1:8000`.

## Nguyên Tắc Chung

- Request upload dùng `multipart/form-data`.
- Boolean gửi rõ bằng string `"true"` hoặc `"false"`.
- Response detect/open-set luôn nên đọc `settings` để biết backend thật sự dùng policy nào.
- Nếu lỗi, backend trả JSON có trường `error` hoặc HTTP error detail.

## Model Endpoints

### `GET /api/model/profiles`

Trả danh sách checkpoint có thể dùng trong demo.

Response chính:

```json
{
  "active": "top500_epoch13",
  "can_rebuild_on_switch": true,
  "profiles": [
    {
      "id": "top500_epoch13",
      "short_label": "Top500 epoch13",
      "exists": true,
      "threshold_hint": 0.3,
      "metrics": [{"label": "ACC@5%FAR", "value": "88.87%"}]
    }
  ]
}
```

### `POST /api/model/select`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `profile_id` | string | `top500_epoch13`, `microset_epoch05`, hoặc `legacy_dscnn`. |
| `enrollment_policy` | string | `rebuild` hoặc `clear`. |

Nếu `rebuild`, backend cố dựng lại enrollment từ waveform cache. Nếu không có cache, UI cần yêu cầu enroll lại.

### `GET /api/model/info`

Trả kiến trúc, checkpoint, input shape, device, số tham số nếu model đã load.

## Enrollment Endpoints

### `GET /api/enroll/status`

Trả keyword đã enroll, số mẫu và threshold từng lớp.

### `POST /api/enroll/gsc`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `words` | string | Danh sách keyword, phân tách bằng dấu phẩy hoặc whitespace. |
| `k` | int | Số mẫu GSC mỗi keyword. |

Dùng để demo nhanh bằng GSC local.

### `POST /api/enroll/mic`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `word` | string | Keyword cần enroll. |
| `audio` | file | WAV/audio mẫu. |

### `POST /api/enroll/clear`

Xóa toàn bộ enrollment trong session.

### `POST /api/enroll/save`

Lưu enrollment profile hiện tại vào `data/enroll_profiles`.

### `POST /api/enroll/load`

Load profile đã lưu.

## Detection Endpoints

### `POST /api/detect/single`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `audio` | file | Clip cần detect. |
| `threshold` | float | Global threshold. |
| `use_per_class` | bool string | Dùng threshold từng lớp. |
| `use_close_word_guard` | bool string | Bật/tắt guard top-1/top-2. |

Response:

```json
{
  "keyword": "yes",
  "detected": true,
  "distance": 0.1493,
  "threshold": 0.681,
  "margin": 0.982,
  "top_3": [{"word": "yes", "dist": 0.1493}],
  "settings": {
    "threshold": 0.3,
    "use_per_class": true,
    "close_word_guard": true,
    "accept_margin": 0.05,
    "engine": "single"
  }
}
```

### `POST /api/detect/long`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `audio` | file | File audio dài. |
| `threshold` | float | Global threshold. |
| `use_per_class` | bool string | Dùng threshold từng lớp. |
| `use_close_word_guard` | bool string | Bật/tắt close-word guard. |
| `seg_method` | string | `Energy` hoặc VAD nếu có. |
| `min_duration_ms` | int | Độ dài segment tối thiểu. |

Response gồm `duration`, `results`, `sequence`, `settings`.

### `POST /api/detect/batch`

Endpoint legacy để detect nhiều file. Không phải flow open-set chính. Giữ lại cho debug hoặc script cũ.

## Open-Set Endpoints

### `POST /api/open-set/test`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `preset` | string | `gsc_17_17` cho preset chính. |
| `known_words` | string | Candidate/enrolled words. |
| `unknown_words` | string | Words cần reject. |
| `samples_per_word` | int | Số sample mỗi word. |
| `threshold` | float | Threshold test. |
| `use_per_class` | bool string | Per-class threshold. |
| `use_close_word_guard` | bool string | Close-word guard. |
| `accept_margin` | float | Margin tối thiểu. |
| `seed` | int | Random seed. |

Metric response:

- `keyword_acc`;
- `unknown_reject_acc`;
- `false_accept_rate`;
- `false_reject_rate`;
- `open_set_acc`;
- `balanced_score`;
- `false_accepts`;
- `known_misses`.

### `POST /api/open-set/calibrate`

Thêm fields:

| Field | Type | Meaning |
|---|---|---|
| `threshold_min` | float | Ngưỡng nhỏ nhất. |
| `threshold_max` | float | Ngưỡng lớn nhất. |
| `threshold_step` | float | Bước grid. |
| `accept_margin_values` | string | Ví dụ `0.00,0.02,0.05,0.08,0.10`. |
| `use_per_class_options` | string | Ví dụ `true,false`. |

Response có `best_balanced`, `best_open_set`, `best_keyword`, `rows`.

## Artifact And Report Endpoints

### `GET /api/artifacts/status`

Trả manifest artifact local: Microset epoch05, Top500 epoch13, Top500 epoch25 historical.

### `POST /api/export/session-report`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `title` | string | Tiêu đề report. |

Response trả Markdown để copy vào report/thesis/debug note.

## Streaming

### `WebSocket /ws/stream`

Client gửi audio chunks Float32. Server trả detection events. Nếu microphone/WebSocket fail, dùng long-audio simulated streaming để demo.
