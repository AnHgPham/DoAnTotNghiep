# API Reference EN

Primary backend: `src/demo/api_server.py`. The demo server runs on `http://127.0.0.1:8000`.

## General Rules

- Upload endpoints use `multipart/form-data`.
- Booleans should be sent as `"true"` or `"false"`.
- Detection and open-set responses expose `settings`; the frontend should display these backend-confirmed settings rather than guessing from local UI state.

## Model

### `GET /api/model/profiles`

Returns available model profiles, active profile id, checkpoint existence, threshold hints, and headline metrics.

### `POST /api/model/select`

Fields:

| Field | Type | Meaning |
|---|---|---|
| `profile_id` | string | Model profile id, for example `top500_epoch13`. |
| `enrollment_policy` | string | `rebuild` or `clear`. |

`rebuild` reloads the encoder and tries to rebuild prototypes from cached enrollment waveforms. If no waveform cache exists, the user must enroll again.

### `GET /api/model/info`

Returns loaded architecture, checkpoint, parameter count, device, and input shape.

## Enrollment

### `GET /api/enroll/status`

Returns enrolled keywords, sample counts, thresholds, profile version, and rebuild capability.

### `POST /api/enroll/gsc`

Fields: `words`, `k`.

Enrolls a set of local GSC words for fast demos.

### `POST /api/enroll/mic`

Fields: `word`, `audio`.

Adds a user-provided audio sample for one keyword.

### `POST /api/enroll/clear`

Clears current session enrollment.

### `POST /api/enroll/save` and `POST /api/enroll/load`

Save and load local enrollment profiles.

## Detection

### `POST /api/detect/single`

Fields:

| Field | Type |
|---|---|
| `audio` | file |
| `threshold` | float |
| `use_per_class` | bool string |
| `use_close_word_guard` | bool string |

Returns predicted keyword or `unknown`, top candidates, distance, threshold, margin, and active policy settings.

### `POST /api/detect/long`

Fields:

| Field | Type |
|---|---|
| `audio` | file |
| `threshold` | float |
| `use_per_class` | bool string |
| `use_close_word_guard` | bool string |
| `seg_method` | string |
| `min_duration_ms` | int |

Returns duration, segment results, sequence, engine, and settings.

### `POST /api/detect/batch`

Legacy multi-file batch detection endpoint. It is kept for compatibility and scripts; the Open-Set UI uses `/api/open-set/test` instead.

## Open-Set

### `POST /api/open-set/test`

Fields:

| Field | Type |
|---|---|
| `preset` | string |
| `known_words` | string |
| `unknown_words` | string |
| `samples_per_word` | int |
| `threshold` | float |
| `use_per_class` | bool string |
| `use_close_word_guard` | bool string |
| `accept_margin` | float |
| `seed` | int |

Returns summary metrics, candidate words, false accepts, known misses, missing words, and settings.

### `POST /api/open-set/calibrate`

Adds calibration grid fields:

- `threshold_min`;
- `threshold_max`;
- `threshold_step`;
- `accept_margin_values`;
- `use_per_class_options`.

Returns best balanced, best open-set, best keyword, and sorted rows.

## Artifacts And Reports

### `GET /api/artifacts/status`

Returns local artifact status for Microset, Top500 epoch13, and historical Top500 epoch25 logs.

### `POST /api/export/session-report`

Returns a Markdown session report that includes model, enrollment, and artifact status.

## Streaming

### `WebSocket /ws/stream`

The browser sends Float32 audio buffers. The backend returns detection events containing keyword, distance, threshold, margin, and policy context.
