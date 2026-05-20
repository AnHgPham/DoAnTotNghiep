# Template Báo Cáo Top500 Full

File này dùng sau khi train Top500 full xong. Không sửa lẫn vào báo cáo Microset.

## 1. Run Metadata

```text
DATA PROFILE = MSWC TOP500 FULL CLIPS
RUN TAG = <run_tag>
GPU = <A100/G4/...>
DATE = <yyyy-mm-dd>
COMMIT = <git commit hash>
```

## 2. Dataset Check

```text
MSWC words = <n_word_dirs>
Train words = <n_train_words>
Val words = <n_val_words>
WAV files = <n_wav>
OPUS files = <n_opus>
max_per_word = 0
```

Điều kiện bắt buộc:

- `max_per_word = 0`.
- Không còn OPUS sau convert.
- Word dirs khoảng 500.
- Không dùng checkpoint Microset để resume.

## 3. Training Summary

```text
Model = EdgeSpotFull T4
Loss = SCAF+GE2E
Epochs = <n>
Episodes per epoch = <n>
Checkpoint selected by = GSC-dev
Selected checkpoint = <path>
```

## 4. Dev Results

Dán result JSON hoặc chạy:

```bash
python scripts/make_result_table.py \
  --results-dir <drive-or-local-results-dir> \
  --out-dir reports/top500_full \
  --profile top500_full
```

| Model | Split | Runs | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | AUC | EER | Keyword ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EdgeSpotFull T4 SCAF+GE2E | dev | 30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 5. Final Test Results

Chỉ chạy test100 sau khi đã khóa checkpoint bằng dev.

| Model | Split | Runs | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | AUC | EER | Keyword ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EdgeSpotFull T4 SCAF+GE2E | test | 100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Error Analysis

```bash
python scripts/analyze_result_errors.py \
  --result-json <selected_test100_json> \
  --out-dir reports/top500_full
```

Cần điền:

- weakest keywords;
- top confusion pairs;
- per-word false accept rate;
- DET curve path.

## 7. Claim Được Phép

Được claim:

```text
MSWC English Top500 full-clips training, evaluated on GSC v2 gsc_edgespot_exact.
```

Không claim:

```text
Full MSWC 38k words.
Exact EdgeSpot paper reproduction.
KD reproduction, nếu chưa chạy teacher KD.
Streaming benchmark, nếu chưa đo streaming.
```
