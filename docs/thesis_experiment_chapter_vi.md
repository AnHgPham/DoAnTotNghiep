# Chương Thực Nghiệm - Few-Shot Open-Set Keyword Spotting

## 1. Mục tiêu thực nghiệm

Mục tiêu của phần thực nghiệm là đánh giá pipeline few-shot open-set keyword spotting trong điều kiện người dùng chỉ cung cấp một số mẫu enrollment nhỏ cho mỗi từ khóa. Hệ thống cần nhận diện đúng các từ khóa đã đăng ký, đồng thời từ chối các âm thanh không thuộc tập từ khóa ở mức false alarm cố định.

Trong giai đoạn hiện tại, thí nghiệm được khóa trên profile **MSWC Microset English official CSV split** để tiết kiệm Colab units và dung lượng ổ đĩa. Kết quả này không được claim là Top500 full, full MSWC, hay reproduction đầy đủ của EdgeSpot paper.

## 2. Dataset và split

Dataset train là MSWC Microset English chính thức của MLCommons. Microset có 31 keyword, giới hạn khoảng 6000 clips mỗi keyword trong bản gốc, và đi kèm các CSV split chính thức:

- `en_train.csv`: 69,868 file train.
- `en_dev.csv`: 13,114 file dev/validation.
- `en_test.csv`: 13,117 file test/eval.
- Tổng sau convert: 96,099 WAV.

Microset là sample-level split: cùng 31 keyword xuất hiện ở train/dev/test nhưng file audio khác nhau. Vì vậy pipeline không quét trực tiếp `clips/<word>`, mà sinh và dùng file manifest:

- `train_files.json`;
- `val_files.json`;
- `eval_files.json`.

Cách này tránh leakage do vô tình dùng toàn bộ folder keyword trong lúc train.

## 3. Protocol đánh giá

Evaluation dùng Google Speech Commands v2 theo protocol `gsc_edgespot_exact`.

Thiết lập chính:

- 10-shot support cho mỗi keyword;
- positive set gồm 10 command words và true `_silence_`;
- negative/open-set gồm 25 non-command speech words;
- classifier là `OpenNCMClassifier`;
- scoring dùng L2 distance;
- checkpoint selection dựa trên GSC-dev;
- test100 chỉ dùng để báo cáo cuối, không dùng để tune.

Các metric chính:

- `ACC@1% FAR`;
- `ACC@5% FAR`;
- `FRR@5% FAR`;
- `AUC`;
- `EER`;
- `Keyword ACC`;
- `F1`.

## 4. Baseline và proposed method

Baseline là **DSCNN-L + MFCC + Triplet loss**. Baseline này đại diện cho pipeline ban đầu của project: encoder DSCNN-L tạo embedding, sau đó so khớp prototype bằng khoảng cách L2.

Proposed method hiện tại là **EdgeSpotFull T4 + SCAF+GE2E**. Mô hình này dùng input mel 40x101, trainable PCEN, backbone EdgeSpot-style, embedding 64-D, kết hợp Sub-center ArcFace và GE2E. SCAF giúp tách class trong embedding space, còn GE2E làm training sát hơn với cơ chế support/query-prototype khi inference.

Checkpoint final đã khóa:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt
```

Result JSON final:

```text
/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100/gsc_edgespot_exact_k10_results.json
```

## 5. Kết quả chính

| Model | Split | Runs | ACC@1% FAR | ACC@5% FAR | FRR@5% FAR | AUC | EER | Keyword ACC | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN-L Triplet | dev | 30 | 76.48% | 79.17% | 47.91% | 89.63% | 19.77% | 69.65% | 71.27% |
| DSCNN-L Triplet | test | 100 | - | 80.54% | 40.58% | 91.22% | 18.22% | 68.39% | 73.30% |
| EdgeSpotFull T4 SCAF | dev | 30 | 82.48% | 84.85% | 21.52% | 95.73% | 11.28% | 72.76% | 82.78% |
| EdgeSpotFull T4 SCAF | test | 100 | 84.64% | 85.21% | 20.61% | 95.69% | 11.89% | 74.52% | 81.92% |
| EdgeSpotFull T4 SCAF+GE2E | dev | 30 | 83.78% | 85.56% | 21.85% | 95.60% | 11.63% | 76.15% | 82.29% |
| EdgeSpotFull T4 SCAF+GE2E | test | 100 | 84.61% | 86.12% | 21.39% | 95.61% | 11.54% | 77.66% | 82.41% |

Model final đạt:

```text
ACC@5% FAR  = 86.12%
Keyword ACC = 77.66%
F1          = 82.41%
EER         = 11.54%
FRR@5% FAR  = 21.39%
```

So với DSCNN-L Triplet test100, model final tăng `ACC@5% FAR` 5.58 điểm %, tăng `Keyword ACC` 9.27 điểm %, tăng `F1` 9.11 điểm %, giảm `EER` 6.68 điểm %, và giảm `FRR@5% FAR` 19.19 điểm %. Điều này cho thấy hướng EdgeSpot-style kết hợp SCAF+GE2E cải thiện rõ rệt khả năng nhận diện keyword và open-set rejection trong profile Microset.

## 6. Figure và bảng tự động

Bảng kết quả chuẩn được sinh từ:

```bash
python scripts/make_result_table.py --results-dir results --out-dir reports/microset --profile microset_en
```

Nếu chưa copy result JSON từ Colab về local, script sẽ dùng fallback manifest:

```text
reports/microset/locked_results_manifest.json
```

Các artifact chuẩn:

- `reports/microset/result_table.md`;
- `reports/microset/result_table.csv`;
- `reports/microset/result_table.tex`.

Phân tích lỗi/per-word khi có result JSON đầy đủ:

```bash
python scripts/analyze_result_errors.py \
  --result-json <path-to-gsc_edgespot_exact_k10_results.json> \
  --out-dir reports/microset
```

## 7. Giới hạn

Các giới hạn cần ghi rõ trong thesis:

- Kết quả hiện tại dùng MSWC Microset English, chưa phải Top500 full.
- Chưa dùng full MSWC English.
- Chưa phải EdgeSpot paper reproduction đầy đủ.
- Chưa chạy KD với Wav2Vec2 teacher.
- Chưa có streaming benchmark chính thức với microphone thật.
- Baseline DSCNN-L test100 hiện thiếu `ACC@1% FAR` nếu run chỉ được đánh giá ở `target_far=5%`.

## 8. Kết luận

Trong điều kiện tài nguyên hạn chế, Microset experiment đã cho thấy việc chuyển từ DSCNN-L Triplet sang EdgeSpotFull T4 SCAF+GE2E cải thiện đáng kể hiệu năng few-shot open-set keyword spotting. Kết quả này đủ làm mốc thesis hiện tại và làm nền cho phase tiếp theo trên Top500 full.
