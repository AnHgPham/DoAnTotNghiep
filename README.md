# Few-Shot Open-Set Keyword Spotting

## Locked Microset Result

The current thesis baseline/result is locked on **MSWC Microset English official
CSV split**. Do not tune against GSC test100 after this point.

- Final model: `EdgeSpotFull T4 + SCAF+GE2E`.
- Final checkpoint: `epoch_05.pt`.
- Final GSC test100: `ACC@5% FAR = 86.12%`, `KW-ACC = 77.66%`, `F1 = 82.41%`.
- Canonical manifest: `reports/microset/locked_results_manifest.json`.
- Generated tables: `reports/microset/result_table.md`, `.csv`, `.tex`.
- Thesis chapter draft: `docs/thesis_experiment_chapter_vi.md`.

Regenerate the result table after copying Colab JSON folders locally:

```bash
python scripts/make_result_table.py --results-dir results --out-dir reports/microset --profile microset_en
```

## Current Training Workflow

Current recommended Colab workflow is command-based, not notebook-driven:

- Use a blank Colab notebook as an A100/GPU terminal.
- Copy cells from `docs/colab_microset_runbook_vi.md`.
- Current temporary data profile: `MSWC Microset English`.
- This is NOT Top500 full, NOT full MSWC, and NOT an EdgeSpot paper reproduction claim.
- `notebooks/02_train_enhanced.ipynb` is legacy/experimental for now.
- The current status note is in `docs/current_training_profile_vi.md`.

Hệ thống nhận diện từ khóa few-shot cho đồ án tốt nghiệp: người dùng chỉ cần
thu 3-5 mẫu cho mỗi từ khóa, hệ thống tạo prototype embedding và nhận diện từ
khóa trong audio tĩnh hoặc luồng microphone streaming. Trọng tâm của project là
open-set KWS: nhận đúng từ đã đăng ký và biết từ chối âm thanh không thuộc bộ từ
khóa.

## Điểm Chính

- Few-shot enrollment: thêm từ mới bằng 3-5 mẫu WAV/microphone, không cần train
  lại toàn bộ model.
- Open-set rejection: dùng khoảng cách L2/prototype, threshold toàn cục hoặc
  threshold riêng từng từ để giảm false alarm.
- Streaming KWS: VAD/energy segmentation, multi-window scoring, voting, margin
  và cooldown để dùng với microphone thực tế.
- Model baseline: DSCNN-L + MFCC, ổn định cho báo cáo và demo.
- Model thử nghiệm: EdgeSpot-lite với mel 40x101, trainable PCEN và temporal
  attention, dùng để tiếp cận hướng EdgeSpot.
- Training data: MSWC English cho train, Google Speech Commands v2 cho đánh giá,
  DEMAND noise cho augmentation.
- Colab workflow: cache MSWC WAV trên Google Drive để lần sau không phải convert
  lại OPUS sang WAV.

## Trạng Thái Khuyến Nghị

- Train hien tai: dung Colab notebook trong va copy cells tu
  `docs/colab_microset_runbook_vi.md`.
- Data profile hien tai: `MSWC Microset English` tam thoi de tiet kiem
  Colab units/disk. Day khong phai Top500 full, khong phai full MSWC,
  va khong phai EdgeSpot paper reproduction.
- `notebooks/02_train_enhanced.ipynb` va `notebooks/03_tier1_edgespot_colab.ipynb`
  chi con la legacy/experimental references, khong phai workflow train chinh.
- Khi co may/disk on dinh, tao runbook rieng cho `top500_full`; khong sua lan
  vao runbook Microset.
- Demo/eval: luon lay `best.pt` trong run tot nhat, khong dung `latest.pt` neu
  checkpoint train tiep bi degrade.
- Artifact lớn như `data/`, `checkpoints/`, `results/`, `outputs/`, `*.zip` không
  được commit vào Git. Lưu chúng ở Google Drive hoặc ổ local.

## Cài Đặt Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Nếu chạy trên máy không có GPU, project vẫn chạy được demo/eval nhỏ bằng CPU,
nhưng training MSWC nên chạy trên Colab hoặc máy có CUDA.

## Chuẩn Bị Dữ Liệu

Google Speech Commands v2 dùng cho benchmark:

```bash
python data/download_gsc.py
```

MSWC Top500 local:

```bash
python data/download_mswc.py --top500-splits --max-per-word 0
python data/convert_opus.py --delete-opus
```

MSWC cache trên Google Drive cho Colab:

```bash
python data/mswc_drive_cache.py \
  --drive-project /content/drive/MyDrive/DoAnTotNghiep_output \
  --split-mode top500 \
  --max-per-word 0 \
  --workers 2
```

Cache hợp lệ cần có `clips/<word>/*.wav`, `splits/train_words.json`,
`splits/val_words.json` và coverage đủ cao cho train/val words. Khi cache hit,
notebook symlink/copy dữ liệu về `data/mswc_en` để API training không đổi.

MSWC Microset chinh thuc cua MLCommons, dung khi Colab/o dia khong du cho Top500/full:

```bash
python data/download_mswc_microset.py --language en --workers 2
python scripts/mswc_data_report.py --data-dir data/mswc_microset_en
```

Train nhanh tren Microset:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data-dir data/mswc_microset_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e_microset_en \
  --epochs 20 \
  --episodes 100 \
  --num-workers 2
```

## Training

Research/publication roadmap:

```bash
python scripts/research_readiness.py --data-profile microset_en
python scripts/research_readiness.py --data-profile top500_full \
  --results-root results \
  --checkpoints-root checkpoints
```

DSCNN baseline sạch:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --model-family dscnn \
  --loss triplet \
  --run-tag dscnn_top500_full_clean \
  --epochs 35 \
  --episodes 200 \
  --num-workers 2 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.001
```

Hard-pair ablation sau khi có confusion matrix:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --model-family dscnn \
  --loss triplet \
  --run-tag dscnn_top500_full_hard02 \
  --hard-pairs-path results/hard_pairs.json \
  --hard-pair-prob 0.2
```

EdgeSpot-lite thử nghiệm:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_lite \
  --loss scaf \
  --run-tag edgespot_lite_top500_scaf \
  --epochs 35 \
  --episodes 200 \
  --num-workers 2
```

Tier-1 EdgeSpot reproduction:

```bash
# Model/parameter report for EdgeSpot-4 style encoder
python scripts/model_report.py --family edgespot_full --tau 4

# Train EdgeSpotFull with SCAF
python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf \
  --run-tag edgespot_full_t4_scaf \
  --epochs 40 \
  --episodes 600 \
  --num-workers 2

# GE2E hybrid objective
python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e
```

Wav2Vec2 teacher KD workflow:

```bash
python scripts/precompute_teacher_embeddings.py \
  --data-dir data/mswc_en \
  --split train \
  --output-dir outputs/teacher_w2v2_train \
  --batch-size 16

python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss kd_scaf_ge2e \
  --teacher-embeddings-dir outputs/teacher_w2v2_train \
  --run-tag edgespot_full_t4_kd_scaf_ge2e
```

## Evaluation

Đánh giá few-shot open-set trên GSC:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/<run_tag>/best.pt \
  --protocol gsc_fixed \
  --k-shot 5 \
  --plot-det

python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/<run_tag>/best.pt \
  --protocol gsc_random \
  --k-shot 5 \
  --plot-det
```

Benchmark streaming/robustness:

```bash
python scripts/benchmark_robust_streaming.py \
  --checkpoint checkpoints/<run_tag>/best.pt \
  --gsc-dir data/gsc_v2 \
  --k-shot 5
```

Canonical EdgeSpot-style benchmark with true silence:

```bash
python scripts/evaluate_edgespot_protocol.py \
  --checkpoint checkpoints/<run_tag>/best.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir results/edgespot_exact/<run_tag>

python scripts/make_research_tables.py results/edgespot_exact/<run_tag>/*_results.json
```

Các metric quan trọng:

- `keyword_acc`: độ chính xác khi audio là một keyword đã biết.
- `open_set_acc@5% FAR`: độ chính xác open-set tại false accept rate 5%.
- `FRR@5% FAR`: tỉ lệ bỏ sót keyword khi FAR được cố định ở 5%.
- `false_alarms_per_hour`: số lần báo nhầm trong streaming.
- `miss_rate`: tỉ lệ không phát hiện keyword trong audio dài.
- `latency_ms`: độ trễ từ lúc nói xong đến lúc hệ thống phát hiện.

## Demo

Gradio demo:

```bash
python src/demo/app.py
```

FastAPI + web UI:

```bash
python -m src.demo.api_server
```

Sau đó mở `http://127.0.0.1:8000`.

Demo hỗ trợ:

- enroll keyword từ GSC hoặc microphone;
- lưu/load enrollment profile;
- detect audio ngắn;
- detect file dài;
- streaming microphone qua WebSocket;
- open-set test với threshold toàn cục hoặc threshold riêng từng từ.

## Cấu Trúc Repo

```text
configs/              YAML cấu hình model, data, training, evaluation
data/                 Script download/cache dataset, không commit dataset thật
docs/                 Tài liệu báo cáo, proposal, phân tích thí nghiệm
notebooks/            Legacy/experimental notebooks; workflow hien tai o docs/colab_microset_runbook_vi.md
scripts/              Train, evaluate, benchmark, confusion analysis
src/
  classifiers/        OpenNCM, OpenMAX, energy OOD classifiers
  data/               Dataset loaders cho MSWC/GSC
  demo/               Gradio app, FastAPI backend, static web UI
  enhancements/       Denoising và speaker verification optional
  evaluation/         Protocols, metrics, DET curves
  features/           MFCC, mel, PCEN, augmentation, SpecAugment
  models/             DSCNN, BCResNetFS, EdgeSpot-lite/full, Triplet/ArcFace/SCAF/GE2E
  streaming/          VAD/energy streaming engines
tests/                Unit tests và smoke tests
```

## Artifact Policy

Không commit các thư mục/file sau:

- `data/gsc_v2`, `data/mswc_en`, `data/demand`
- `checkpoints/`
- `results/`
- `outputs/`
- `runs/`, `wandb/`, `logs/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.rar`
- `__pycache__/`, `*.pyc`

Khi cần chia sẻ model hoặc kết quả, nén riêng artifact hoặc lưu trên Google
Drive. Git chỉ nên chứa source code, config, notebook, test và tài liệu nhẹ.

## Test

```bash
python -m pytest tests -q
```

Nhóm test hiện kiểm tra feature shapes, model forward, metrics, open-set
classifier, MSWC Drive cache và streaming engine.

## Hướng Phát Triển

Ưu tiên thực tế để tăng chất lượng demo:

1. Hoàn thiện streaming state machine và threshold calibration theo từng từ.
2. Thêm benchmark audio dài: false alarm/hour, miss rate, latency, duplicate
   detection.
3. Train lại trên streaming-style augmentation: silence, noise, unknown speech,
   keyword offset ngẫu nhiên.
4. Dùng hard negative/impostor bank cho enrollment 3-5 mẫu.
5. Sau khi baseline ổn, mở phase EdgeSpot đầy đủ với Wav2Vec2 teacher KD.

## Tham Khảo

- EdgeSpot: Efficient and High-Performance Few-Shot Model for Keyword Spotting,
  arXiv 2601.16316.
- Google Speech Commands v2.
- Multilingual Spoken Words Corpus.
- Few-shot open-set KWS và query-by-example KWS literature.
