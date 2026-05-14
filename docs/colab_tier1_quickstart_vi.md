# Colab Tier-1 Quickstart

## 1. Clone code moi nhat

```python
%cd /content
!rm -rf DoAnTotNghiep
!git clone https://github.com/AnHgPham/DoAnTotNghiep.git
%cd /content/DoAnTotNghiep
```

## 2. Cai dependencies

```python
!pip install -q -r requirements.txt
```

Neu Colab bi cham o `transformers`, co the cai rieng:

```python
!pip install -q "transformers>=4.40.0"
```

## 3. Smoke test truoc khi train

```python
!python -m pytest tests/test_edgespot_full.py tests/test_ge2e.py tests/test_gsc_silence_provider.py -q
!python scripts/model_report.py --family edgespot_full --tau 4
```

## 4. Kiem tra data cache

Mo notebook rieng cho Tier-1:

```text
notebooks/03_tier1_edgespot_colab.ipynb
```

Chay cac cell mount Drive va setup dataset trong notebook nay truoc. Dataset debug cu
co dang:

```text
MSWC: ~95k WAV neu Top500 mpw200
GSC: ~105k WAV
DEMAND: ~272 WAV
```

Muc `~95k WAV` chi dung de debug/smoke test. Khong dung de bao cao reproduce
EdgeSpot vi moi word da bi gioi han toi da 200 clips.

## 5. Data profile quan trong

De debug nhanh, co the dung cache cu:

```python
MSWC_SPLIT_MODE = 'top500'
MSWC_MAX_PER_WORD = 200
```

De chay nghiem tuc theo huong paper-grade, phai dung all clips/word:

```python
MSWC_SPLIT_MODE = 'top500'
MSWC_MAX_PER_WORD = 0
```

Khi `MSWC_MAX_PER_WORD = 0`, cache moi se la
`mswc_en_wav_top500_full`. Lan dau se tai/extract/convert lon hon nhieu so
voi `mswc_en_wav_top500_mpw200`. Khong dung ket qua `mpw200` de claim
reproduce EdgeSpot.

Sau khi setup MSWC xong, bat buoc in data profile:

```python
!python scripts/mswc_data_report.py --data-dir data/mswc_en --top-n 20
```

Neu output canh bao `mpw200 debug cache`, dung train nghiem tuc va doi lai:

```python
MSWC_MAX_PER_WORD = 0
```

## 6. Train EdgeSpotFull + SCAF + GE2E

```python
!python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e \
  --epochs 40 \
  --episodes 600 \
  --num-workers 2 \
  --select-by-gsc-dev \
  --gsc-dev-runs 5 \
  --gsc-dev-k-shot 10
```

Neu train qua lau, debug truoc bang ban nho:

```python
!python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 1 \
  --loss scaf_ge2e \
  --run-tag smoke_edgespot_t1 \
  --epochs 1 \
  --episodes 5 \
  --max-per-word 20 \
  --num-workers 2
```

## 7. Benchmark EdgeSpot exact

```python
!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint checkpoints/edgespot_full_t4_scaf_ge2e/best.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir results/edgespot_exact/edgespot_full_t4_scaf_ge2e
```

## 8. Tao bang ket qua

```python
!python scripts/make_research_tables.py results/edgespot_exact/edgespot_full_t4_scaf_ge2e/*_results.json
```

## 9. KD phase sau khi SCAF/GE2E on dinh

```python
!python scripts/precompute_teacher_embeddings.py \
  --data-dir data/mswc_en \
  --split train \
  --output-dir outputs/teacher_w2v2_train \
  --batch-size 16

!python scripts/train.py \
  --config configs/default.yaml \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss kd_scaf_ge2e \
  --teacher-embeddings-dir outputs/teacher_w2v2_train \
  --run-tag edgespot_full_t4_kd_scaf_ge2e \
  --epochs 40 \
  --episodes 600 \
  --num-workers 2
```

Luu y: KD chi nen dung de claim khoa hoc khi projection head teacher da duoc
train/kiem chung. Khong claim paper-grade neu chi dung head random.
