# Runbook Colab cap620 development experiments - 2026-06-12

Muc tieu cua runbook nay la chay tiep sau matrix fixed 16 pipeline cap620. Khong lap lai full 16 pipeline neu da co ket qua; chi chay cac nhanh co kha nang tao them bang chung nghien cuu.

## 1. Trang thai xuat phat

Ket qua cap620 fixed hien tai:

- Best accuracy: `DSCNN-L + PCEN + GE2E`, test100 `ACC@1%FAR=82.34 +/- 1.19`, `AUC=92.42`, `EER=14.89`, `F1=77.75`.
- Best compact EdgeSpotFull T4 theo ACC@1%FAR: `PCEN + GE2E`, test100 `79.98 +/- 0.98`.
- Best compact EdgeSpotFull T4 theo AUC/EER/F1: `PCEN + Triplet`, test100 `AUC=89.85`, `EER=18.22`, `F1=73.29`.
- SCAF va SCAF+GE2E tren full cap620 bi collapse o nhieu cau hinh; khong nen chay tiep full cap620 SCAF truoc khi co ablation nho.

Huong thuc nghiem moi duoc cau hinh trong:

```bash
colab/run_mswc_cap620_development_experiments.sh
```

Runner nay them 4 nhanh:

- Accuracy: `DSCNN-L + PCEN + GE2E`, tang episode budget, hard-pair episode seeding, chon checkpoint bang composite metric.
- Compact: `EdgeSpotFull T4 + PCEN + Triplet` va `EdgeSpotFull T4 + PCEN + GE2E`, tranh SCAF.
- KD: `EdgeSpotFull T4 + PCEN + KD-GE2E`, mac dinh tat vi ton thoi gian/disk.
- SCAF ablation: subset-only, GE2E warmup roi finetune SCAF+GE2E voi weight/margin/scale nho.

## 2. Neu Colab dang bao gan day disk

Neu Colab hien `Disk is almost full`, vi du `221 GB / 235 GB`, khong nen bam chay tiep full experiment trong runtime do.

Lam theo thu tu nay:

1. Dam bao run cu da sync ve Drive:

```bash
%%bash
ls -lh /content/drive/MyDrive/DoAnTotNghiep_colab_runs
```

2. Neu da co artifact tren Drive, dung runtime cu va tao runtime moi:

```text
Runtime -> Disconnect and delete runtime
Runtime -> Change runtime type -> A100/T4 tuy quota -> Connect
```

3. Mount Drive lai va giai nen code moi. Khong can giu `/content/data/mswc_en` cu vi audio cap620 se chiem rat nhieu disk.

## 3. Lenh chay chinh nen thu dau tien

Day la luot nen chay truoc. No chay 3 stage: mot accuracy stage va hai compact stage. KD va SCAF ablation deu tat.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Neu ban upload zip code:

```bash
%%bash
cd /content
rm -rf DoAnTotNghiep
unzip -q /content/DoAnTotNghiep_code_colab_development_POSIX.zip
cd /content/DoAnTotNghiep
ls colab/run_mswc_cap620_development_experiments.sh
```

Chay:

```bash
%%bash
cd /content/DoAnTotNghiep

MAX_SECONDS=172800 \
SYNC_SECONDS=300 \
RUN_ACCURACY=1 \
RUN_COMPACT=1 \
RUN_KD=0 \
RUN_SCAF_ABLATION=0 \
ACC_EPOCHS=60 \
ACC_EPISODES=300 \
COMPACT_EPOCHS=60 \
COMPACT_EPISODES=300 \
GSC_SELECT_METRIC=composite \
bash colab/run_mswc_cap620_development_experiments.sh
```

Neu runtime/disk yeu hon, dung ban ngan hon:

```bash
%%bash
cd /content/DoAnTotNghiep

MAX_SECONDS=172800 \
SYNC_SECONDS=300 \
RUN_ACCURACY=1 \
RUN_COMPACT=1 \
RUN_KD=0 \
RUN_SCAF_ABLATION=0 \
ACC_EPOCHS=50 \
ACC_EPISODES=250 \
COMPACT_EPOCHS=50 \
COMPACT_EPISODES=250 \
GSC_SELECT_METRIC=composite \
bash colab/run_mswc_cap620_development_experiments.sh
```

## 4. Y nghia cau hinh chinh

Checkpoint selection:

- `--select-by-gsc-dev`
- `--gsc-dev-select-metric composite`
- Composite metric = trung binh cua `ACC@1%FAR`, `AUC`, `F1` tren GSC-dev.

Hard episode mining:

- `--hard-pairs-path results/hard_pairs.json`
- `--hard-pair-prob 0.35`

Accuracy stage:

- `model_family=dscnn`
- `feature=mel_pcen`
- `loss=ge2e`
- `epochs=60`
- `episodes=300`
- `n_classes=40`
- `n_samples=10`

Compact stages:

- `EdgeSpotFull T4 + PCEN + Triplet`, `--mining hard`, `--margin 1.0`
- `EdgeSpotFull T4 + PCEN + GE2E`
- Ca hai deu dung `edge_tau=4`, hard-pair episodes, composite checkpoint selection.

## 5. KD chay rieng sau

Chi chay KD sau khi luot chinh da xong. KD se cai `transformers`, train teacher head Wav2Vec2, precompute teacher embeddings tren subset, roi train student KD.

```bash
%%bash
cd /content/DoAnTotNghiep

MAX_SECONDS=172800 \
SYNC_SECONDS=300 \
RUN_ACCURACY=0 \
RUN_COMPACT=0 \
RUN_KD=1 \
RUN_SCAF_ABLATION=0 \
KD_MAX_WORDS=1000 \
KD_MAX_PER_WORD=80 \
KD_HEAD_EPOCHS=20 \
KD_EPOCHS=40 \
KD_EPISODES=250 \
KD_WEIGHT=0.5 \
GSC_SELECT_METRIC=composite \
bash colab/run_mswc_cap620_development_experiments.sh
```

Luu y claim:

- Day la `teacher-guided subset experiment`, chua phai bang chung vuot EdgeSpot-4 neu chua co `test100_far1` va so sanh voi baseline cung data/subset.
- Neu KD tot tren subset, buoc tiep theo moi mo rong `KD_MAX_WORDS` va `KD_MAX_PER_WORD`.

## 6. SCAF ablation chay rieng

Chi dung de tim cach sua collapse, khong chay full cap620 SCAF ngay.

```bash
%%bash
cd /content/DoAnTotNghiep

MAX_SECONDS=172800 \
SYNC_SECONDS=300 \
RUN_ACCURACY=0 \
RUN_COMPACT=0 \
RUN_KD=0 \
RUN_SCAF_ABLATION=1 \
SCAF_SUBSET_WORDS=1000 \
SCAF_MAX_PER_WORD=80 \
SCAF_WARMUP_EPOCHS=10 \
SCAF_FINETUNE_EPOCHS=30 \
SCAF_EPISODES=180 \
SCAF_WEIGHT=0.05 \
SCAF_SCALE=16.0 \
SCAF_MARGIN=0.2 \
GSC_SELECT_METRIC=composite \
bash colab/run_mswc_cap620_development_experiments.sh
```

Runner se:

1. Train warmup `EdgeSpotFull T4 + PCEN + GE2E` tren subset.
2. Load `best.pt` cua warmup bang `--resume-encoder-only`.
3. Finetune `SCAF+GE2E` voi `scaf_weight=0.05`, `scale=16`, `margin=0.2`.

## 7. Theo doi va trich ket qua

Xem log moi nhat:

```bash
%%bash
cd /content/DoAnTotNghiep
tail -n 120 "$(find logs_colab -name run.log | sort | tail -n 1)"
```

Xem summary stage:

```bash
%%bash
cd /content/DoAnTotNghiep
find logs_colab -name stages.tsv -print -exec cat {} \;
```

In ket qua test100 FAR1/FAR5 dang co:

```bash
%%bash
cd /content/DoAnTotNghiep
python - <<'PY'
from pathlib import Path
import json

for p in sorted(Path("results").glob("**/metrics.json")):
    try:
        m = json.loads(p.read_text())
    except Exception:
        continue
    if "open_set_acc_at_1far" not in m and "open_set_acc_at_5far" not in m:
        continue
    print("\n", p)
    for k in [
        "open_set_acc_at_1far", "open_set_acc_at_5far",
        "auc", "eer", "f1", "keyword_acc", "frr_at_5far",
    ]:
        if k in m:
            print(f"  {k}: {m[k]:.4f}")
PY
```

Drive output mac dinh:

```bash
/content/drive/MyDrive/DoAnTotNghiep_colab_runs/<RUN_ID>/
```

## 8. Tieu chi quyet dinh sau khi co ket qua

Accuracy branch:

- Neu `DSCNN-L + PCEN + GE2E` tang hon `82.34` va AUC/F1 cung tang, dung lam ket qua accuracy chinh.
- Neu ACC tang nhung AUC/F1 giam, bao cao nhu trade-off do composite metric va can xem DET.

Compact branch:

- So sanh `EdgeSpotFull T4 + PCEN + Triplet` voi `EdgeSpotFull T4 + PCEN + GE2E`.
- Triplet dang duoc uu tien neu tiep tuc giu AUC/EER/F1 tot hon.
- Muon claim vuot EdgeSpot-4 compact thi can dat hon moc paper `82.0% ACC@1%FAR` tren `test100_far1`, cung protocol.

KD branch:

- Chi tiep tuc mo rong neu KD vuot baseline EdgeSpotFull T4 cung subset hoac it nhat tang AUC/F1 ro rang.
- Neu KD khong tang, khong dung lam claim chinh; dua vao thesis nhu huong thu nghiem can tuning teacher/subset/weight.

SCAF ablation:

- Thanh cong khi AUC thoat khoi khoang `50`, F1 khac `0`, FRR@FAR khong con `100`.
- Neu van collapse, giam tiep `SCAF_WEIGHT`, `SCAF_MARGIN`, hoac tang warmup GE2E; khong quay lai full cap620 SCAF.
