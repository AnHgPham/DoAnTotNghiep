# Colab Microset Runbook

Day la runbook train tam thoi bang MSWC Microset English tren Colab.
A100 la lua chon khuyen nghi; G4 co the dung neu muon tiet kiem hon nhung se cham hon.
Copy tung cell vao mot Colab notebook trong va chay tu tren xuong.

```text
CURRENT DATA PROFILE = MSWC MICROSET ENGLISH
TEMPORARY RUN FOR UNIT/DISK SAVING
NOT TOP500 FULL
NOT FULL MSWC
NOT EDGESPOT PAPER REPRODUCTION
```

Notebook chi la noi chay lenh. Logic that nam trong repo. Khong dung
`notebooks/02_train_enhanced.ipynb` cho workflow nay.

## 0. Chon Runtime

Trong Colab:

```text
Runtime -> Change runtime type -> GPU -> A100
```

A100 la runtime khuyen nghi cho run nay. Neu khong can toc do cao, co the chon
G4. Khong dung H100 cho Microset vi khong dang chi phi/unit.

Chay cell kiem tra:

```python
import torch, shutil

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Free disk GB:", round(shutil.disk_usage("/content").free / 1024**3, 1))
```

Neu khong thay A100 hoac G4 thi dung lai va chon runtime lai.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")

import os

DRIVE_PROJECT = "/content/drive/MyDrive/DoAnTotNghiep_output"
os.makedirs(f"{DRIVE_PROJECT}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_PROJECT}/results", exist_ok=True)
os.makedirs(f"{DRIVE_PROJECT}/notes", exist_ok=True)

print("Drive output:", DRIVE_PROJECT)
```

## 2. Clone Repo Moi Nhat

```python
%cd /content
!rm -rf DoAnTotNghiep
!git clone https://github.com/AnHgPham/DoAnTotNghiep.git
%cd /content/DoAnTotNghiep
!git log -1 --oneline
```

## 3. Ghi Note Microset Vao Drive

```python
from pathlib import Path

NOTE = """
CURRENT TRAINING PROFILE: MSWC Microset English

Trang thai:
- Dang dung MSWC Microset English TAM THOI.
- Muc dich: tiet kiem Colab units/disk, test pipeline, lay baseline nhanh.
- KHONG claim day la Top500 full.
- KHONG claim day la full MSWC.
- KHONG claim reproduce EdgeSpot paper bang ket qua Microset.

Khi co may truong/disk/GPU on dinh:
- Tao runbook rieng cho Top500 full clips/word.
- Chay lai EdgeSpotFull va benchmark GSC exact.
"""

note_path = Path(f"{DRIVE_PROJECT}/notes/CURRENT_TRAINING_PROFILE_MICROSET_TEMP.txt")
note_path.write_text(NOTE, encoding="utf-8")
print(NOTE)
print("Saved note:", note_path)
```

## 4. Cai Dependencies

```python
%cd /content/DoAnTotNghiep
!pip install -q -r requirements.txt
```

Neu cell tren loi thieu package, chay them:

```python
!pip install -q torch torchaudio numpy pyyaml matplotlib tensorboard scikit-learn soundfile requests tqdm noisereduce speechbrain gradio
```

## 5. Test Code Truoc Khi Tai Data

```python
%cd /content/DoAnTotNghiep

from pathlib import Path
import subprocess

test_files = [
    "tests/test_edgespot_full.py",
    "tests/test_ge2e.py",
    "tests/test_gsc_silence_provider.py",
    "tests/test_demo_api_robust.py",  # optional neu repo clone da co test API moi
]
existing_tests = [p for p in test_files if Path(p).exists()]
missing_tests = [p for p in test_files if not Path(p).exists()]
if missing_tests:
    print("Skip missing optional tests:", missing_tests)
subprocess.run(["python", "-m", "pytest", *existing_tests, "-q"], check=True)

!python scripts/model_report.py --family edgespot_full --tau 4
```

Dieu kien OK:

- Pytest bao passed. Neu chi missing `tests/test_demo_api_robust.py` thi van co the tiep tuc;
  do la test API phu cua ban moi hon, khong anh huong EdgeSpot train.
- Model report co `family: edgespot_full`.
- Model report co `tau: 4`.
- Output shape la `(1, 64)`.

## 6. Tao Config Microset Rieng

```python
import yaml
from pathlib import Path

%cd /content/DoAnTotNghiep

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["family"] = "edgespot_full"
cfg["model"]["edge_width_mult"] = 4
cfg["model"]["edge_embedding_dim"] = 64
cfg["model"]["edge_use_pcen"] = True

cfg["data"]["train_dir"] = "data/mswc_microset_en"
cfg["data"]["mswc_dir"] = "data/mswc_microset_en"
cfg["data"]["gsc_dir"] = "data/gsc_v2"
cfg["data"]["demand_dir"] = "data/demand"
cfg["data"]["max_per_word"] = 0
cfg["data"]["val_max_per_word"] = 0

cfg["training"]["epochs"] = 25
cfg["training"]["episodes_per_epoch"] = 200
cfg["training"]["n_classes"] = 31
cfg["training"]["n_samples"] = 16
cfg["training"]["loss"] = "scaf_ge2e"
cfg["training"]["grad_clip"] = 5.0
cfg["training"]["optimizer"]["lr"] = 0.001
cfg["training"]["optimizer"]["weight_decay"] = 0.0001
cfg["training"]["scheduler"]["type"] = "CosineAnnealingWarmRestarts"
cfg["training"]["scheduler"]["T_0"] = 10
cfg["training"]["scheduler"]["T_mult"] = 2
cfg["training"]["scheduler"]["eta_min"] = 0.00001

cfg["augmentation"]["noise_prob"] = 0.50
cfg["augmentation"]["spec_augment"]["enabled"] = True
cfg["augmentation"]["spec_augment"]["freq_mask"] = 6
cfg["augmentation"]["spec_augment"]["time_mask"] = 8
cfg["augmentation"]["spec_augment"]["n_freq_masks"] = 1
cfg["augmentation"]["spec_augment"]["n_time_masks"] = 1

cfg["noise"]["demand_dir"] = "data/demand"
cfg["noise"]["prob"] = 0.50

cfg["checkpoint"]["dir"] = f"{DRIVE_PROJECT}/checkpoints"

out = Path("/content/tier1_microset_colab.yaml")
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Wrote:", out)
print("CURRENT DATA PROFILE: MSWC MICROSET ENGLISH TEMPORARY")
```

Ghi chu: Microset English co 31 keyword nhung `en_train.csv` khong can bang
tuyet doi. Tu `sheila` chi co 16 clips trong train split, nen `n_samples` phai
la `16` neu muon moi episode dung du 31 keyword. Dat `n_samples=20` se loi:
`Need at least 31 classes with >=20 samples each, but only 30 classes qualify`.

## 7. Tai GSC V2 Cho Evaluation

```python
%cd /content/DoAnTotNghiep
!python data/download_gsc.py
```

## 8. Tai MSWC Microset English

```python
%cd /content/DoAnTotNghiep
!python data/download_mswc_microset.py --language en --workers 2 --split-source official --rewrite-splits
```

Lenh nay cung sua lai split cu neu Colab da tai data tu truoc. Ket qua dung cho
profile tam thoi nay la split sample-level chinh thuc cua Microset:

- `train_words.json`: 31 keywords tu `en_train.csv`.
- `val_words.json`: 31 keywords tu `en_dev.csv`.
- `eval_words.json`: 31 keywords tu `en_test.csv`.
- `train_files.json`, `val_files.json`, `eval_files.json`: danh sach file chinh
  thuc de train/val/test. Train se dung file manifest nay thay vi quet toan bo
  folder `clips/<word>`.

Neu ban thay folder nhu `clips/one` co hon 6000 OPUS/WAV thi khong sao: folder
chua ca train+dev+test. Gioi han 6000 clips/keyword ap dung cho `en_train.csv`,
khong phai tong folder. Code hien tai se dung `train_files.json` sinh tu CSV de
tranh train nham ca dev/test trong folder.
Checkpoint se duoc chon bang GSC-dev, khong bang Microset val.

Kiem tra data:

```python
%cd /content/DoAnTotNghiep
!python scripts/mswc_data_report.py --data-dir data/mswc_microset_en --top-n 20
```

## 9. Tai DEMAND Noise Dataset

Nen chay cell nay de augmentation tot hon. Neu chi smoke test cuc nhanh, co the
skip.

```python
%%time
%cd /content/DoAnTotNghiep

import shutil, time, zipfile, requests
from pathlib import Path
from tqdm import tqdm

DEMAND_DIR = Path("data/demand")
MIN_DEMAND_WAV = 250
BASE = "https://zenodo.org/records/1227121/files"
ENVS = [
    "DKITCHEN","DLIVING","DWASHING","NFIELD","NPARK","NRIVER",
    "OHALLWAY","OMEETING","OOFFICE","PCAFETER","PRESTO","PSTATION",
    "SPSQUARE","STRAFFIC","TBUS","TCAR","TMETRO",
]

def download_file(url, out_path, retries=3):
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"Retry {attempt}/{retries}: {exc}. Wait {wait}s")
            time.sleep(wait)

DEMAND_DIR.mkdir(parents=True, exist_ok=True)
existing = len(list(DEMAND_DIR.rglob("*.wav")))
if 0 < existing < MIN_DEMAND_WAV:
    print("DEMAND incomplete, re-downloading...")
    shutil.rmtree(DEMAND_DIR, ignore_errors=True)
    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    for stale in Path("data").glob("*_16k.zip*"):
        stale.unlink(missing_ok=True)

if len(list(DEMAND_DIR.rglob("*.wav"))) < MIN_DEMAND_WAV:
    for env in tqdm(ENVS, desc="Downloading DEMAND"):
        url = f"{BASE}/{env}_16k.zip?download=1"
        zp = Path(f"data/{env}_16k.zip")
        if not zp.exists():
            download_file(url, zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(DEMAND_DIR)
        zp.unlink(missing_ok=True)

print("DEMAND wav:", len(list(DEMAND_DIR.rglob("*.wav"))))
```

## 10. Kiem Tra Data Cuoi

```python
%cd /content/DoAnTotNghiep

from pathlib import Path

for d in ["data/gsc_v2", "data/mswc_microset_en", "data/demand"]:
    p = Path(d)
    if p.exists():
        wav_n = len(list(p.rglob("*.wav")))
        size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**3
        print(f"{d}: {wav_n:,} wav | {size_gb:.2f} GB")
    else:
        print(f"{d}: missing")

!python scripts/mswc_data_report.py --data-dir data/mswc_microset_en --top-n 20
```

## 11. Smoke Train 1 Epoch

Chay cell nay truoc train that.

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_microset_colab.yaml \
  --data-dir data/mswc_microset_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag smoke_edgespot_full_t4_microset_en \
  --epochs 1 \
  --episodes 5 \
  --max-per-word 0 \
  --num-workers 2
```

Dieu kien OK:

- Khong loi.
- Co log `Epoch 1/1`.
- Co checkpoint trong Drive.

## 12. Train Baseline: DSCNN-L Triplet

Run nay de lam baseline va de xem `d_pos` / `d_neg` trong log. DSCNN-L dung
MFCC, con EdgeSpotFull dung mel, nen hai run nay giup so sanh baseline cu voi
nhanh EdgeSpot.

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_microset_colab.yaml \
  --data-dir data/mswc_microset_en \
  --model-family dscnn \
  --loss triplet \
  --run-tag dscnn_l_triplet_microset_en_v1 \
  --epochs 20 \
  --episodes 200 \
  --max-per-word 0 \
  --num-workers 2 \
  --early-stop-patience 0 \
  --early-stop-min-delta 0.001 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10
```

Log cua run nay se co:

```text
d_pos=...
d_neg=...
```

Nen doc nhanh:

- `d_pos` la khoang cach cung tu.
- `d_neg` la khoang cach khac tu.
- Tot hon khi `d_pos` nho hon `d_neg` va hai gia tri tach nhau ro hon.

## 13. Evaluate DSCNN-L Dev 30 Runs

Chon checkpoint theo log GSC-dev, vi du epoch 05:

```python
%cd /content/DoAnTotNghiep

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/dscnn_l_triplet_microset_en_v1/epoch_05.pt \
  --model-family dscnn \
  --k-shot 10 \
  --n-runs 30 \
  --gsc-query-split dev \
  --output-dir /content/drive/MyDrive/DoAnTotNghiep_output/results/dscnn_l_triplet_microset_en_v1_epoch05_dev30
```

Neu epoch 10/15/20 tot hon, thay checkpoint va output-dir tuong ung.

## 14. Train Run 1: EdgeSpotFull T4 SCAF

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_microset_colab.yaml \
  --data-dir data/mswc_microset_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf \
  --run-tag edgespot_full_t4_scaf_microset_en_v1 \
  --epochs 20 \
  --episodes 200 \
  --max-per-word 0 \
  --num-workers 2 \
  --early-stop-patience 0 \
  --early-stop-min-delta 0.001 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10
```

## 15. Evaluate Run 1 Dev 30 Runs

```python
%cd /content/DoAnTotNghiep

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_microset_en_v1/epoch_05.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 30 \
  --gsc-query-split dev \
  --output-dir /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_dev30
```

Neu log train cho thay epoch 10/15/20 co GSC-dev tot hon epoch 05, thay checkpoint
va output-dir tuong ung. Khong mac dinh dung `best.pt` cho run Microset neu
`best.pt` duoc chon theo MSWC val.

## 16. Train Run 2: EdgeSpotFull T4 SCAF GE2E

Chay run nay sau khi Run 1 xong.

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_microset_colab.yaml \
  --data-dir data/mswc_microset_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e_microset_en_v1 \
  --epochs 25 \
  --episodes 200 \
  --max-per-word 0 \
  --num-workers 2 \
  --early-stop-patience 0 \
  --early-stop-min-delta 0.001 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10
```

## 17. Evaluate Run 2 Dev 30 Runs

Chon checkpoint theo log GSC-dev cua Run 2. Vi du neu epoch 05 tot nhat:

```python
%cd /content/DoAnTotNghiep

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 30 \
  --gsc-query-split dev \
  --output-dir /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_dev30
```

## 18. Final Test 100 Runs

Chi chay cho checkpoint tot nhat trong Run 1 hoac Run 2.
Thay `SELECTED_CKPT` bang checkpoint da thang dev30, thuong la `epoch_05.pt`,
`epoch_10.pt`, `epoch_15.pt`, ... Khong mac dinh dung `best.pt` neu Colab dang
chay code cu.

Neu Run 2 tot hon, vi du chon epoch 05:

```python
%cd /content/DoAnTotNghiep

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/epoch_05.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_microset_en_v1_epoch05_test100
```

Neu Run 1 tot hon, vi du chon epoch 05:

```python
%cd /content/DoAnTotNghiep

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_microset_en_v1/epoch_05.pt \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir /content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_microset_en_v1_epoch05_test100
```

## 19. Xem Ket Qua JSON

```python
from pathlib import Path
import json

result_files = sorted(Path("/content/drive/MyDrive/DoAnTotNghiep_output/results").rglob("*results.json"))
print("Found result files:", len(result_files))

for p in result_files[-10:]:
    print("\n===", p, "===")
    data = json.loads(p.read_text())
    for k in [
        "open_set_acc_at_1far",
        "open_set_acc_at_5far",
        "frr_at_5far",
        "auc",
        "eer",
        "keyword_acc",
        "f1",
    ]:
        if k in data:
            print(k, "=", data[k])
```

## 20. Resume Neu Colab Disconnect

Tim checkpoint:

```python
from pathlib import Path

RUN_TAG = "edgespot_full_t4_scaf_ge2e_microset_en_v1"
ckpt_dir = Path(f"/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/{RUN_TAG}")

print("Best (co the la MSWC-val neu code cu):", ckpt_dir / "best.pt")
print("Latest:", ckpt_dir / "latest.pt")
print("Epoch checkpoints:")
for p in sorted(ckpt_dir.glob("epoch_*.pt")):
    print(p)
```

Resume tu `latest.pt` de tiep tuc train dung epoch gan nhat:

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_microset_colab.yaml \
  --data-dir data/mswc_microset_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --resume /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_microset_en_v1/latest.pt \
  --run-tag edgespot_full_t4_scaf_ge2e_microset_en_v1 \
  --epochs 25 \
  --episodes 200 \
  --max-per-word 0 \
  --num-workers 2 \
  --early-stop-patience 0 \
  --early-stop-min-delta 0.001 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10
```

## Khong Chay Trong Workflow Nay

- Khong chay H100 cho Microset.
- Khong chay Top500 full.
- Khong chay full MSWC.
- Khong chay teacher distillation/KD.
- Khong chay grid tau 1/2/3/4.
- Khong chay `notebooks/02_train_enhanced.ipynb`.

## Luu Ten Run

Tat ca run Microset phai co `microset_en` trong ten:

- `smoke_edgespot_full_t4_microset_en`
- `dscnn_l_triplet_microset_en_v1`
- `edgespot_full_t4_scaf_microset_en_v1`
- `edgespot_full_t4_scaf_ge2e_microset_en_v1`

Sau nay neu chuyen sang data lon hon, tao runbook rieng va dung hau to dataset
rieng. Khong sua lan vao runbook Microset nay.
