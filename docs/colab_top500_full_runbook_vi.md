# Colab Runbook - MSWC Top500 Full Clips + EdgeSpotFull T4 SCAF+GE2E

Muc tieu: chay profile nghiem tuc hon Microset, dung **MSWC English Top500 full clips per word** de train `EdgeSpotFull T4 + SCAF+GE2E`, roi evaluate bang `gsc_edgespot_exact`.

```text
CURRENT DATA PROFILE = MSWC TOP500 FULL CLIPS
RUN TAG SUFFIX = top500_full_v1
NOT MICROSET
NOT FULL MSWC 38K WORDS
NOT EDGESPOT PAPER REPRODUCTION YET
```

Canh bao quan trong:

- Top500 full clips lon hon Microset rat nhieu. Lan dau co the can 1-2 gio de download/extract/convert va co the dung tren 100GB tam thoi tuy vao so clip thuc te.
- Khong doi sang `max_per_word=200` neu dang claim Top500 full. Neu phai giam cap de tiet kiem unit, doi run tag thanh `top500_mpwXXXX`, khong goi la full.
- Khong resume checkpoint Microset vao Top500 full bang `--resume`; SCAF head 31 class va Top500 450 class khac shape.
- Chon checkpoint bang GSC-dev. Chi chay GSC-test 100 runs sau khi da khoa checkpoint.

## 1. Chon Runtime

Trong Colab:

```text
Runtime -> Change runtime type -> A100 GPU
```

Khuyen nghi:

- A100: nen dung.
- G4/T4: chi dung smoke/pilot nho, khong nen train final.
- H100: chua can, ton units; chi can cho KD phase sau.

## 2. Clone Repo Moi Nhat

```python
%cd /content
!rm -rf DoAnTotNghiep
!git clone https://github.com/AnHgPham/DoAnTotNghiep.git
%cd /content/DoAnTotNghiep
!git log -1 --oneline
```

Neu da clone san:

```python
%cd /content/DoAnTotNghiep
!git pull origin main
!git log -1 --oneline
```

## 3. Cai Dependencies

```python
%cd /content/DoAnTotNghiep

!pip install -q torch torchaudio numpy pyyaml matplotlib tensorboard \
    scikit-learn soundfile requests tqdm noisereduce speechbrain gradio
!pip install -q -r requirements.txt

import torch, os
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 4. Mount Drive Va Dat Bien Chinh

```python
from google.colab import drive
drive.mount("/content/drive")

import os
from pathlib import Path

%cd /content/DoAnTotNghiep

DRIVE_PROJECT = "/content/drive/MyDrive/DoAnTotNghiep_output"
Path(DRIVE_PROJECT).mkdir(parents=True, exist_ok=True)
Path(f"{DRIVE_PROJECT}/checkpoints").mkdir(parents=True, exist_ok=True)
Path(f"{DRIVE_PROJECT}/results").mkdir(parents=True, exist_ok=True)

DATA_PROFILE = "top500_full_v1"
MSWC_SPLIT_MODE = "top500"
MSWC_MAX_PER_WORD = 0          # 0 = full/unlimited clips per word
MIN_CACHE_COVERAGE = 0.98

RUN_TAG_SCAF_GE2E = "edgespot_full_t4_scaf_ge2e_top500_full_v1"
RUN_TAG_SCAF = "edgespot_full_t4_scaf_top500_full_v1"

print("DRIVE_PROJECT:", DRIVE_PROJECT)
print("DATA_PROFILE:", DATA_PROFILE)
print("MSWC policy:", MSWC_SPLIT_MODE, "max_per_word=", MSWC_MAX_PER_WORD)
```

## 5. Kiem Tra Code Nhanh

```python
%cd /content/DoAnTotNghiep

TESTS = [
    "tests/test_edgespot_full.py",
    "tests/test_ge2e.py",
    "tests/test_gsc_silence_provider.py",
    "tests/test_mswc_microset.py",
]
existing = [p for p in TESTS if Path(p).exists()]
missing = [p for p in TESTS if not Path(p).exists()]
if missing:
    print("Missing optional tests:", missing)
if existing:
    !python -m pytest {" ".join(existing)} -q

!python scripts/model_report.py --family edgespot_full --tau 4
```

Dieu kien OK:

- `tests/test_edgespot_full.py` pass.
- Model report co:
  - `family: edgespot_full`
  - `tau: 4`
  - `params: 130598`
  - `output_shape: (1, 64)`

## 6. Tao Config Top500 Full

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

cfg["data"]["train_dir"] = "data/mswc_en"
cfg["data"]["mswc_dir"] = "data/mswc_en"
cfg["data"]["gsc_dir"] = "data/gsc_v2"
cfg["data"]["demand_dir"] = "data/demand"
cfg["data"]["max_per_word"] = 0       # full clips per word
cfg["data"]["val_max_per_word"] = 0   # full val clips too

cfg["training"]["epochs"] = 25
cfg["training"]["episodes_per_epoch"] = 300
cfg["training"]["n_classes"] = 30
cfg["training"]["n_samples"] = 20
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
cfg["checkpoint"]["save_every"] = 5

out = Path("/content/tier1_top500_full_colab.yaml")
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Wrote:", out)
print("CURRENT DATA PROFILE: MSWC TOP500 FULL CLIPS")
```

## 7. Tai GSC V2 Cho Evaluation

```python
%cd /content/DoAnTotNghiep
!python data/download_gsc.py
```

## 8. Preflight MSWC Top500 Full

Cell nay chi tai metadata va tao split, chua tai audio. Muc dich la xem so clip
Top500 du kien truoc khi ton units/disk.

```python
%%time
%cd /content/DoAnTotNghiep

!python data/download_mswc.py --top500-splits --max-per-word 0 --splits-only

import json
from pathlib import Path

counts = json.loads(Path("data/mswc_en/metadata/en_word_counts.json").read_text())
train_words = json.loads(Path("data/mswc_en/splits/train_words.json").read_text())
val_words = json.loads(Path("data/mswc_en/splits/val_words.json").read_text())
target_words = train_words + val_words
total_clips = sum(int(counts[w]) for w in target_words)

print("Top500 target words:", len(target_words))
print("Train words:", len(train_words), "Val words:", len(val_words))
print("Expected clips from metadata:", f"{total_clips:,}")
print("Top 10 target counts:")
for w in sorted(target_words, key=lambda x: counts[x], reverse=True)[:10]:
    print(f"  {w}: {counts[w]:,}")

# Rough WAV size estimate after 16k mono PCM conversion.
for kb in [32, 48, 64]:
    print(f"Estimated WAV at {kb}KB/clip:", f"{total_clips * kb / 1024 / 1024:.1f} GB")
```

Quyet dinh:

- Neu estimated WAV <= 120GB va `/content` con tren 160GB: co the tiep tuc.
- Neu estimated WAV qua lon, khong doi claim thanh full. Dung profile khac nhu
  `top500_mpw1000_v1` de debug, roi chay full tren may truong.

## 9. Tai/Convert MSWC Top500 Full Va Cache Len Drive

Lan dau se lau. Cell nay:

1. kiem tra Drive cache `mswc_en_wav_top500_full`;
2. neu cache hop le thi dung lai;
3. neu cache miss thi tai `en.tar.gz`, extract Top500 train+val, convert OPUS sang WAV, xoa OPUS, save cache len Drive.

```python
%%time
%cd /content/DoAnTotNghiep

import os, shutil, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

free_gb = shutil.disk_usage("/content").free / 1024**3
print(f"Free disk (/content): {free_gb:.1f} GB")
if free_gb < 150:
    print("WARNING: Top500 full may be tight. Stop here if estimated WAV is too large.")

from data.mswc_drive_cache import setup_mswc_from_drive, drive_cache_status, cache_dir_name

print("MSWC cache:", cache_dir_name(MSWC_SPLIT_MODE, MSWC_MAX_PER_WORD))
print("Policy:", MSWC_SPLIT_MODE, "max_per_word=", MSWC_MAX_PER_WORD)

from_drive_cache = setup_mswc_from_drive(
    DRIVE_PROJECT,
    split_mode=MSWC_SPLIT_MODE,
    max_per_word=MSWC_MAX_PER_WORD,
    min_train_val_coverage=MIN_CACHE_COVERAGE,
    n_cpu=min(8, os.cpu_count() or 8),
)

print("Loaded from Drive cache:", from_drive_cache)
status = drive_cache_status(DRIVE_PROJECT, MSWC_SPLIT_MODE, MSWC_MAX_PER_WORD)
print("Drive cache status:", {k: status[k] for k in [
    "n_wav", "n_word_dirs", "required_present", "required_total", "required_coverage"
]})

clips = Path("data/mswc_en/clips")
print("Local clips exists:", clips.exists())
print("Local split files:", list(Path("data/mswc_en/splits").glob("*.json")))
```

Neu download bi timeout, chay lai cell nay. Downloader co resume `.partial`.

## 10. Kiem Tra MSWC Sau Khi Cache Xong

```python
%cd /content/DoAnTotNghiep

!python scripts/mswc_data_report.py --data-dir data/mswc_en --top-n 20

from pathlib import Path
clips = Path("data/mswc_en/clips")
wav_n = sum(1 for _ in clips.rglob("*.wav")) if clips.exists() else 0
opus_n = sum(1 for _ in clips.rglob("*.opus")) if clips.exists() else 0
word_n = sum(1 for p in clips.iterdir() if p.is_dir()) if clips.exists() else 0
print("MSWC words:", word_n)
print("MSWC WAV:", f"{wav_n:,}")
print("MSWC OPUS:", f"{opus_n:,}")
assert word_n >= 490, "Top500 full should have around 500 word dirs"
assert opus_n == 0, "OPUS should be deleted after conversion"
```

Dieu kien OK:

- train words: 450;
- val words: 50;
- word dirs gan 500;
- `OPUS = 0`;
- report khong canh bao `mpw200`/capped extraction.

## 11. Tai DEMAND Noise Dataset

Neu da co `data/demand` tu session truoc thi cell nay se skip.

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

## 12. Smoke Train

Chay cell nay truoc train that. Neu smoke fail thi dung, khong train main.

```python
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_top500_full_colab.yaml \
  --data-dir data/mswc_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag smoke_edgespot_full_t4_scaf_ge2e_top500_full_v1 \
  --epochs 1 \
  --episodes 5 \
  --max-per-word 0 \
  --num-workers 2
```

Dieu kien OK:

- log co `Dataset: ... samples, 450 words`;
- log co `DataLoader: 30 classes x 20 samples x 5 episodes`;
- log co `Epoch 1/1`;
- khong loi CUDA/shape.

## 13. Train Main: EdgeSpotFull T4 SCAF+GE2E

Day la run chinh nen chay truoc.

```python
%%time
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_top500_full_colab.yaml \
  --data-dir data/mswc_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --run-tag edgespot_full_t4_scaf_ge2e_top500_full_v1 \
  --epochs 25 \
  --episodes 300 \
  --max-per-word 0 \
  --num-workers 2 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10 \
  --early-stop-patience 0
```

Neu A100/CPU doc data tot va GPU chua duoc dung nhieu, lan sau co the tang:

```text
--episodes 400
--num-workers 4
```

Khong tang worker qua cao khi data nam tren Drive symlink; nhieu worker co the lam nghen I/O.

## 14. Resume Neu Colab Disconnect

Chi dung cho cung run tag Top500 full, khong dung de resume Microset.

```python
%%time
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_top500_full_colab.yaml \
  --data-dir data/mswc_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf_ge2e \
  --resume /content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/latest.pt \
  --run-tag edgespot_full_t4_scaf_ge2e_top500_full_v1 \
  --epochs 25 \
  --episodes 300 \
  --max-per-word 0 \
  --num-workers 2 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10 \
  --early-stop-patience 0
```

Neu checkpoint da du 25 epoch, script moi se log `No additional training needed` va thoat sach.

## 15. Liet Ke Checkpoints

```python
from pathlib import Path

ckpt_dir = Path("/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1")
print("best:", ckpt_dir / "best.pt", (ckpt_dir / "best.pt").exists())
print("latest:", ckpt_dir / "latest.pt", (ckpt_dir / "latest.pt").exists())
print("epoch checkpoints:")
for p in sorted(ckpt_dir.glob("epoch_*.pt")):
    print(p)
```

## 16. Evaluate Dev 30 Runs

Chay dev30 cho checkpoint ung vien. Thuong bat dau bang `epoch_05.pt`, sau do
co the chay `best.pt` hoac cac epoch 10/15/20/25 neu ton units.

```python
%%time
%cd /content/DoAnTotNghiep

CKPT = "/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_05.pt"
OUT = "/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_top500_full_v1_epoch05_dev30"

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint "$CKPT" \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 30 \
  --gsc-query-split dev \
  --output-dir "$OUT"
```

Neu muon evaluate nhanh nhieu epoch:

```python
%%time
%cd /content/DoAnTotNghiep

import subprocess
from pathlib import Path

run_tag = "edgespot_full_t4_scaf_ge2e_top500_full_v1"
ckpt_dir = Path(f"/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/{run_tag}")
for ep in [5, 10, 15, 20, 25]:
    ckpt = ckpt_dir / f"epoch_{ep:02d}.pt"
    if not ckpt.exists():
        print("skip missing", ckpt)
        continue
    out = f"/content/drive/MyDrive/DoAnTotNghiep_output/results/{run_tag}_epoch{ep:02d}_dev30"
    print("Evaluating", ckpt)
    subprocess.run([
        "python", "scripts/evaluate_edgespot_protocol.py",
        "--checkpoint", str(ckpt),
        "--model-family", "edgespot_full",
        "--edge-tau", "4",
        "--k-shot", "10",
        "--n-runs", "30",
        "--gsc-query-split", "dev",
        "--output-dir", out,
    ], check=True)
```

Chon checkpoint bang dev30. Tieu chi uu tien:

1. `ACC@1% FAR`;
2. `ACC@5% FAR`;
3. `FRR@5% FAR` thap;
4. `Keyword ACC`;
5. `F1`.

## 17. Test 100 Runs Chi Sau Khi Khoa Checkpoint

Thay `SELECTED_CKPT` bang checkpoint thang dev30. Khong tune tiep sau khi da
nhin test100.

```python
%%time
%cd /content/DoAnTotNghiep

SELECTED_CKPT = "/content/drive/MyDrive/DoAnTotNghiep_output/checkpoints/edgespot_full_t4_scaf_ge2e_top500_full_v1/epoch_05.pt"
OUT = "/content/drive/MyDrive/DoAnTotNghiep_output/results/edgespot_full_t4_scaf_ge2e_top500_full_v1_selected_test100"

!python scripts/evaluate_edgespot_protocol.py \
  --checkpoint "$SELECTED_CKPT" \
  --model-family edgespot_full \
  --edge-tau 4 \
  --k-shot 10 \
  --n-runs 100 \
  --gsc-query-split test \
  --output-dir "$OUT"
```

## 18. Tong Hop Result JSON

```python
import glob, json
from pathlib import Path

files = sorted(glob.glob("/content/drive/MyDrive/DoAnTotNghiep_output/results/*top500_full_v1*/gsc_edgespot_exact_k10_results.json"))
print("Found result files:", len(files))

keys = [
    "open_set_acc_at_1far",
    "open_set_acc_at_5far",
    "frr_at_5far",
    "auc",
    "eer",
    "keyword_acc",
    "f1",
]

for f in files:
    print("\n===", f, "===")
    data = json.loads(Path(f).read_text())
    for k in keys:
        print(k, "=", data.get(k))
```

## 19. Optional Baseline: EdgeSpotFull T4 SCAF-only

Chi chay sau khi run SCAF+GE2E xong. Baseline nay dung de viet ablation.

```python
%%time
%cd /content/DoAnTotNghiep

!python scripts/train.py \
  --config /content/tier1_top500_full_colab.yaml \
  --data-dir data/mswc_en \
  --model-family edgespot_full \
  --edge-tau 4 \
  --loss scaf \
  --run-tag edgespot_full_t4_scaf_top500_full_v1 \
  --epochs 25 \
  --episodes 300 \
  --max-per-word 0 \
  --num-workers 2 \
  --select-by-gsc-dev \
  --gsc-dev-every 5 \
  --gsc-dev-runs 3 \
  --gsc-dev-k-shot 10 \
  --early-stop-patience 0
```

## 20. Khi Nao Dung Lai

Dung lai va bao ket qua neu gap cac tinh huong sau:

- Step 8 bao estimated WAV qua lon so voi free disk.
- Step 9 chay qua 2 gio ma van khong qua download/extract va log khong tien trien.
- Step 10 report co canh bao capped extraction.
- Smoke train khong in `Dataset: ... 450 words`.
- Loss/metric bi NaN.
- GSC-dev sau epoch 5 thap bat thuong, vi du `ACC@1% FAR < 0.75`.

## 21. Cach Claim Trong Bao Cao

Neu chay thanh cong, claim dung:

```text
MSWC English Top500 full-clips experiment.
Train: top 450 words by MSWC metadata count.
Validation: next 50 words by MSWC metadata count.
Evaluation: GSC v2 gsc_edgespot_exact, 10-shot, true silence, open-set.
```

Khong claim:

```text
Full MSWC English 38k words.
Exact EdgeSpot paper reproduction.
KD reproduction.
Streaming result.
```

