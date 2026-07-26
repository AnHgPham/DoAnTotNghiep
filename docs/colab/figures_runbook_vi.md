# Hướng dẫn bổ sung ảnh cho report (Colab + local)

Tài liệu này cho biết **ảnh nào còn thiếu** trong `report_ict2026_en.tex` và **cách tạo
chính xác từng ảnh**. Copy-paste từng cell vào Google Colab (hoặc chạy local) theo
đúng thứ tự.

## Tóm tắt: report cần đúng 1 ảnh bổ sung (demo UI)

| Ảnh | Nguồn | Tình trạng |
|---|---|---|
| `assets/scaf_collapse.png` | **Local** (`make_collapse_figure.py`) | **ĐÃ TẠO.** Vẽ từ chỉ số cuối thật cap-620 (cùng nguồn `tab:matrix`), không cần Colab/train lại. |
| `assets/demo_ui.png` | **Máy local** (chụp màn hình) | Chỉ là ảnh chụp web demo đang chạy. |

> **Cập nhật 30/06:** `scaf_collapse.png` không còn cần Colab. Vì log per-epoch của
> run SCAF+GE2E full-vocab không nằm trong repo, ta **không** vẽ đường-theo-epoch
> (sẽ phải bịa số). Thay vào đó hình mới hiển thị **dấu hiệu sụp đổ** từ chỉ số GSC
> test100 cuối cùng thật: AUC=50 (ngẫu nhiên), F1=0, keyword ACC=9,09% ($=1/11$),
> FRR=100% (từ chối tất cả), trong khi ACC@1%FAR vẫn ~69% (gây hiểu nhầm). Tạo lại
> bằng: `python docs/thesis/make_collapse_figure.py`. Phần Colab bên dưới chỉ còn
> hữu ích **nếu** bạn muốn đường-theo-epoch thật và đã có log trên Drive.

> Tất cả 18 ảnh còn lại (audio features, mel filterbank, kiến trúc DSCNN/EdgeSpot,
> training curves Microset, ranked bar, effect deltas, data saturation, DET curve,
> heatmap cap620, long-audio inference, t-SNE embedding...) **đã được sinh từ dữ liệu
> thật và đã nhúng sẵn** trong PDF/DOCX. Bạn không cần làm lại trừ khi muốn cập nhật.

---

## Colab mới tinh: cần gì & mất bao lâu (đọc trước)

Có **2 đường** để tạo `scaf_collapse.png`. Hãy ưu tiên **Cách 1**.

| | Cách 1 — Vẽ lại từ log Drive (KHUYÊN DÙNG) | Cách 2 — Train lại từ đầu |
|---|---|---|
| Điều kiện | Drive còn log run `*_scaf*` full-vocab (ngày 30/05) | Mất hết log, buộc train lại |
| Cần GPU? | **Không** (chọn runtime CPU cũng được) | Có (A100 giúp **chỉ** bước train) |
| Tải MSWC 32.5 GB? | **Không** | **Có** (bắt buộc, xem cảnh báo) |
| Thời gian (A100 Pro) | **~5 phút** | **~1.5 – 2.5 giờ** |

> **Cảnh báo quan trọng về Cách 2:** `--max-per-word 20` **không** làm giảm lượng tải.
> `data/download_mswc.py` luôn tải **trọn `en.tar.gz` 32.5 GB** rồi mới cắt clip. Các
> bước nặng nhất là **giải nén tar 6.6M file** và **convert OPUS→FLAC ~750k file** —
> cả hai đều **CPU/IO-bound, A100 không tăng tốc**. Bước train trên A100 mới nhanh.

**Trả lời ngắn cho câu hỏi của bạn:** runbook bản cũ *không* chạy được trên Colab trắng
(nó `cp` code từ Drive). Bản này đã sửa: Cách 2 **`git clone`** code nên chạy được trên
Colab trắng hoàn toàn; còn nếu Drive của bạn vẫn còn log run cũ (rất có thể) thì dùng
**Cách 1, chỉ ~5 phút**, khỏi tải data, khỏi train.

---

## PHẦN A — `scaf_collapse.png`

Mục tiêu: minh hoạ **SCAF collapse** — khi train SCAF+GE2E (trọng số 1.0) trên toàn bộ
37,387 lớp, validation AUC bị ghim ở 0.50 từ epoch ~2 (so với run Microset khoẻ mạnh).

### ▶ Cách 1 — Vẽ lại từ log đã có trên Drive (KHUYÊN DÙNG, ~5 phút, không cần GPU)

#### Cell 1.1 — Mount Drive + cài thư viện vẽ (nhẹ)

```python
from google.colab import drive
drive.mount('/content/drive')
!pip -q install tensorboard matplotlib "numpy<2.0"
```

#### Cell 1.2 — Tự dò các run có chứa SCAF và in AUC cuối để nhận diện run collapse

```python
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DRIVE = Path('/content/drive/MyDrive')
roots = [DRIVE/'DoAnTotNghiep_colab_runs', DRIVE/'DoAnTotNghiep', DRIVE]
cands = []
for base in roots:
    if base.exists():
        cands += [p for p in base.glob('**/*scaf*/runs') if p.is_dir()]
cands = sorted(set(cands))
print(f'Tìm thấy {len(cands)} run có "scaf":\n')

def last_auc(run):
    try:
        ea = EventAccumulator(str(run)); ea.Reload()
        if 'val/auc' in ea.Tags().get('scalars', []):
            vs = [e.value for e in ea.Scalars('val/auc')]
            return (round(min(vs),3), round(max(vs),3), round(vs[-1],3), len(vs))
    except Exception as e:
        return ('err', str(e)[:40])
    return None

for i, p in enumerate(cands):
    info = last_auc(p)
    # run COLLAPSE: AUC ~0.5 (hoặc ~50). run khoẻ: AUC ~0.9+
    print(f'[{i}] min/max/last/n = {info}')
    print('     ', p)
```

> Run **collapse** là run có `val/auc` quanh **0.5** (hoặc 50 nếu lưu theo %).
> Run khoẻ mạnh sẽ ~0.9+. Ghi nhớ chỉ số `[i]` của run collapse cho cell sau.

#### Cell 1.3 — Vẽ `scaf_collapse.png` từ run đã chọn

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IDX = 0                       # <-- ĐỔI thành [i] của run collapse ở Cell 1.2
RUN = cands[IDX]
ea = EventAccumulator(str(RUN)); ea.Reload()

def s(tag):
    if tag not in ea.Tags().get("scalars", []): return [], []
    ev = ea.Scalars(tag); return [e.step for e in ev], [e.value for e in ev]

ep_l, loss = s("train/loss")
ep_a, auc  = s("val/auc")
auc = np.array(auc) * (100 if (len(auc) and max(auc) <= 1.5) else 1)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.4))
a0.plot(ep_l, loss, "-o", ms=3, color="#a50f15", lw=1.8)
a0.set_title("(a) Training loss (SCAF+GE2E, full vocab)")
a0.set_xlabel("Epoch"); a0.set_ylabel("Loss"); a0.grid(True, ls=":", alpha=.4)

a1.plot(ep_a, auc, "-s", ms=4, color="#a50f15", label="val AUC")
a1.axhline(50, ls="--", color="gray", lw=1, label="chance (AUC=50)")
a1.set_ylim(45, 100)
a1.set_title("(b) Validation AUC pinned at 50% (collapse)")
a1.set_xlabel("Epoch"); a1.set_ylabel("AUC (%)"); a1.grid(True, ls=":", alpha=.4)
a1.legend(fontsize=8, loc="upper right")

fig.suptitle("SCAF+GE2E collapse at 37k-class vocabulary (weight=1.0)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path("scaf_collapse.png"); fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out.resolve())

from google.colab import files
files.download(str(out))
```

Xong → nhảy xuống mục **"Chèn ảnh vào LaTeX"**. Không cần làm Cách 2.

---

### ▶ Cách 2 — Train lại từ đầu trên Colab trắng (CHỈ khi mất hết log; ~1.5–2.5h trên A100)

> Chỉ dùng khi Cell 1.2 không tìm thấy run collapse nào trên Drive.
> Nhớ chọn runtime **A100** (Runtime → Change runtime type → A100).

#### Cell 2.1 — Lấy code bằng git clone (chạy được trên Colab trắng)

```python
%cd /content
!git clone https://github.com/AnHgPham/DoAnTotNghiep.git
%cd /content/DoAnTotNghiep
!ls
```

#### Cell 2.2 — Cài dependencies (không đụng torch của Colab)

```python
!pip -q install "numpy<2.0" pyyaml scipy soundfile scikit-learn matplotlib tensorboard tqdm requests
!apt-get -qq install -y ffmpeg >/dev/null
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
```

#### Cell 2.3 — Tải MSWC + tạo manifest cap20 (BƯỚC NẶNG NHẤT)

> Tải 32.5 GB là **bắt buộc** (không thể tránh). Đây là phần lâu nhất, A100 không giúp.
> Không cần GSC vì ta chỉ đọc `val/auc` của MSWC để thấy collapse.

```python
# (1) tải trọn en.tar.gz 32.5GB, (2) cắt 20 clip/từ cho ~37k từ, (3) xoá tar
!python data/download_mswc.py --min-clips 1 --val-fraction 0.02 --split-seed 42 --max-per-word 20 --mirror cloudflare
# convert OPUS->FLAC (~750k file; có thể 20-45 phút)
!python data/convert_opus_to_flac.py --clips-dir data/mswc_en/clips --workers 12 --delete-opus
# manifest cap20
!python data/build_mswc_file_splits.py --data-dir data/mswc_en --max-per-word 20 --output-suffix cap20_flac --source clips --overwrite
```

#### Cell 2.4 — Train ngắn để KÍCH HOẠT collapse (full vocab, SCAF weight = 1.0, BỎ QUA GSC)

```python
!python scripts/train.py \
  --config configs/default.yaml \
  --data-dir data/mswc_en \
  --model-family dscnn --feature-type mel_pcen \
  --loss scaf_ge2e \
  --scaf-weight 1.0 --arcface-scale 30 --arcface-margin 0.5 --ge2e-weight 1.0 \
  --epochs 3 --episodes 120 --n-classes 30 --n-samples 10 \
  --max-per-word 20 \
  --train-files train_files_cap20_flac.json --val-files val_files_cap20_flac.json \
  --num-workers 8 \
  --run-tag scaf_collapse_demo \
  --gsc-dev-every 999 \
  --save-every 1 --save-latest-every-epoch
```

Trong log sẽ thấy `val/auc` ~ 0.50 và in-episode accuracy ~ 0.033 ngay từ epoch 2 — đó
là collapse. (Chỉ cần 2–3 epoch là đủ để vẽ.)

#### Cell 2.5 — Vẽ `scaf_collapse.png`

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sorted(Path("checkpoints/scaf_collapse_demo/runs").glob("**/events*"))
RUN = run[0].parent if run else Path("checkpoints/scaf_collapse_demo/runs")
ea = EventAccumulator(str(RUN)); ea.Reload()

def s(tag):
    if tag not in ea.Tags().get("scalars", []): return [], []
    ev = ea.Scalars(tag); return [e.step for e in ev], [e.value for e in ev]

ep_l, loss = s("train/loss")
ep_a, auc  = s("val/auc")
auc = np.array(auc) * (100 if (len(auc) and max(auc) <= 1.5) else 1)

fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.4))
a0.plot(ep_l, loss, "-o", ms=3, color="#a50f15", lw=1.8)
a0.set_title("(a) Training loss (SCAF+GE2E, full vocab)")
a0.set_xlabel("Epoch"); a0.set_ylabel("Loss"); a0.grid(True, ls=":", alpha=.4)

a1.plot(ep_a, auc, "-s", ms=4, color="#a50f15", label="val AUC")
a1.axhline(50, ls="--", color="gray", lw=1, label="chance (AUC=50)")
a1.set_ylim(45, 100)
a1.set_title("(b) Validation AUC pinned at 50% (collapse)")
a1.set_xlabel("Epoch"); a1.set_ylabel("AUC (%)"); a1.grid(True, ls=":", alpha=.4)
a1.legend(fontsize=8, loc="upper right")

fig.suptitle("SCAF+GE2E collapse at 37k-class vocabulary (weight=1.0)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path("scaf_collapse.png"); fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out.resolve())

from google.colab import files
files.download(str(out))
```

---

### Chèn ảnh vào LaTeX

Bỏ ảnh vào `docs/thesis/assets/scaf_collapse.png` ở máy local, rồi trong
`report_ict2026_en.tex` thay khối `\TODOfig{...}` của mục **Q2** bằng:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{scaf_collapse.png}
\caption{Full-vocabulary SCAF$+$GE2E collapse: validation AUC pinned at 50\% from
epoch~2 (weight$=$1.0, scale$=$30). Contrast with the healthy Microset run.}
\label{fig:scafcollapse}
\end{figure}
```

---

## PHẦN B — `demo_ui.png` (chụp ở máy local, KHÔNG cần Colab)

```powershell
# Terminal 1 (backend)
python -m src.demo.api_server
# Terminal 2 (frontend)
cd src/demo/ui ; npm install ; npm run dev
```

Mở `http://127.0.0.1:5173/ui/`, enroll vài từ (yes/stop/...), bật phần streaming hoặc
long-audio timing, rồi chụp màn hình ở độ phân giải ~1280×720, lưu thành
`docs/thesis/assets/demo_ui.png`. Trong `.tex`, thay khối `\TODOfig{...}` ở cuối mục
**III.8 Deployment** bằng:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{demo_ui.png}
\caption{Offline web demonstrator: model switcher and long-audio timing view.}
\label{fig:demoui}
\end{figure}
```

---

## PHẦN C — Build lại PDF + DOCX sau khi thêm ảnh

```powershell
cd docs/thesis
pdflatex -interaction=nonstopmode report_ict2026_en.tex
pdflatex -interaction=nonstopmode report_ict2026_en.tex
python -c "import pypandoc; pypandoc.convert_file('report_ict2026_en.tex','docx',outputfile='report_ict2026_en.docx', extra_args=['--resource-path=assets','--toc','--number-sections'])"
```

> Lưu ý Overleaf: chỉ cần upload `report_ict2026_en.tex` + cả thư mục `assets/`.
> File đã cấu hình `\graphicspath{{assets/}}` nên ảnh tự nhận.
