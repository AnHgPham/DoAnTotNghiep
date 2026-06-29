# Ban do du an KWS

File nay dung de giam roi khi doc va phat trien project. Neu moi quay lai codebase, hay doc theo thu tu o phan "Thu tu nen doc".

## 1. Thu muc nen xem dau tien

| Duong dan | Vai tro |
|---|---|
| `scripts/train.py` | Entry point train model. Chon model family, feature type, loss, optimizer, scheduler, checkpoint. |
| `scripts/evaluate_edgespot_protocol.py` | Entry point danh gia GSC protocol, gom ACC@1%FAR, ACC@5%FAR, AUC, EER, F1. |
| `src/data/mswc_dataset.py` | Dataset + episodic loader cho MSWC/GSC. |
| `src/features/` | Feature extraction: MFCC, mel, PCEN, SpecAugment. |
| `src/models/` | Backbone va loss: DSCNN, EdgeSpot, BCResNetFS, Triplet, SCAF, GE2E. |
| `src/classifiers/open_ncm.py` | Prototype/OpenNCM scoring cho inference few-shot. |
| `src/demo/api_server.py` | FastAPI backend cho demo web. |
| `src/demo/ui/` | React/Vite UI moi. |
| `src/demo/web/` | UI legacy fallback, chi sua khi can duy tri tuong thich. |

## 2. Thu tu nen doc de hieu co che

1. `scripts/train.py`
   - Doc doan build model family.
   - Doc doan build dataset.
   - Doc doan build loss.
   - Doc doan train loop va save checkpoint.

2. `src/data/mswc_dataset.py`
   - Hieu audio duoc load, resample, pad/trim 1s.
   - Hieu `feature_type="mfcc"` va `feature_type="mel"`.
   - Hieu episodic sampler: `n_classes x n_samples x n_episodes`.

3. `src/features/mfcc.py`, `src/features/mel.py`, `src/features/pcen.py`
   - DSCNN hien dung MFCC.
   - EdgeSpot/BCResNet hien dung mel va PCEN nam trong model.

4. `src/models/dscnn.py`, `src/models/edgespot_full.py`
   - Hieu backbone bien feature thanh embedding.
   - DSCNN-L embedding 276-D.
   - EdgeSpotFull T4 embedding 64-D.

5. `src/models/prototypical.py`, `src/models/arcface.py`, `src/models/ge2e.py`
   - Triplet: anchor-positive-negative.
   - SCAF: angular margin + sub-centers.
   - GE2E: support/query centroid loss gan voi inference prototype.

6. `scripts/evaluate_edgespot_protocol.py`
   - Hieu dev/test, k-shot, n-runs.
   - Hieu vi sao ACC@1%FAR khac long-audio accuracy.

7. `src/demo/api_server.py` va `src/demo/ui/`
   - Chi doc sau khi da hieu training/evaluation.
   - Day la phan demo, khong phai metric chinh cua thesis.

## 3. Source code chinh

| Nhom | Duong dan | Ghi chu |
|---|---|---|
| Training | `scripts/train.py` | File train chinh. |
| Evaluation | `scripts/evaluate_edgespot_protocol.py`, `src/evaluation/` | Protocol va metric. |
| Data | `src/data/`, `data/download_*.py`, `data/mswc_drive_cache.py` | Loader va script chuan bi data. |
| Features | `src/features/` | MFCC, mel, PCEN, augmentation. |
| Models | `src/models/` | Backbone + metric-learning losses. |
| Demo backend | `src/demo/api_server.py` | FastAPI server. |
| Demo frontend | `src/demo/ui/` | React UI moi. |
| Legacy demo | `src/demo/web/`, `demo_quick.py`, `demo_web.py` | Giu lai de fallback/so sanh. |
| Tests | `tests/` | Unit/API tests. |

## 4. Artifact va file local

| Duong dan | Y nghia | Co nen commit? |
|---|---|---|
| `server/` | Artifact local tu Colab/server, notebook va checkpoint package. | Khong. Da ignore. |
| `checkpoints/` | Checkpoint local. | Khong. |
| `results/` | Ket qua evaluation local. | Khong, tru khi la bang ket qua da chot va duoc dua vao `reports/`. |
| `reports/microset/` | Bang ket qua Microset da khoa. | Co the commit. |
| `reports/project_status/` | Bang/tom tat sinh ra tu script. | Mac dinh khong commit. |
| `data/gsc_v2/`, `data/mswc_en/`, `data/demand/` | Dataset local. | Khong. |
| `data/test/` | Audio demo nho. Mot so file da tracked. | Chi commit file demo can thiet. |
| `data/test/generated/` | Long audio generated moi. | Khong. |
| `archive/raw_downloads/` | File zip/tar.gz tai ve. | Khong. |
| `logs/server/` | Log server/demo. | Khong. |

## 5. Cac lenh hay dung

Train smoke nhanh:

```powershell
python scripts/train.py `
  --config configs/default.yaml `
  --data-dir data/mswc_microset_en `
  --model-family edgespot_full `
  --edge-tau 4 `
  --loss scaf_ge2e `
  --run-tag smoke_edgespot_t4_scaf_ge2e `
  --epochs 2 `
  --episodes 20 `
  --max-per-word 30 `
  --num-workers 0 `
  --save-every 1 `
  --save-latest-every-epoch
```

Danh gia GSC protocol:

```powershell
python scripts/evaluate_edgespot_protocol.py `
  --checkpoint checkpoints/smoke_edgespot_t4_scaf_ge2e/latest.pt `
  --model-family edgespot_full `
  --edge-tau 4 `
  --k-shot 10 `
  --n-runs 30 `
  --gsc-query-split dev `
  --output-dir results/smoke_edgespot_t4_scaf_ge2e_dev30
```

Chay demo server:

```powershell
python -m uvicorn src.demo.api_server:app --host 127.0.0.1 --port 8000
```

Build UI moi:

```powershell
cd src/demo/ui
npm run typecheck
npm run build
```

## 6. Nguyen tac de project khong roi lai

- Source code moi de trong `src/`, `scripts/`, `tests/`, `configs/`, `docs/`.
- Dataset, checkpoint, zip, log, Colab export khong de o root.
- File long-audio demo moi sinh ra de trong `data/test/generated/`.
- Ket qua chot de trong `reports/<experiment_name>/`.
- Tai lieu thesis/report de trong `docs/`.
- Neu tao notebook/server artifact moi, de trong `server/` hoac `archive/`, khong commit truc tiep.

