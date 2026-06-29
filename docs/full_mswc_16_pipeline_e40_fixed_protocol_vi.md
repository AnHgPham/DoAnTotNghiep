# Protocol Co Dinh: Full MSWC 16 Pipeline, 40 Epoch

Muc dich: tao mot lan test chuan de so sanh 16 pipeline trong cung dieu kien, tranh thay doi epoch/episode/manifest lam ket qua khong con cong bang.

## Script

- Runner: `server/run_full_mswc_16_pipeline_manifest20_e40_fixed.sh`
- Launcher: `server/launch_full_mswc_16_pipeline_manifest20_e40_fixed.sh`
- Run ID: `full_mswc_16_pipeline_manifest20_e40_ep150_fixed`

## Cau hinh hard-coded

| Truong | Gia tri |
|---|---:|
| Dataset | Full MSWC English vocabulary, manifest20 |
| Train manifest | `data/mswc_en/splits/train_files.json` |
| Val manifest | `data/mswc_en/splits/val_files.json` |
| Train files | 527,069 |
| Val files | 10,637 |
| Epochs | 40 |
| Episodes/epoch | 150 |
| Episode batch | 30 classes x 10 samples |
| GSC checkpoint selection | dev, ACC@1%FAR |
| GSC selection cadence | every 5 epochs |
| GSC selection runs | 3 |
| Final eval | dev30 FAR1, test100 FAR1, test100 FAR5 |

## 16 pipeline

| # | Architecture | Frontend | Loss |
|---:|---|---|---|
| 1 | DSCNN-L | MFCC | Triplet |
| 2 | DSCNN-L | MFCC | SCAF |
| 3 | DSCNN-L | MFCC | GE2E |
| 4 | DSCNN-L | MFCC | SCAF+GE2E |
| 5 | DSCNN-L | PCEN | Triplet |
| 6 | DSCNN-L | PCEN | SCAF |
| 7 | DSCNN-L | PCEN | GE2E |
| 8 | DSCNN-L | PCEN | SCAF+GE2E |
| 9 | EdgeSpotFull T4 | MFCC | Triplet |
| 10 | EdgeSpotFull T4 | MFCC | SCAF |
| 11 | EdgeSpotFull T4 | MFCC | GE2E |
| 12 | EdgeSpotFull T4 | MFCC | SCAF+GE2E |
| 13 | EdgeSpotFull T4 | PCEN | Triplet |
| 14 | EdgeSpotFull T4 | PCEN | SCAF |
| 15 | EdgeSpotFull T4 | PCEN | GE2E |
| 16 | EdgeSpotFull T4 | PCEN | SCAF+GE2E |

## Nguyen tac bao cao

- Chi so sanh cac pipeline trong protocol nay voi nhau khi tat ca cung dung `40 epochs x 150 episodes`.
- Ket qua phase-1 5 epoch chi dung de screening, khong dung de ket luan final.
- Ket qua shortlist 20 epoch chi dung de chon huong train tiep, khong tron bang voi protocol 40 epoch.
- Neu server bi ngat, runner co resume tu `latest.pt` va giu `best.pt` cu bang `initial-best-metric`.
